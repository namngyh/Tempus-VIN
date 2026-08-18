from __future__ import annotations

import math
from pathlib import Path

import torch

from raemf_mc.bayesian.torch_backend import (
    PooledPosterior,
    append_fallback_log,
    sample_joint_draw,
)
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    MSEGARCHParams,
    expected_abs_standardized_t,
    nu_from_raw,
    run_ms_egarch_recursion,
    sample_ms_egarch_draw,
    transition_matrix,
)

# ---------------------------------------------------------------------------
# Why forward simulation needs its OWN bounds instead of reusing ms_egarch.py's
# `_LOG_VAR_CLAMP = 30`.
#
# That clamp is a GRADIENT-SAFETY device (see its own comment): it exists so
# `exp(log_var)` cannot underflow to 0.0 or overflow to inf in float32 and
# hand ADVI a NaN gradient. It is harmless during FITTING because the filter
# is anchored by real data at every step — it divides the OBSERVED return
# back out (`z_prev = returns[t] / sigma_bar_t`), so the standardized
# innovation stays in a normal range, and any theta that pushes the variance
# somewhere silly is punished immediately by the likelihood.
#
# Forward simulation has neither anchor, and reusing a +/-30 bound there is
# not conservative, it is meaningless: log_var = 30 is a daily volatility of
# exp(15) ~ 3.3 MILLION (a several-million-percent daily move). The bound can
# therefore never catch anything economically wrong; it only prevents a
# single-step inf.
#
# It has to catch something, because the simulated variance process is genuinely
# divergent. Forward simulation draws its own innovation from the model's
# Student-t and feeds it straight back into the EGARCH news-impact term, so
# log_var_{h+1} contains `alpha * |eps_h|` and the volatility it implies is
# `exp(log_var/2)`. `E[exp(alpha * |eps|)]` does not exist when `eps` is
# Student-t: the t has power-law tails, so its moment generating function is
# infinite everywhere, and the simulated conditional variance therefore has NO
# finite moment of any order. This is precisely the moment condition Nelson
# (1991) imposes on EGARCH innovations — he requires tails no heavier than
# exponential (GED with shape >= 1) so that E[sigma_t^2] exists; a Student-t
# innovation violates it. The consequence is not theoretical: one tail draw of
# |eps| (near-certain somewhere in n_paths * horizon draws once the fitted nu
# sits near its 2.05 floor) multiplies the next day's variance by
# exp(alpha * |eps|), the beta ~ 0.9 persistence then HOLDS that level for the
# rest of the path, and the cumulative log return runs into the hundreds —
# which overflows float32 `exp()` in the downstream drawdown calculation and
# poisons the whole risk table.
#
# The two bounds below replace the borrowed numerical clamp with bounds that
# are actually about the simulated economics.
# ---------------------------------------------------------------------------

# (1) Range over which the news-impact term is EXTRAPOLATED.
#
# `alpha * (|z| - E|z|) + gamma * z` is a LINEAR function of the standardized
# innovation, and it was identified on exactly one sample of standardized
# innovations: the `z_t = returns[t] / sigma_bar_t` this draw's own filter
# produced on the estimation window. Evaluating that linear fit at a |z|
# several times beyond anything in that sample is unsupported extrapolation,
# and per the note above it is the specific term that destroys the variance
# process's moments. So the innovation is winsorized to the range the filter
# actually saw, per draw, before it enters the variance recursion.
#
# Deliberately NOT applied to `eps_h` where it generates the return itself:
# the fat tail of the return distribution is the thing a VaR/CVaR model exists
# to measure and must never be censored. Only the variance FEEDBACK loop is
# bounded.

# (2) Economic band on the simulated conditional volatility.
#
# sigma_h is confined to [vol_window / M, vol_window * M], where vol_window is
# the realized volatility of the estimation window the draw was fit on. M is
# read off the VN-Index record rather than picked: across the full ~6260-session
# history the most volatile 20-session stretch ran at 4.07x the full-sample
# daily volatility (5.9%/day, the 2008 crash) and the calmest at 0.10x. M = 10
# therefore leaves ~2.5x headroom beyond the most violent volatility regime ever
# recorded, and still admits the calmest one, so it cannot censor any
# economically plausible scenario — while a simulated day outside that band is
# not a tail scenario at all, it is a numerical artifact.
#
# This band is applied to the state INHERITED from the filter as well as to
# every simulated step. That matters because the ADVI mean-field Gaussian is
# unconstrained and leaks mass outside the model's own |beta| < 1 stationarity
# region (Nelson 1991: the EGARCH log-variance AR(1) is strictly stationary iff
# |beta| < 1); such a draw compounds over a 1500-session filtering window until
# it sits on ms_egarch's +30 clamp, and forward simulation would otherwise start
# from that already-absurd variance.
SIM_VOL_BAND_MULTIPLE = 10.0

# Fractions above which the run's diagnostics get written to the fallback log.
# Same reasoning as ms_egarch.py's CLAMP_SATURATION_REPORT_FRACTION: the bounds
# above keep the output sane, but a run where they bind constantly is telling
# us the posterior being simulated from is unhealthy, and that must be reported
# rather than silently absorbed. Non-stationary draws are counted but NOT
# rejected — dropping them would silently retruncate the posterior, which is a
# modelling decision, not a simulator's call.
NONSTATIONARY_DRAW_REPORT_FRACTION = 0.05
VOL_BAND_SATURATION_REPORT_FRACTION = 0.05


def _sample_standardized_t(nu: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """One draw from a unit-variance-standardized Student-t(nu), built
    manually from primitives that accept an explicit generator --
    torch.distributions.StudentT.sample() has NO generator parameter
    (verified against torch==2.13.0+cpu), so it cannot be used here.
    T = Z / sqrt(V/nu), Z~N(0,1), V~ChiSquared(nu)=2*Gamma(nu/2,rate=1);
    then rescaled by sqrt((nu-2)/nu) to make Var(T)=1 exactly (matching
    student_t_log_pdf_with_variance's own scale convention)."""
    z = torch.randn((), dtype=nu.dtype, device=nu.device, generator=generator)
    chi2 = 2.0 * torch._standard_gamma(nu / 2, generator=generator)
    raw_t = z / torch.sqrt(chi2 / nu)
    scale_at_unit_variance = torch.sqrt((nu - 2) / nu)
    return raw_t * scale_at_unit_variance


def simulation_log_var_band(
    centered_returns: torch.Tensor, vol_band_multiple: float = SIM_VOL_BAND_MULTIPLE
) -> tuple[torch.Tensor, torch.Tensor]:
    """(lo, hi) bounds on simulated log-VARIANCE implied by allowing the
    simulated daily volatility to sit anywhere within a factor
    `vol_band_multiple` of the estimation window's own realized volatility.
    The half-width is 2*log(M) rather than log(M) because the band is stated
    in volatility but enforced in log-variance."""
    log_var_window = torch.log(centered_returns.var())
    half_width = 2.0 * math.log(vol_band_multiple)
    return log_var_window - half_width, log_var_window + half_width


def simulate_mc_paths(
    ms_egarch_posterior: PooledPosterior,
    mu_posterior: PooledPosterior,
    centered_returns: torch.Tensor,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    n_paths: int,
    horizon: int = 20,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    vol_band_multiple: float = SIM_VOL_BAND_MULTIPLE,
    fallback_log_path: str | Path = "fallbacks.json",
) -> torch.Tensor:
    """Simulate n_paths independent Monte Carlo return paths of length
    horizon. Each path draws its own MS-EGARCH theta and mu vector
    (independent of every other path, and of whatever draws were used to
    fit mu), holds both fixed for the whole path, samples a REALIZED
    regime path (not the filtered/expected distribution) day by day, and
    returns daily log returns. Shape (n_paths, horizon).

    The simulated volatility process is kept inside the economically
    meaningful band described at the top of this module; how often that band
    binds, and how many sampled draws fell outside the model's own
    stationarity region, are counted and reported to `fallback_log_path`."""
    centered_returns = centered_returns.to(device) if device is not None else centered_returns
    init_log_var = init_log_var.to(device) if device is not None else init_log_var
    init_log_state_prob = (
        init_log_state_prob.to(device) if device is not None else init_log_state_prob
    )
    daily_returns = torch.zeros(
        n_paths, horizon, dtype=centered_returns.dtype, device=centered_returns.device
    )
    log_var_lo, log_var_hi = simulation_log_var_band(centered_returns, vol_band_multiple)

    n_nonstationary_draws = 0
    n_band_saturated_steps = 0

    for p in range(n_paths):
        params = sample_ms_egarch_draw(ms_egarch_posterior, layout, generator=generator)
        mu = sample_joint_draw(mu_posterior, generator=generator)
        if device is not None:
            params = MSEGARCHParams(
                omega=params.omega.to(device),
                alpha=params.alpha.to(device),
                beta=params.beta.to(device),
                gamma=params.gamma.to(device),
                transition_logits=params.transition_logits.to(device),
                nu_raw=params.nu_raw.to(device),
            )
            mu = mu.to(device)
        trans = transition_matrix(params.transition_logits)
        nu = nu_from_raw(params.nu_raw)
        e_abs_z = expected_abs_standardized_t(nu)
        if bool((params.beta.abs() >= 1.0).any()):
            n_nonstationary_draws += 1

        history = run_ms_egarch_recursion(
            centered_returns, params, init_log_var, init_log_state_prob
        )
        state_dist = torch.exp(history["log_filtered_prob"][-1])
        state = int(torch.multinomial(state_dist, 1, generator=generator).item())
        # The standardized innovations this draw's filter actually produced on
        # the estimation window -- the support over which its alpha/gamma were
        # identified, and hence the widest |z| the news-impact term may be
        # evaluated at (see note (1) at the top of the module).
        z_impact_max = (centered_returns / torch.exp(0.5 * history["log_var_bar"])).abs().max()
        log_var_bar_prev = torch.clamp(
            history["log_var_bar"][-1], min=log_var_lo, max=log_var_hi
        )
        sigma_bar_prev = torch.exp(0.5 * log_var_bar_prev)
        z_prev = centered_returns[-1] / sigma_bar_prev

        for h in range(horizon):
            state = int(torch.multinomial(trans[state], 1, generator=generator).item())
            z_impact = torch.clamp(z_prev, min=-z_impact_max, max=z_impact_max)
            raw_log_var = (
                params.omega[state]
                + params.beta[state] * log_var_bar_prev
                + params.alpha[state] * (torch.abs(z_impact) - e_abs_z)
                + params.gamma[state] * z_impact
            )
            log_var_h = torch.clamp(raw_log_var, min=log_var_lo, max=log_var_hi)
            if bool((raw_log_var < log_var_lo) | (raw_log_var > log_var_hi)):
                n_band_saturated_steps += 1
            eps_h = _sample_standardized_t(nu, generator)
            sigma_h = torch.exp(0.5 * log_var_h)
            r_h = mu[state] + sigma_h * eps_h
            daily_returns[p, h] = r_h
            z_prev = eps_h
            log_var_bar_prev = log_var_h

    nonstationary_fraction = n_nonstationary_draws / max(n_paths, 1)
    band_saturation_fraction = n_band_saturated_steps / max(n_paths * horizon, 1)
    if (
        nonstationary_fraction > NONSTATIONARY_DRAW_REPORT_FRACTION
        or band_saturation_fraction > VOL_BAND_SATURATION_REPORT_FRACTION
    ):
        append_fallback_log(
            {
                "reason": "mc_simulation_bounds_engaged",
                "nonstationary_draw_fraction": nonstationary_fraction,
                "nonstationary_draw_threshold": NONSTATIONARY_DRAW_REPORT_FRACTION,
                "vol_band_saturation_fraction": band_saturation_fraction,
                "vol_band_saturation_threshold": VOL_BAND_SATURATION_REPORT_FRACTION,
                "vol_band_multiple": vol_band_multiple,
                "n_paths": n_paths,
                "horizon": horizon,
                "detail": (
                    "forward Monte Carlo needed its economic volatility band, "
                    "and/or sampled MS-EGARCH draws with |beta| >= 1 (outside "
                    "the model's stationarity region). The simulated paths are "
                    "bounded and usable, but the posterior they came from is "
                    "not well identified"
                ),
            },
            fallback_log_path,
        )
    return daily_returns
