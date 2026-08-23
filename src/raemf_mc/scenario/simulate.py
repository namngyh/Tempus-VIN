from __future__ import annotations

import math
from pathlib import Path

import torch

from raemf_mc.bayesian.torch_backend import (
    PooledPosterior,
    append_fallback_log,
    sample_joint_draws,
)
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    MSEGARCHParams,
    expected_abs_standardized_t,
    nu_from_raw,
    run_ms_egarch_recursion,
    sample_ms_egarch_draws,
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
    z = torch.randn(nu.shape, dtype=nu.dtype, device=nu.device, generator=generator)
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


def _path_chunks(n_paths: int, path_chunk_size: int | None) -> list[int]:
    """Chia n_paths thanh cac lo. `None` = mot lo duy nhat (mac dinh, giu
    nguyen hanh vi don gian nhat). Bo nho dinh cua mot lo do recursion batch
    quyet dinh: 2 tensor `(B, T, n_states)` float32, tuc
    `2 * B * T * n_states * 4` byte (B=500, T=1500, n=4 -> ~24 MB;
    B=10000, T=6264 -> ~2 GB). Dat lo khi chay quy mo nghien cuu de chan OOM.
    """
    if path_chunk_size is None or path_chunk_size >= n_paths:
        return [n_paths]
    if path_chunk_size <= 0:
        raise ValueError(f"path_chunk_size phai duong, nhan {path_chunk_size!r}")
    full, rest = divmod(n_paths, path_chunk_size)
    return [path_chunk_size] * full + ([rest] if rest else [])


def _simulate_path_chunk(
    ms_egarch_posterior: PooledPosterior,
    mu_posterior: PooledPosterior,
    centered_returns: torch.Tensor,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    n_paths: int,
    horizon: int,
    layout: MSEGARCHParamLayout,
    generator: torch.Generator | None,
    log_var_lo: torch.Tensor,
    log_var_hi: torch.Tensor,
    device: torch.device | None,
) -> tuple[torch.Tensor, int, int]:
    """Mot lo path, hoan toan vector hoa. Tra ve
    (daily_returns `(n_paths, horizon)`, so draw khong dung, so buoc cham
    bien dai bien dong)."""
    params = sample_ms_egarch_draws(ms_egarch_posterior, n_paths, layout, generator=generator)
    mu = sample_joint_draws(mu_posterior, n_paths, generator=generator)
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

    trans = transition_matrix(params.transition_logits)      # (B, n, n)
    # (B, n): mot nu cho moi (path, trang thai). Mo phong tien phai chon
    # dung nu cua trang thai HIEN TAI o moi buoc, khong phai mot nu chung.
    nu_by_state = nu_from_raw(params.nu_raw.reshape(n_paths, -1).expand(n_paths, layout.n_states))
    e_abs_z_by_state = expected_abs_standardized_t(nu_by_state)
    n_nonstationary_draws = int((params.beta.abs() >= 1.0).any(dim=-1).sum().item())

    history = run_ms_egarch_recursion(
        centered_returns, params, init_log_var, init_log_state_prob
    )
    rows = torch.arange(n_paths, device=centered_returns.device)
    state = torch.multinomial(
        torch.exp(history["log_filtered_prob"][:, -1]), 1, generator=generator
    ).squeeze(-1)                                            # (B,)
    # The standardized innovations this draw's filter actually produced on
    # the estimation window -- the support over which its alpha/gamma were
    # identified, and hence the widest |z| the news-impact term may be
    # evaluated at (see note (1) at the top of the module).
    z_impact_max = (
        centered_returns.unsqueeze(0) / torch.exp(0.5 * history["log_var_bar"])
    ).abs().max(dim=-1).values                               # (B,)
    # The variance INHERITED from the filter is subject to exactly the
    # same band as every simulated step, so it must be counted the same
    # way -- it is in fact the single most diagnostic saturation event
    # there is ("the filter saturated the fitting-time clamp and handed
    # simulation an already-absurd starting variance"), and leaving it
    # out silently under-reported precisely that pathology.
    inherited_log_var_bar = history["log_var_bar"][:, -1]     # (B,)
    n_saturated = (
        (inherited_log_var_bar < log_var_lo) | (inherited_log_var_bar > log_var_hi)
    ).sum()
    log_var_bar_prev = torch.clamp(inherited_log_var_bar, min=log_var_lo, max=log_var_hi)
    sigma_bar_prev = torch.exp(0.5 * log_var_bar_prev)
    z_prev = centered_returns[-1] / sigma_bar_prev            # (B,)

    daily_returns = torch.zeros(
        n_paths, horizon, dtype=centered_returns.dtype, device=centered_returns.device
    )
    for h in range(horizon):
        state = torch.multinomial(trans[rows, state], 1, generator=generator).squeeze(-1)
        z_impact = torch.clamp(z_prev, min=-z_impact_max, max=z_impact_max)
        raw_log_var = (
            params.omega[rows, state]
            + params.beta[rows, state] * log_var_bar_prev
            + params.alpha[rows, state] * (torch.abs(z_impact) - e_abs_z_by_state[rows, state])
            + params.gamma[rows, state] * z_impact
        )
        log_var_h = torch.clamp(raw_log_var, min=log_var_lo, max=log_var_hi)
        n_saturated = n_saturated + (
            (raw_log_var < log_var_lo) | (raw_log_var > log_var_hi)
        ).sum()
        eps_h = _sample_standardized_t(nu_by_state[rows, state], generator)
        sigma_h = torch.exp(0.5 * log_var_h)
        daily_returns[:, h] = mu[rows, state] + sigma_h * eps_h
        z_prev = eps_h
        log_var_bar_prev = log_var_h

    # Mot lan dong bo duy nhat o cuoi thay vi moi buoc -- `.item()` trong
    # vong lap la thu ep CPU cho GPU va giet throughput.
    return daily_returns, n_nonstationary_draws, int(n_saturated.item())


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
    path_chunk_size: int | None = None,
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
    stationarity region, are counted and reported to `fallback_log_path`.

    Returns CENTERED daily log returns (consistent with the
    `centered_returns` input and the mu_k posterior, which was fit in the
    same centered space) — NOT real-world returns. A caller that wants
    real-world risk metrics (e.g. for VaR/CVaR reporting) must add the
    window's own historical mean return back to this output before
    computing them; this function deliberately does not do that itself,
    to keep its contract consistent with its input.

    Caveat: the risk/regime numbers these paths produce under a fast,
    few-step ADVI config (see configs/mc_smoke.yaml) are for verifying
    pipeline correctness, not for reporting as real research-grade risk
    estimates — a meaningful fraction of such a fit's posterior draws can
    sit outside the model's own stationarity region (see the module-level
    comment above and the non-stationary-draw diagnostics this function
    reports), which this function's bounds make SAFE to simulate from but
    do not make STATISTICALLY WELL-IDENTIFIED. Full-scale numbers require a
    properly converged ADVI fit (more steps, more seeds), not just a longer
    window."""
    centered_returns = centered_returns.to(device) if device is not None else centered_returns
    init_log_var = init_log_var.to(device) if device is not None else init_log_var
    init_log_state_prob = (
        init_log_state_prob.to(device) if device is not None else init_log_state_prob
    )
    if generator is not None and device is not None and generator.device != device:
        raise ValueError(
            f"generator is on {generator.device} but device={device} was "
            f"requested — construct the generator with "
            f"torch.Generator(device=device) to match."
        )
    log_var_lo, log_var_hi = simulation_log_var_band(centered_returns, vol_band_multiple)

    # Vector hoa theo path. Truoc day moi path chay MOT recursion rieng
    # (n_paths lan lap Python, moi lan T buoc) cong mot vong lap horizon voi
    # `.item()` moi buoc. Gio toan bo path la chieu batch dan dau: dung MOT
    # recursion batch, roi `horizon` buoc vector hoa, khong con `.item()`
    # trong vong lap. So hoc tung path khong doi -- chi khac thu tu tieu thu
    # RNG (xem `sample_joint_draws`), nen ket qua khong trung bit-doi-bit voi
    # ban tuan tu cu du cung seed.
    outputs = []
    n_nonstationary_draws = 0
    n_band_saturated_steps = 0
    for chunk_n_paths in _path_chunks(n_paths, path_chunk_size):
        chunk_returns, chunk_nonstat, chunk_saturated = _simulate_path_chunk(
            ms_egarch_posterior, mu_posterior, centered_returns, init_log_var,
            init_log_state_prob, chunk_n_paths, horizon, layout, generator,
            log_var_lo, log_var_hi, device,
        )
        outputs.append(chunk_returns)
        n_nonstationary_draws += chunk_nonstat
        n_band_saturated_steps += chunk_saturated
    daily_returns = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    nonstationary_fraction = n_nonstationary_draws / max(n_paths, 1)
    # horizon + 1 bounded quantities per path: the variance inherited from the
    # filter, plus one per simulated day.
    n_banded_steps = n_paths * (horizon + 1)
    band_saturation_fraction = n_band_saturated_steps / max(n_banded_steps, 1)
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
                # Denominator of vol_band_saturation_fraction: the inherited
                # starting variance plus one per simulated day, per path.
                "n_banded_steps": n_banded_steps,
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
