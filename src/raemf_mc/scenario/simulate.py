from __future__ import annotations

import torch

from raemf_mc.bayesian.torch_backend import PooledPosterior, sample_joint_draw
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    MSEGARCHParams,
    expected_abs_standardized_t,
    nu_from_raw,
    run_ms_egarch_recursion,
    sample_ms_egarch_draw,
    transition_matrix,
)

_LOG_VAR_CLAMP = 30.0  # same bound as ms_egarch.py's recursion


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
) -> torch.Tensor:
    """Simulate n_paths independent Monte Carlo return paths of length
    horizon. Each path draws its own MS-EGARCH theta and mu vector
    (independent of every other path, and of whatever draws were used to
    fit mu), holds both fixed for the whole path, samples a REALIZED
    regime path (not the filtered/expected distribution) day by day, and
    returns daily log returns. Shape (n_paths, horizon)."""
    centered_returns = centered_returns.to(device) if device is not None else centered_returns
    init_log_var = init_log_var.to(device) if device is not None else init_log_var
    init_log_state_prob = (
        init_log_state_prob.to(device) if device is not None else init_log_state_prob
    )
    daily_returns = torch.zeros(
        n_paths, horizon, dtype=centered_returns.dtype, device=centered_returns.device
    )

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

        history = run_ms_egarch_recursion(
            centered_returns, params, init_log_var, init_log_state_prob
        )
        state_dist = torch.exp(history["log_filtered_prob"][-1])
        state = int(torch.multinomial(state_dist, 1, generator=generator).item())
        log_var_bar_prev = history["log_var_bar"][-1]
        sigma_bar_prev = torch.exp(0.5 * log_var_bar_prev)
        z_prev = centered_returns[-1] / sigma_bar_prev

        for h in range(horizon):
            state = int(torch.multinomial(trans[state], 1, generator=generator).item())
            raw_log_var = (
                params.omega[state]
                + params.beta[state] * log_var_bar_prev
                + params.alpha[state] * (torch.abs(z_prev) - e_abs_z)
                + params.gamma[state] * z_prev
            )
            log_var_h = torch.clamp(raw_log_var, min=-_LOG_VAR_CLAMP, max=_LOG_VAR_CLAMP)
            eps_h = _sample_standardized_t(nu, generator)
            sigma_h = torch.exp(0.5 * log_var_h)
            r_h = mu[state] + sigma_h * eps_h
            daily_returns[p, h] = r_h
            z_prev = eps_h
            log_var_bar_prev = log_var_h

    return daily_returns
