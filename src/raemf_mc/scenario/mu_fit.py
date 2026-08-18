from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch

from raemf_mc.bayesian.priors import hierarchical_normal_log_prob, state_shrinkage_weight
from raemf_mc.bayesian.torch_backend import AdviConfig, PooledPosterior, fit_multi_seed_advi
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    run_ms_egarch_recursion,
    sample_ms_egarch_draw,
    student_t_log_pdf_with_variance,
)


def _precompute_draws(
    centered_returns: torch.Tensor,
    ms_egarch_posterior: PooledPosterior,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    layout: MSEGARCHParamLayout,
    n_draws: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run n_draws independent MS-EGARCH recursions ONCE, detached from
    autograd, before any mu ADVI step — none of this depends on mu, so
    recomputing it inside the optimization loop would multiply cost by
    n_steps for no benefit. Returns stacked (n_draws, T, n_states)
    log_filtered_prob and log_var, and (n_draws,) nu."""
    T = centered_returns.shape[0]
    n = layout.n_states
    log_filtered_all = torch.zeros(
        n_draws, T, n, dtype=centered_returns.dtype, device=centered_returns.device
    )
    log_var_all = torch.zeros(
        n_draws, T, n, dtype=centered_returns.dtype, device=centered_returns.device
    )
    nu_all = torch.zeros(n_draws, dtype=centered_returns.dtype, device=centered_returns.device)
    for i in range(n_draws):
        params = sample_ms_egarch_draw(ms_egarch_posterior, layout, generator=generator)
        result = run_ms_egarch_recursion(
            centered_returns, params, init_log_var, init_log_state_prob
        )
        log_filtered_all[i] = result["log_filtered_prob"].detach()
        log_var_all[i] = result["log_var"].detach()
        nu_all[i] = result["nu"].detach()
    return log_filtered_all, log_var_all, nu_all


def build_mu_log_joint(
    centered_returns: torch.Tensor,
    ms_egarch_posterior: PooledPosterior,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    n_draws: int = 50,
    mu_prior_scale: float = 0.01,
    min_effective_observations: float = 30.0,
    generator: torch.Generator | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build log_joint(mu) for the mu_k ADVI fit. `theta` passed to the
    returned callable IS mu directly (length n_states) — mu is
    unconstrained real-valued, no unpack/transform needed.

    Conditions on n_draws real MS-EGARCH posterior draws via a
    logsumexp-averaged likelihood: log p(r|mu) ~= logsumexp_i(ll_i(mu)) -
    log(n_draws), the correct Monte Carlo approximation of the marginal
    likelihood E_theta[p(r|mu,theta)] on the likelihood scale — NOT a mean
    of log-likelihoods, which is a biased (Jensen's-inequality-downward)
    quantity. Same reasoning as MS-EGARCH's own level-space variance
    collapsing (docs/ms_egarch_design_decisions.md, decision (a)).
    """
    log_filtered_all, log_var_all, nu_all = _precompute_draws(
        centered_returns, ms_egarch_posterior, init_log_var, init_log_state_prob,
        layout, n_draws, generator,
    )
    returns_col = centered_returns.view(1, -1, 1)  # (1, T, 1)
    variance_all = torch.exp(log_var_all)  # (n_draws, T, n_states)
    nu_col = nu_all.view(-1, 1, 1)  # (n_draws, 1, 1)
    log_n = math.log(n_draws)

    # Per-state effective observation count, averaged across draws --
    # drives the shrinkage prior below (sparse regimes pulled harder toward
    # the zero hyper-mean), same mechanism sub-project 1 built for
    # omega/alpha/beta/gamma.
    effective_obs = torch.exp(log_filtered_all).sum(dim=1).mean(dim=0)  # (n_states,)
    shrink = state_shrinkage_weight(effective_obs, min_effective_observations)
    hyper_mean = torch.zeros(
        layout.n_states, dtype=centered_returns.dtype, device=centered_returns.device
    )
    base_scale = torch.full(
        (layout.n_states,),
        mu_prior_scale,
        dtype=centered_returns.dtype,
        device=centered_returns.device,
    )

    def log_joint(mu: torch.Tensor) -> torch.Tensor:
        loc = mu.view(1, 1, -1)  # (1, 1, n_states)
        per_cell_ll = student_t_log_pdf_with_variance(returns_col, loc, variance_all, nu_col)
        # mixture over state at each (draw, t): logsumexp_k[log_filtered + ll]
        ll_per_draw_t = torch.logsumexp(log_filtered_all + per_cell_ll, dim=2)  # (n_draws, T)
        ll_per_draw = ll_per_draw_t.sum(dim=1)  # (n_draws,)
        log_lik = torch.logsumexp(ll_per_draw, dim=0) - log_n
        log_prior = hierarchical_normal_log_prob(mu, hyper_mean, base_scale, shrink)
        return log_lik + log_prior

    return log_joint


def fit_regime_mu(
    centered_returns: torch.Tensor,
    ms_egarch_posterior: PooledPosterior,
    advi_config: AdviConfig,
    seeds: list[int],
    device: torch.device,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    n_draws: int = 50,
    mu_prior_scale: float = 0.01,
    min_effective_observations: float = 30.0,
    generator: torch.Generator | None = None,
    fallback_log_path: str | Path = "fallbacks.json",
) -> PooledPosterior:
    """Fit mu_k's posterior via the shared generic ADVI engine, conditioned
    on n_draws real MS-EGARCH posterior draws (see build_mu_log_joint)."""
    centered_returns = centered_returns.to(device)
    init_log_var = init_log_var.to(device)
    init_log_state_prob = init_log_state_prob.to(device)
    log_joint = build_mu_log_joint(
        centered_returns, ms_egarch_posterior, init_log_var, init_log_state_prob,
        layout=layout, n_draws=n_draws, mu_prior_scale=mu_prior_scale,
        min_effective_observations=min_effective_observations, generator=generator,
    )
    init_mu = torch.zeros(layout.n_states, dtype=centered_returns.dtype, device=device)
    init_log_sigma = torch.full(
        (layout.n_states,), -1.0, dtype=centered_returns.dtype, device=device
    )
    return fit_multi_seed_advi(
        log_joint, init_mu, init_log_sigma, advi_config, seeds, device, fallback_log_path
    )
