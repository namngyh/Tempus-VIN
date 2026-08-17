from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from raemf_mc.bayesian.priors import (
    HierarchicalPriorConfig,
    hierarchical_normal_log_prob,
    state_shrinkage_weight,
)
from raemf_mc.bayesian.torch_backend import (
    AdviConfig,
    PooledPosterior,
    append_fallback_log,
    fit_multi_seed_advi,
    sample_joint_draw,
)

N_STATES = 4

# Fraction of (timestep, state) recursion cells allowed to sit on the
# log-variance clamp boundary before the fit is flagged. The clamp below
# prevents a NaN, but it is not free: a saturated cell has exactly zero
# gradient, so ADVI sees a flat plateau rather than an error and keeps
# reporting a clean status while learning nothing from those cells. That is
# a silent fallback in substance, so it needs to be reported like one. 5% is
# a judgement call: isolated saturation during early exploration is normal
# and self-correcting, but a persistent fraction this high means the sampled
# thetas are routinely explosive and the resulting posterior is not
# trustworthy.
CLAMP_SATURATION_REPORT_FRACTION = 0.05

# Bound on |log_var_t| enforced every recursion step. The EGARCH log-variance
# recursion has no built-in stationarity constraint (unlike GARCH, beta_k is
# free to exceed 1 in absolute value in this unconstrained ADVI parameter
# space), so a single explored theta can compound exponentially over a long
# `returns` series (T in the hundreds) and drive log_var_t far enough from 0
# that `exp(log_var_t)` underflows to exactly 0.0 or overflows to inf in
# float32. Both are silent in the forward pass but poison the backward pass:
# the recursion divides by sigma_bar_t (`z_prev = returns[t] / sigma_bar_t`)
# and by scale in the Student-t log-density, and d/dx sqrt(x) and d/dx (a/x)
# are singular at x=0, so a literal zero variance produces a NaN gradient
# even though the forward loss value can still look finite that step. This
# clamp keeps variance within [exp(-30), exp(30)] ~= [9e-14, 1e13], far
# inside float32's safe range in both directions, so every division in the
# recursion stays well-conditioned regardless of what ADVI samples.
_LOG_VAR_CLAMP = 30.0


@dataclass(frozen=True)
class MSEGARCHParamLayout:
    n_states: int = N_STATES

    @property
    def n_egarch_params(self) -> int:
        return 4 * self.n_states

    @property
    def n_transition_params(self) -> int:
        return self.n_states * (self.n_states - 1)

    @property
    def n_nu_params(self) -> int:
        return 1

    @property
    def total(self) -> int:
        return self.n_egarch_params + self.n_transition_params + self.n_nu_params


@dataclass(frozen=True)
class MSEGARCHParams:
    omega: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    transition_logits: torch.Tensor
    nu_raw: torch.Tensor


def unpack_params(
    theta: torch.Tensor, layout: MSEGARCHParamLayout = MSEGARCHParamLayout()
) -> MSEGARCHParams:
    n = layout.n_states
    idx = 0
    omega = theta[idx : idx + n]
    idx += n
    alpha = theta[idx : idx + n]
    idx += n
    beta = theta[idx : idx + n]
    idx += n
    gamma = theta[idx : idx + n]
    idx += n
    transition_logits = theta[idx : idx + n * (n - 1)].reshape(n, n - 1)
    idx += n * (n - 1)
    nu_raw = theta[idx : idx + 1].squeeze(-1)
    return MSEGARCHParams(omega, alpha, beta, gamma, transition_logits, nu_raw)


def transition_matrix(transition_logits: torch.Tensor) -> torch.Tensor:
    """Each row's simplex = softmax of [0, logits_row] — fixes one free
    reference category per row so an (n_states-1)-length logit vector maps
    onto an n_states-simplex (avoids Dirichlet, which is a poor fit for
    ADVI's Gaussian variational family)."""
    n = transition_logits.shape[0]
    zeros = torch.zeros(n, 1, dtype=transition_logits.dtype, device=transition_logits.device)
    full_logits = torch.cat([zeros, transition_logits], dim=1)
    return torch.softmax(full_logits, dim=1)


def transition_logit_prior_loc(
    n_states: int,
    stay_prob: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Prior mean for the (n, n-1) transition-logit block that implies a
    self-transition probability of `stay_prob` in every row.

    Must respect the reference-category convention in `transition_matrix`:
    row k's simplex is softmax([0, logits_k]), so column 0 is pinned at logit
    0. For row 0 the self-transition IS that pinned reference, so staying put
    means pushing every explicit logit DOWN; for rows k > 0 the self entry
    sits at index k-1 and must be pushed UP. A flat zero matrix — the old
    prior mean — is the uniform transition matrix, i.e. p_stay = 1/n.
    """
    off = (1.0 - stay_prob) / (n_states - 1)
    loc = torch.zeros(n_states, n_states - 1, dtype=dtype, device=device)
    loc[0, :] = math.log(off / stay_prob)
    for k in range(1, n_states):
        loc[k, k - 1] = math.log(stay_prob / off)
    return loc


def default_recursion_init(
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard (init_log_var, init_log_state_prob) for the recursion:
    unit initial variance in every state and a uniform initial state
    distribution. Shared so `fit_ms_egarch` and every downstream caller
    start the filter identically instead of re-deriving it by hand."""
    n = layout.n_states
    init_log_var = torch.zeros(n, dtype=dtype, device=device)
    init_log_state_prob = torch.log(
        torch.full((n,), 1.0 / n, dtype=dtype, device=device)
    )
    return init_log_var, init_log_state_prob


def nu_from_raw(nu_raw: torch.Tensor, min_nu: float = 2.05) -> torch.Tensor:
    """nu = min_nu + softplus(nu_raw), guaranteed strictly > min_nu.

    softplus(nu_raw) is mathematically positive for every finite nu_raw, but
    for very negative nu_raw it underflows below the float ULP at min_nu
    (e.g. softplus(-100) ~ 3.7e-44, while 2.05's float64 ULP is ~4.4e-16),
    so `min_nu + softplus(nu_raw)` rounds back to exactly min_nu in any
    standard floating-point precision. Clamping to the next representable
    float above min_nu enforces the intended strict inequality without
    perturbing normal-range values, where softplus is many orders above
    the ULP and the clamp is a no-op.
    """
    nu = min_nu + torch.nn.functional.softplus(nu_raw)
    min_nu_t = torch.as_tensor(min_nu, dtype=nu.dtype, device=nu.device)
    floor = torch.nextafter(min_nu_t, torch.full_like(min_nu_t, math.inf))
    return torch.clamp(nu, min=floor)


def expected_abs_standardized_t(nu: torch.Tensor) -> torch.Tensor:
    """E|Z| for Z = a unit-variance-standardized Student-t with df=nu
    (nu > 2), used as the E|z| term in the EGARCH leverage recursion."""
    log_numer = math.log(2.0) + 0.5 * torch.log(nu - 2) + torch.lgamma((nu + 1) / 2)
    log_denom = torch.log(nu - 1) + 0.5 * math.log(math.pi) + torch.lgamma(nu / 2)
    return torch.exp(log_numer - log_denom)


def _student_t_log_pdf(
    x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    z = (x - loc) / scale
    log_kernel = -0.5 * (nu + 1) * torch.log1p((z**2) / nu)
    log_norm = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * torch.tensor(math.pi))
        - torch.log(scale)
    )
    return log_norm + log_kernel


def student_t_log_pdf_with_variance(
    x: torch.Tensor, loc: torch.Tensor, variance: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    """Student-t log-density parametrized so Var(X) = variance exactly
    (for nu > 2), rather than the raw location-scale parametrization where
    Var(X) = scale^2 * nu / (nu - 2)."""
    scale = torch.sqrt(variance * (nu - 2) / nu)
    return _student_t_log_pdf(x, loc, scale, nu)


def run_ms_egarch_recursion(
    returns: torch.Tensor,
    params: MSEGARCHParams,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
) -> dict:
    """Gray's-collapsing MS-EGARCH recursion fused with a causal Hamilton
    forward filter, entirely in log-domain for numerical stability.

    `returns` must already be centered (MS-EGARCH models the innovation
    process, not the conditional mean).
    """
    T = returns.shape[0]
    n = params.omega.shape[0]
    trans = transition_matrix(params.transition_logits)
    log_trans = torch.log(trans + 1e-12)
    nu = nu_from_raw(params.nu_raw)
    e_abs_z = expected_abs_standardized_t(nu)

    log_var = torch.zeros(T, n, dtype=returns.dtype, device=returns.device)
    log_filtered = torch.zeros(T, n, dtype=returns.dtype, device=returns.device)
    log_var_bar = torch.zeros(T, dtype=returns.dtype, device=returns.device)

    log_var_bar_prev = torch.logsumexp(init_log_state_prob + init_log_var, dim=0)
    log_filt_prev = init_log_state_prob
    z_prev = torch.zeros((), dtype=returns.dtype, device=returns.device)
    total_log_lik = torch.zeros((), dtype=returns.dtype, device=returns.device)
    n_saturated = torch.zeros((), dtype=returns.dtype, device=returns.device)

    for t in range(T):
        raw_log_var_t = (
            params.omega
            + params.beta * log_var_bar_prev
            + params.alpha * (torch.abs(z_prev) - e_abs_z)
            + params.gamma * z_prev
        )
        log_var_t = torch.clamp(raw_log_var_t, min=-_LOG_VAR_CLAMP, max=_LOG_VAR_CLAMP)
        # A clamped cell has an exactly-zero gradient, so it is invisible to
        # ADVI: no NaN, no retry, no log entry, just a flat plateau. Count the
        # saturated cells so the caller can tell "the clamp quietly saved us"
        # apart from "the clamp was never needed". Detached — diagnostic only,
        # never part of the differentiated objective.
        n_saturated = n_saturated + (
            raw_log_var_t.detach().abs() >= _LOG_VAR_CLAMP
        ).sum().to(returns.dtype)
        log_var[t] = log_var_t

        # Hamilton predict step: log P(S_t=k | F_{t-1})
        log_pred = torch.logsumexp(log_trans + log_filt_prev.unsqueeze(1), dim=0)

        variance_t = torch.exp(log_var_t)
        loglik_t = student_t_log_pdf_with_variance(
            returns[t], torch.zeros_like(variance_t), variance_t, nu
        )
        log_joint_t = log_pred + loglik_t
        log_norm = torch.logsumexp(log_joint_t, dim=0)
        total_log_lik = total_log_lik + log_norm

        log_filt_t = log_joint_t - log_norm
        log_filtered[t] = log_filt_t

        log_var_bar_t = torch.logsumexp(log_filt_t + log_var_t, dim=0)
        log_var_bar[t] = log_var_bar_t

        sigma_bar_t = torch.exp(0.5 * log_var_bar_t)
        z_prev = returns[t] / sigma_bar_t
        log_var_bar_prev = log_var_bar_t
        log_filt_prev = log_filt_t

    n_cells = max(T * n, 1)
    return {
        "log_var": log_var,
        "log_filtered_prob": log_filtered,
        "log_var_bar": log_var_bar,
        "total_log_lik": total_log_lik,
        "nu": nu,
        "clamp_saturation_fraction": float(n_saturated) / n_cells,
    }


def build_ms_egarch_log_joint(
    returns: torch.Tensor,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    prior_config: HierarchicalPriorConfig,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    diagnostics: dict | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the MS-EGARCH log_joint(theta) callable consumed by the
    generic ADVI engine: log-likelihood from the Gray's-collapsing
    recursion, plus a two-layer prior on the per-state EGARCH parameters
    (hierarchical shrinkage toward a shared hyper-mean, weighted by each
    state's effective occupancy, AND a level prior on that hyper-mean),
    plus weakly-informative priors on the transition logits and nu chosen
    on the transformed scale rather than the raw one.

    `diagnostics`, if given, accumulates non-differentiable fit diagnostics
    across every evaluation (currently log-variance clamp saturation).
    """
    dtype, dev = returns.dtype, returns.device
    n_states = layout.n_states

    scale = torch.tensor(prior_config.hyper_mean_scale, dtype=dtype, device=dev)
    # Threshold scales with the window, so the shrinkage mechanism behaves
    # identically on the 500-session smoke run and the full ~6264-session one.
    shrinkage_threshold = prior_config.shrinkage_threshold(returns.shape[0])

    def _const(x: float) -> torch.Tensor:
        return torch.tensor(x, dtype=dtype, device=dev)

    # (loc, scale) level prior on each parameter family's hyper-mean. Without
    # this the hierarchical term is translation-invariant — it penalizes only
    # the spread of the four states, so adding a constant to all four beta
    # values costs nothing and beta is free to wander past 1 (explosive).
    hyper_mean_priors = (
        (_const(prior_config.omega_hyper_mean_loc), _const(prior_config.omega_hyper_mean_scale)),
        (_const(prior_config.alpha_hyper_mean_loc), _const(prior_config.alpha_hyper_mean_scale)),
        (_const(prior_config.beta_hyper_mean_loc), _const(prior_config.beta_hyper_mean_scale)),
        (_const(prior_config.gamma_hyper_mean_loc), _const(prior_config.gamma_hyper_mean_scale)),
    )
    trans_loc = transition_logit_prior_loc(
        n_states, prior_config.transition_stay_prob, dtype=dtype, device=dev
    )
    trans_scale = _const(prior_config.transition_logit_scale)
    nu_loc = _const(prior_config.nu_raw_loc)
    nu_scale = _const(prior_config.nu_raw_scale)

    def log_joint(theta: torch.Tensor) -> torch.Tensor:
        params = unpack_params(theta, layout)
        result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
        log_lik = result["total_log_lik"]

        if diagnostics is not None:
            frac = result["clamp_saturation_fraction"]
            diagnostics["max_clamp_saturation_fraction"] = max(
                diagnostics.get("max_clamp_saturation_fraction", 0.0), frac
            )
            diagnostics["n_log_joint_evals"] = diagnostics.get("n_log_joint_evals", 0) + 1

        # .detach() is deliberate: occupancy is used here as a fixed,
        # empirical-Bayes-style weighting of the prior, NOT as part of the
        # generative model being differentiated through. Left attached, the
        # prior's normalizing term becomes a function of theta and the model
        # gains a gradient incentive to starve a regime of occupancy purely to
        # lighten that regime's own prior penalty — state-killing pressure
        # that has nothing to do with fitting the data.
        effective_obs = torch.exp(result["log_filtered_prob"]).sum(dim=0).detach()
        weight = state_shrinkage_weight(effective_obs, shrinkage_threshold)

        log_prior = torch.zeros((), dtype=theta.dtype, device=theta.device)
        families = (params.omega, params.alpha, params.beta, params.gamma)
        for state_params, (loc, sc) in zip(families, hyper_mean_priors):
            hyper_mean = state_params.mean()
            log_prior = log_prior + hierarchical_normal_log_prob(
                state_params, hyper_mean, scale, weight
            )
            log_prior = log_prior + torch.distributions.Normal(loc, sc).log_prob(hyper_mean)
        log_prior = log_prior + torch.distributions.Normal(
            trans_loc, trans_scale
        ).log_prob(params.transition_logits).sum()
        log_prior = log_prior + torch.distributions.Normal(
            nu_loc, nu_scale
        ).log_prob(params.nu_raw).sum()

        return log_lik + log_prior

    return log_joint


def fit_ms_egarch(
    returns: torch.Tensor,
    advi_config: AdviConfig,
    prior_config: HierarchicalPriorConfig,
    seeds: list[int],
    device: torch.device,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    fallback_log_path: str | Path = "fallbacks.json",
) -> PooledPosterior:
    # Everything that meets theta must live on theta's device. theta is built
    # from mu/log_sigma inside the ADVI engine, which moves them to `device`,
    # so returns and every prior/init constant have to follow — otherwise
    # gpu_research.yaml dies on a device-mismatch the moment it sees CUDA.
    returns = returns.to(device)
    init_log_var, init_log_state_prob = default_recursion_init(
        layout, dtype=returns.dtype, device=device
    )
    diagnostics: dict = {}
    log_joint = build_ms_egarch_log_joint(
        returns, init_log_var, init_log_state_prob, prior_config, layout, diagnostics
    )
    init_mu = torch.zeros(layout.total, dtype=returns.dtype, device=device)
    init_log_sigma = torch.full(
        (layout.total,), -1.0, dtype=returns.dtype, device=device
    )
    posterior = fit_multi_seed_advi(
        log_joint, init_mu, init_log_sigma, advi_config, seeds, device, fallback_log_path
    )

    max_saturation = diagnostics.get("max_clamp_saturation_fraction", 0.0)
    if max_saturation > CLAMP_SATURATION_REPORT_FRACTION:
        append_fallback_log(
            {
                "reason": "log_var_clamp_saturation",
                "max_clamp_saturation_fraction": max_saturation,
                "threshold": CLAMP_SATURATION_REPORT_FRACTION,
                "n_log_joint_evals": diagnostics.get("n_log_joint_evals", 0),
                "detail": (
                    "log_var hit the +/-30 clamp in a large fraction of "
                    "(timestep, state) cells; those cells have zero gradient, "
                    "so ADVI saw a flat plateau instead of a divergence"
                ),
            },
            fallback_log_path,
        )
    return posterior


def sample_ms_egarch_draw(
    posterior: PooledPosterior,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    generator: torch.Generator | None = None,
) -> MSEGARCHParams:
    theta = sample_joint_draw(posterior, generator=generator)
    return unpack_params(theta, layout)
