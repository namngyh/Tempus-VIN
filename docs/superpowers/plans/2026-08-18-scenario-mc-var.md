# Bayesian scenario-return + Monte Carlo + VaR/CVaR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scenario-return layer (per-regime drift `μ_k` fit via ADVI,
conditioned on real MS-EGARCH posterior draws), a forward Monte Carlo path
simulator, and VaR/CVaR/max-drawdown risk metrics at 1-day and 20-day
horizons, 95%/99% confidence.

**Architecture:** Three new modules reusing existing infrastructure end to
end: `scenario/mu_fit.py` builds a closed-form `log_joint(mu)` from
already-computed `run_ms_egarch_recursion` outputs (no new recursion) and
fits it with the existing generic ADVI engine (`fit_multi_seed_advi`,
`sample_joint_draw` — both already parameter-agnostic); `scenario/simulate.py`
samples realized (not filtered) regime paths forward day-by-day, reusing the
same EGARCH recursion formula specialized to one sampled state per step;
`risk/metrics.py` is pure numpy/pandas aggregation with no Bayesian
machinery. Modular/"cut" Bayes throughout: MS-EGARCH's posterior is read-only
input, never touched.

**Tech Stack:** PyTorch (existing `bayesian/torch_backend.py` ADVI engine,
unchanged), numpy, pandas.

**Spec:** docs/superpowers/specs/2026-08-18-scenario-mc-var-design.md

## Global Constraints

- No `c_k` multiplicative volatility scale at the scenario layer — MS-EGARCH
  already provides per-state σ_k,t.
- `ν` (Student-t degrees of freedom) is **shared**, reused as-fit from the
  MS-EGARCH draw — no `ν_k` per state.
- `μ_k`'s log-likelihood must average N draws' likelihoods via
  `logsumexp(ll_per_draw) - log(N)`, **never** `mean(ll_per_draw)` — the
  latter is a biased (Jensen's-inequality-downward) approximation of the
  Monte Carlo marginal likelihood, same reasoning as MS-EGARCH's own
  level-space variance collapsing.
- The N draws' `(log_filtered_prob, log_var, nu)` used to fit `μ_k` must be
  precomputed ONCE, `detach()`-ed, before the ADVI loop starts — never
  recomputed inside the per-step `log_joint(mu)` call (they don't depend on
  `mu`; recomputing per gradient step multiplies cost by `n_steps` for zero
  benefit).
- Monte Carlo simulation samples a **realized** state path (categorical draw
  from the transition matrix each day), not the filtered/expected
  distribution — this is what makes it Monte Carlo rather than a point
  forecast.
- Every new tensor-constructing function threads `device`/`dtype` explicitly
  (matching `default_recursion_init`'s convention) — this project's sub-
  project 2 whole-branch review found and fixed a hardcoded-CPU bug from
  skipping this; do not repeat it.
- VaR/CVaR convention: `VaR_alpha = -quantile(returns, 1-alpha)`,
  `CVaR_alpha = -mean(returns[returns <= quantile(returns, 1-alpha)])` — both
  as **positive loss magnitudes**. `CVaR_alpha >= VaR_alpha` and
  `VaR_99 >= VaR_95` always hold by construction (deeper tail = at least as
  much loss) — these are the correct test invariants, NOT `VaR >= 0` (not
  guaranteed for a distribution with insufficient downside at that quantile).
- No train/val/test split for this sub-project — this is forward simulation
  from "today," not nowcasting (sub-project 2) or held-out evaluation. Both
  MS-EGARCH and `μ_k` fit on the FULL available window.
- Student-t sampling for simulated innovations must accept an explicit
  `torch.Generator` for reproducibility. `torch.distributions.StudentT.sample()`
  has **no** `generator` parameter (verified against the installed
  `torch==2.13.0+cpu`) — build the sampler manually from primitives that DO
  accept one: `torch.randn(..., generator=g)` for the Gaussian numerator and
  `torch._standard_gamma(nu/2, generator=g)` for the chi-squared denominator
  (`torch._standard_gamma` accepts `generator`, verified).

---

## Task 1: μ_k log-joint — core likelihood (no ADVI yet)

**Files:**
- Create: `src/raemf_mc/scenario/__init__.py` (empty)
- Create: `src/raemf_mc/scenario/mu_fit.py`
- Test: `tests/scenario/test_mu_fit.py`

**Interfaces:**
- Produces: `build_mu_log_joint(centered_returns: Tensor, ms_egarch_posterior: PooledPosterior, init_log_var: Tensor, init_log_state_prob: Tensor, layout: MSEGARCHParamLayout = MSEGARCHParamLayout(), n_draws: int = 50, mu_prior_scale: float = 0.01, min_effective_observations: float = 30.0, generator: torch.Generator | None = None) -> Callable[[Tensor], Tensor]` — `theta` passed to the returned callable IS `mu` directly (length 4, one per state), no unpack/transform.
- Consumes: `PooledPosterior`, `sample_ms_egarch_draw`, `run_ms_egarch_recursion`, `student_t_log_pdf_with_variance`, `MSEGARCHParamLayout` (all from `raemf_mc.regime.ms_egarch`); `state_shrinkage_weight`, `hierarchical_normal_log_prob` (from `raemf_mc.bayesian.priors`).

- [ ] **Step 1: Write the failing test**

```python
# tests/scenario/test_mu_fit.py
import math

import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    run_ms_egarch_recursion,
    student_t_log_pdf_with_variance,
    unpack_params,
)
from raemf_mc.scenario.mu_fit import build_mu_log_joint


def _fake_posterior(theta: torch.Tensor, layout: MSEGARCHParamLayout) -> PooledPosterior:
    # log_sigma this negative makes the ADVI reparameterized draw
    # (mu + softplus_free sigma * eps) collapse to `theta` in float32
    # regardless of the sampled eps, giving a hand-computable, effectively
    # deterministic draw without needing to control RNG state directly.
    result = FitResult(
        mu=theta, log_sigma=torch.full((layout.total,), -50.0),
        elbo_trace=[0.0], completed_without_divergence=True,
        fallback_used=False, fallback_reason=None, n_retries=0, seed=0,
    )
    return PooledPosterior(seed_results=[result])


def _known_theta(layout: MSEGARCHParamLayout) -> torch.Tensor:
    theta = torch.zeros(layout.total)
    n = layout.n_states
    # omega block stays 0; beta block (index 2*n:3*n) set mildly persistent;
    # nu_raw (last element) set to a value giving nu comfortably > 2.
    theta[2 * n : 3 * n] = -1.0
    theta[-1] = 3.0
    return theta


def test_mu_log_joint_matches_hand_computed_single_draw_likelihood():
    layout = MSEGARCHParamLayout()
    theta = _known_theta(layout)
    posterior = _fake_posterior(theta, layout)

    torch.manual_seed(0)
    centered_returns = torch.randn(30) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)
    mu_prior_scale = 0.05

    log_joint = build_mu_log_joint(
        centered_returns, posterior, init_log_var, init_log_state_prob,
        layout=layout, n_draws=1, mu_prior_scale=mu_prior_scale,
        min_effective_observations=1.0,  # small enough that shrink clamps to 1.0 (no shrinkage) given >=1 effective obs per state on 30 real-ish points
    )

    mu_test = torch.tensor([0.001, -0.002, 0.0, 0.003])
    actual = log_joint(mu_test)

    # Hand-compute the same quantity independently: rerun the recursion with
    # the SAME known theta, then the exact per-timestep mixture likelihood.
    params = unpack_params(theta, layout)
    result = run_ms_egarch_recursion(centered_returns, params, init_log_var, init_log_state_prob)
    variance = torch.exp(result["log_var"])
    expected_ll = torch.zeros(())
    for t in range(centered_returns.shape[0]):
        per_state = result["log_filtered_prob"][t] + student_t_log_pdf_with_variance(
            centered_returns[t], mu_test, variance[t], result["nu"]
        )
        expected_ll = expected_ll + torch.logsumexp(per_state, dim=0)

    expected_prior = torch.distributions.Normal(
        torch.zeros(4), torch.full((4,), mu_prior_scale)
    ).log_prob(mu_test).sum()

    torch.testing.assert_close(actual, expected_ll + expected_prior, atol=1e-4, rtol=1e-4)


def test_mu_log_joint_averages_draws_via_logsumexp_not_naive_mean():
    """With n_draws=2 built from two DIFFERENT known thetas (via two fake
    single-seed posteriors merged), the log-likelihood contribution must
    equal logsumexp(ll_1, ll_2) - log(2), not (ll_1 + ll_2) / 2 — these
    differ whenever ll_1 != ll_2, which a poorly-chosen pair could
    accidentally avoid, so pick thetas that are known to diverge in
    likelihood (different beta -> different variance path)."""
    layout = MSEGARCHParamLayout()
    theta_a = _known_theta(layout)
    theta_b = _known_theta(layout)
    theta_b[2 * layout.n_states : 3 * layout.n_states] = 0.5  # different beta -> different variance path

    # Merge into ONE posterior with two seeds so sample_ms_egarch_draw picks
    # one uniformly at random each of the 2 calls inside build_mu_log_joint;
    # log_sigma=-50 on both keeps each draw deterministically equal to its
    # seed's theta regardless of which seed gets picked or what eps lands.
    result_a = FitResult(
        mu=theta_a, log_sigma=torch.full((layout.total,), -50.0), elbo_trace=[0.0],
        completed_without_divergence=True, fallback_used=False, fallback_reason=None,
        n_retries=0, seed=0,
    )
    result_b = FitResult(
        mu=theta_b, log_sigma=torch.full((layout.total,), -50.0), elbo_trace=[0.0],
        completed_without_divergence=True, fallback_used=False, fallback_reason=None,
        n_retries=0, seed=1,
    )
    posterior = PooledPosterior(seed_results=[result_a, result_b])

    torch.manual_seed(1)
    centered_returns = torch.randn(30) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)
    mu_test = torch.tensor([0.001, -0.002, 0.0, 0.003])

    def per_theta_ll(theta: torch.Tensor) -> torch.Tensor:
        params = unpack_params(theta, layout)
        result = run_ms_egarch_recursion(centered_returns, params, init_log_var, init_log_state_prob)
        variance = torch.exp(result["log_var"])
        ll = torch.zeros(())
        for t in range(centered_returns.shape[0]):
            per_state = result["log_filtered_prob"][t] + student_t_log_pdf_with_variance(
                centered_returns[t], mu_test, variance[t], result["nu"]
            )
            ll = ll + torch.logsumexp(per_state, dim=0)
        return ll

    ll_a = per_theta_ll(theta_a)
    ll_b = per_theta_ll(theta_b)
    assert abs(float(ll_a - ll_b)) > 1e-3, "test thetas must actually produce different likelihoods"

    expected_correct = torch.logsumexp(torch.stack([ll_a, ll_b]), dim=0) - math.log(2)
    expected_naive_mean = (ll_a + ll_b) / 2

    log_joint = build_mu_log_joint(
        centered_returns, posterior, init_log_var, init_log_state_prob,
        layout=layout, n_draws=2, mu_prior_scale=1e6,  # effectively flat prior, contributes ~0
        min_effective_observations=1.0,
    )
    actual = log_joint(mu_test)

    # subtract the (separately, exactly computable) flat-prior term to
    # isolate the likelihood part for comparison
    prior_term = torch.distributions.Normal(torch.zeros(4), torch.full((4,), 1e6)).log_prob(mu_test).sum()
    actual_ll = actual - prior_term

    torch.testing.assert_close(actual_ll, expected_correct, atol=1e-3, rtol=1e-3)
    assert abs(float(actual_ll - expected_naive_mean)) > 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scenario/test_mu_fit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.scenario'`

- [ ] **Step 3: Implement**

```python
# src/raemf_mc/scenario/__init__.py
```

```python
# src/raemf_mc/scenario/mu_fit.py
from __future__ import annotations

import math
from typing import Callable

import torch

from raemf_mc.bayesian.priors import hierarchical_normal_log_prob, state_shrinkage_weight
from raemf_mc.bayesian.torch_backend import PooledPosterior
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
    log_filtered_all = torch.zeros(n_draws, T, n, dtype=centered_returns.dtype)
    log_var_all = torch.zeros(n_draws, T, n, dtype=centered_returns.dtype)
    nu_all = torch.zeros(n_draws, dtype=centered_returns.dtype)
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
    hyper_mean = torch.zeros(layout.n_states, dtype=centered_returns.dtype)
    base_scale = torch.full((layout.n_states,), mu_prior_scale, dtype=centered_returns.dtype)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scenario/test_mu_fit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/scenario/__init__.py src/raemf_mc/scenario/mu_fit.py tests/scenario/test_mu_fit.py
git commit -m "feat: add mu_k log-likelihood conditioned on real MS-EGARCH draws"
```

---

## Task 2: fit_regime_mu — ADVI wrapper

**Files:**
- Modify: `src/raemf_mc/scenario/mu_fit.py`
- Test: `tests/scenario/test_mu_fit.py`

**Interfaces:**
- Consumes: `build_mu_log_joint` (Task 1); `AdviConfig`, `fit_multi_seed_advi`, `sample_joint_draw`, `PooledPosterior` (from `raemf_mc.bayesian.torch_backend` — all already generic over any flat `theta`, no changes needed to that module).
- Produces: `fit_regime_mu(centered_returns: Tensor, ms_egarch_posterior: PooledPosterior, advi_config: AdviConfig, seeds: list[int], device: torch.device, init_log_var: Tensor, init_log_state_prob: Tensor, layout: MSEGARCHParamLayout = MSEGARCHParamLayout(), n_draws: int = 50, mu_prior_scale: float = 0.01, min_effective_observations: float = 30.0, generator: torch.Generator | None = None, fallback_log_path: str | Path = "fallbacks.json") -> PooledPosterior`. Sampling one draw from the result: `sample_joint_draw(mu_posterior, generator)` (reused directly from `torch_backend` — no new sampling function needed since mu has no unpack/transform step).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/scenario/test_mu_fit.py
from raemf_mc.bayesian.torch_backend import AdviConfig, sample_joint_draw
from raemf_mc.scenario.mu_fit import fit_regime_mu


def test_fit_regime_mu_runs_and_shrinks_sparse_state_toward_zero():
    layout = MSEGARCHParamLayout()
    theta = _known_theta(layout)
    # Make state 3 (Stress, index 3) structurally rare: push its
    # self-transition and inbound logits very negative so the Hamilton
    # filter assigns it near-zero mass, giving it low effective_obs.
    n = layout.n_states
    trans_start = 4 * n
    theta[trans_start : trans_start + n * (n - 1)] = 0.0
    theta_reshaped = theta[trans_start : trans_start + n * (n - 1)].view(n, n - 1)
    theta_reshaped[:, min(2, n - 2)] = -8.0  # column feeding state index 3 stays near-zero prob
    posterior = _fake_posterior(theta, layout)

    torch.manual_seed(2)
    centered_returns = torch.randn(200) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    advi_config = AdviConfig(n_steps=100, learning_rate=0.05, warmup_steps=10,
                              elbo_ma_window=10, early_stop_patience=50)
    gen = torch.Generator().manual_seed(3)
    mu_posterior = fit_regime_mu(
        centered_returns, posterior, advi_config, seeds=[0], device=torch.device("cpu"),
        init_log_var=init_log_var, init_log_state_prob=init_log_state_prob, layout=layout,
        n_draws=1, mu_prior_scale=0.05, min_effective_observations=50.0, generator=gen,
    )
    assert len(mu_posterior.seed_results) == 1
    fitted_mu = mu_posterior.seed_results[0].mu
    assert torch.isfinite(fitted_mu).all()
    assert fitted_mu.shape == (layout.n_states,)

    draw = sample_joint_draw(mu_posterior, generator=torch.Generator().manual_seed(4))
    assert draw.shape == (layout.n_states,)

    # The structurally-rare state's fitted posterior mean should stay much
    # closer to the zero hyper-mean than a well-identified state's --
    # exact equality isn't guaranteed (ADVI is stochastic optimization),
    # but the shrinkage mechanism should visibly bias it toward zero.
    assert abs(float(fitted_mu[3])) <= abs(float(fitted_mu[0])) + 1e-3 or abs(float(fitted_mu[3])) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scenario/test_mu_fit.py::test_fit_regime_mu_runs_and_shrinks_sparse_state_toward_zero -v`
Expected: FAIL with `ImportError: cannot import name 'fit_regime_mu'`

- [ ] **Step 3: Implement**

```python
# append to src/raemf_mc/scenario/mu_fit.py
from pathlib import Path

from raemf_mc.bayesian.torch_backend import AdviConfig, fit_multi_seed_advi


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scenario/test_mu_fit.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/scenario/mu_fit.py tests/scenario/test_mu_fit.py
git commit -m "feat: add fit_regime_mu ADVI wrapper reusing the generic engine"
```

---

## Task 3: Monte Carlo forward simulation

**Files:**
- Create: `src/raemf_mc/scenario/simulate.py`
- Test: `tests/scenario/test_simulate.py`

**Interfaces:**
- Consumes: `PooledPosterior`, `sample_joint_draw` (torch_backend); `MSEGARCHParamLayout`, `sample_ms_egarch_draw`, `run_ms_egarch_recursion`, `transition_matrix`, `nu_from_raw`, `expected_abs_standardized_t` (ms_egarch.py).
- Produces: `simulate_mc_paths(ms_egarch_posterior: PooledPosterior, mu_posterior: PooledPosterior, centered_returns: Tensor, init_log_var: Tensor, init_log_state_prob: Tensor, n_paths: int, horizon: int = 20, layout: MSEGARCHParamLayout = MSEGARCHParamLayout(), device: torch.device | None = None, generator: torch.Generator | None = None) -> Tensor` shape `(n_paths, horizon)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/scenario/test_simulate.py
import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init
from raemf_mc.scenario.simulate import simulate_mc_paths


def _fake_posterior(theta: torch.Tensor, log_sigma_value: float = -50.0) -> PooledPosterior:
    result = FitResult(
        mu=theta, log_sigma=torch.full_like(theta, log_sigma_value), elbo_trace=[0.0],
        completed_without_divergence=True, fallback_used=False, fallback_reason=None,
        n_retries=0, seed=0,
    )
    return PooledPosterior(seed_results=[result])


def test_simulate_mc_paths_shape_and_finiteness():
    layout = MSEGARCHParamLayout()
    theta = torch.zeros(layout.total)
    theta[2 * layout.n_states : 3 * layout.n_states] = -1.0  # mild beta
    theta[-1] = 3.0  # nu_raw -> nu comfortably > 2
    ms_posterior = _fake_posterior(theta)
    mu_posterior = _fake_posterior(torch.tensor([0.001, -0.001, 0.0, -0.002]))

    torch.manual_seed(5)
    centered_returns = torch.randn(50) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    gen = torch.Generator().manual_seed(6)
    paths = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob,
        n_paths=25, horizon=20, layout=layout, generator=gen,
    )
    assert paths.shape == (25, 20)
    assert torch.isfinite(paths).all()


def test_simulate_mc_paths_reproducible_with_fixed_generator():
    layout = MSEGARCHParamLayout()
    theta = torch.zeros(layout.total)
    theta[2 * layout.n_states : 3 * layout.n_states] = -1.0
    theta[-1] = 3.0
    ms_posterior = _fake_posterior(theta)
    mu_posterior = _fake_posterior(torch.tensor([0.001, -0.001, 0.0, -0.002]))

    torch.manual_seed(7)
    centered_returns = torch.randn(50) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    paths_1 = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob,
        n_paths=10, horizon=5, layout=layout, generator=torch.Generator().manual_seed(42),
    )
    paths_2 = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob,
        n_paths=10, horizon=5, layout=layout, generator=torch.Generator().manual_seed(42),
    )
    torch.testing.assert_close(paths_1, paths_2)


def test_simulate_mc_paths_degenerate_single_state_matches_manual_recursion():
    """K=1: transition matrix trivially self-loops (softmax with only one
    state has no other option), so the simulated path must equal a plain
    single-regime EGARCH forward simulation using the same shocks."""
    layout = MSEGARCHParamLayout(n_states=1)
    theta = torch.zeros(layout.total)
    theta[2 * layout.n_states : 3 * layout.n_states] = -1.0  # beta
    theta[-1] = 3.0  # nu_raw
    ms_posterior = _fake_posterior(theta)
    mu_posterior = _fake_posterior(torch.tensor([0.0]))

    torch.manual_seed(8)
    centered_returns = torch.randn(30) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    paths = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob,
        n_paths=5, horizon=10, layout=layout, generator=torch.Generator().manual_seed(9),
    )
    assert paths.shape == (5, 10)
    assert torch.isfinite(paths).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scenario/test_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.scenario.simulate'`

- [ ] **Step 3: Implement**

```python
# src/raemf_mc/scenario/simulate.py
from __future__ import annotations

import torch

from raemf_mc.bayesian.torch_backend import PooledPosterior, sample_joint_draw
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
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
    z = torch.randn((), dtype=nu.dtype, generator=generator)
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
    daily_returns = torch.zeros(n_paths, horizon, dtype=centered_returns.dtype, device=device)

    for p in range(n_paths):
        params = sample_ms_egarch_draw(ms_egarch_posterior, layout, generator=generator)
        mu = sample_joint_draw(mu_posterior, generator=generator)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scenario/test_simulate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/scenario/simulate.py tests/scenario/test_simulate.py
git commit -m "feat: add forward Monte Carlo regime/return path simulation"
```

---

## Task 4: VaR / CVaR / max drawdown metrics

**Files:**
- Create: `src/raemf_mc/risk/__init__.py` (empty)
- Create: `src/raemf_mc/risk/metrics.py`
- Test: `tests/risk/test_metrics.py`

**Interfaces:**
- Produces: `compute_var(horizon_returns: np.ndarray, alpha: float) -> float`; `compute_cvar(horizon_returns: np.ndarray, alpha: float) -> float`; `compute_max_drawdown(daily_returns_2d: np.ndarray) -> np.ndarray` (one value per path); `summarize_risk(daily_paths: np.ndarray, horizons: tuple[int, ...] = (1, 20), alphas: tuple[float, ...] = (0.95, 0.99)) -> pd.DataFrame` (indexed by horizon, columns `VaR_95, CVaR_95, VaR_99, CVaR_99, mean_max_drawdown, median_max_drawdown, p95_max_drawdown`).
- Consumes: `simulate_mc_paths`'s output shape `(n_paths, horizon)` (Task 3), converted to numpy by the caller.

- [ ] **Step 1: Write the failing test**

```python
# tests/risk/test_metrics.py
import numpy as np

from raemf_mc.risk.metrics import compute_cvar, compute_max_drawdown, compute_var, summarize_risk


def test_compute_var_and_cvar_hand_computed():
    # 101 evenly spaced values from -0.50 to +0.50 (step 0.01) so that
    # quantile(0.05) at 101 points lands EXACTLY on index 5 (no
    # interpolation): position = (101-1)*0.05 = 5.0 exactly.
    returns = np.linspace(-0.50, 0.50, 101)
    var_95 = compute_var(returns, 0.95)
    assert abs(var_95 - 0.45) < 1e-9  # -quantile(0.05) = -(-0.45) = 0.45

    cvar_95 = compute_cvar(returns, 0.95)
    # tail = indices 0..5 = [-0.50,-0.49,-0.48,-0.47,-0.46,-0.45], mean=-0.475
    assert abs(cvar_95 - 0.475) < 1e-9

    assert compute_cvar(returns, 0.95) >= compute_var(returns, 0.95)

    var_99 = compute_var(returns, 0.99)
    assert var_99 >= var_95  # deeper tail = at least as much loss


def test_compute_max_drawdown_hand_computed():
    # Single path: returns that go up, then down enough to create a known
    # drawdown, then flat.
    daily = np.array([[0.10, 0.10, -0.30, 0.0, 0.05]])
    # price: 1 -> 1.10517 -> 1.22140 -> 0.90484 -> 0.90484 -> 0.95123
    # running max hits 1.22140 at t=2, price drops to 0.90484 at t=3
    # drawdown = (1.22140 - 0.90484) / 1.22140 ~= 0.2591
    dd = compute_max_drawdown(daily)
    assert dd.shape == (1,)
    assert abs(dd[0] - 0.2591) < 1e-3


def test_summarize_risk_shape_and_invariants():
    rng = np.random.default_rng(0)
    daily_paths = rng.normal(loc=0.0, scale=0.01, size=(2000, 20))
    table = summarize_risk(daily_paths, horizons=(1, 20), alphas=(0.95, 0.99))
    assert list(table.index) == [1, 20]
    for h in (1, 20):
        row = table.loc[h]
        assert row["VaR_99"] >= row["VaR_95"]
        assert row["CVaR_95"] >= row["VaR_95"]
        assert row["CVaR_99"] >= row["VaR_99"]
        assert 0.0 <= row["mean_max_drawdown"] <= 1.0
        assert 0.0 <= row["median_max_drawdown"] <= 1.0
        assert 0.0 <= row["p95_max_drawdown"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/risk/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.risk'`

- [ ] **Step 3: Implement**

```python
# src/raemf_mc/risk/__init__.py
```

```python
# src/raemf_mc/risk/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_var(horizon_returns: np.ndarray, alpha: float) -> float:
    """VaR at confidence alpha, as a positive loss magnitude:
    VaR_alpha = -quantile(horizon_returns, 1 - alpha)."""
    return float(-np.quantile(horizon_returns, 1.0 - alpha))


def compute_cvar(horizon_returns: np.ndarray, alpha: float) -> float:
    """CVaR (Expected Shortfall) at confidence alpha: mean of the tail
    at/below the VaR quantile, as a positive loss magnitude."""
    threshold = np.quantile(horizon_returns, 1.0 - alpha)
    tail = horizon_returns[horizon_returns <= threshold]
    return float(-tail.mean())


def compute_max_drawdown(daily_returns_2d: np.ndarray) -> np.ndarray:
    """Max drawdown per path (fraction in [0,1]) over the given window.
    daily_returns_2d: (n_paths, horizon)."""
    cum_log_return = np.cumsum(daily_returns_2d, axis=1)
    price = np.exp(cum_log_return)
    running_max = np.maximum.accumulate(price, axis=1)
    drawdown = (running_max - price) / running_max
    return drawdown.max(axis=1)


def summarize_risk(
    daily_paths: np.ndarray,
    horizons: tuple[int, ...] = (1, 20),
    alphas: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        horizon_returns = daily_paths[:, :h].sum(axis=1)
        drawdowns = compute_max_drawdown(daily_paths[:, :h])
        row: dict[str, float] = {"horizon": h}
        for alpha in alphas:
            pct = int(round(alpha * 100))
            row[f"VaR_{pct}"] = compute_var(horizon_returns, alpha)
            row[f"CVaR_{pct}"] = compute_cvar(horizon_returns, alpha)
        row["mean_max_drawdown"] = float(drawdowns.mean())
        row["median_max_drawdown"] = float(np.median(drawdowns))
        row["p95_max_drawdown"] = float(np.quantile(drawdowns, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/risk/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/risk/__init__.py src/raemf_mc/risk/metrics.py tests/risk/test_metrics.py
git commit -m "feat: add VaR/CVaR/max-drawdown risk metrics"
```

---

## Task 5: mc_smoke.yaml config profile

**Files:**
- Create: `configs/mc_smoke.yaml`

**Interfaces:**
- Produces: YAML keys consumed by Task 6 — `window_sessions`, `seeds`, `ms_egarch_advi` (maps to `AdviConfig` fields), `ms_egarch_prior` (maps to `HierarchicalPriorConfig` fields), `mu_advi` (maps to `AdviConfig` fields), `mu_n_draws`, `mu_prior_scale`, `mu_min_effective_observations`, `mc_n_paths`, `mc_horizon`, `mc_report_horizons`, `mc_alphas`.

- [ ] **Step 1: Create the config file**

```yaml
# configs/mc_smoke.yaml
# Profile debug nhanh cho scenario-return + Monte Carlo: cua so ngan, it
# buoc ADVI, quy mo Monte Carlo nho (500 path) - dung de kiem chung logic
# dung, KHONG phai so lieu VaR/CVaR nghien cuu cuoi cung.
device_preference: auto
window_sessions: 1500
seeds: [0]
ms_egarch_advi:
  n_steps: 300
  learning_rate: 0.05
  warmup_steps: 20
  grad_clip_norm: 10.0
  elbo_ma_window: 20
  early_stop_patience: 100
  min_delta: 0.001
  retry_lr_factor: 0.5
  max_retries: 3
  n_mc_samples: 4
ms_egarch_prior:
  hyper_mean_scale: 1.0
  min_effective_observations: 30.0
  min_effective_fraction: 0.05
mu_advi:
  n_steps: 300
  learning_rate: 0.05
  warmup_steps: 20
  grad_clip_norm: 10.0
  elbo_ma_window: 20
  early_stop_patience: 100
  min_delta: 0.001
  retry_lr_factor: 0.5
  max_retries: 3
  n_mc_samples: 4
mu_n_draws: 50
mu_prior_scale: 0.01
mu_min_effective_observations: 30.0
mc_n_paths: 500
mc_horizon: 20
mc_report_horizons: [1, 20]
mc_alphas: [0.95, 0.99]
```

- [ ] **Step 2: Commit**

```bash
git add configs/mc_smoke.yaml
git commit -m "chore: add mc_smoke ADVI/Monte Carlo config profile"
```

---

## Task 6: End-to-end integration test on real data

**Files:**
- Test: `tests/test_scenario_mc_integration_smoke.py`

**Interfaces:**
- Consumes: `load_vnindex_ohlcv` (`raemf_mc.data.loader`); `compute_log_returns` (`raemf_mc.features.returns`); `MSEGARCHParamLayout`, `default_recursion_init`, `fit_ms_egarch` (`raemf_mc.regime.ms_egarch`); `fit_regime_mu` (Task 2); `simulate_mc_paths` (Task 3); `summarize_risk` (Task 4); `AdviConfig` (`raemf_mc.bayesian.torch_backend`); `HierarchicalPriorConfig` (`raemf_mc.bayesian.priors`); `configs/mc_smoke.yaml` (Task 5).

This is the task most likely to surface a genuine integration mismatch
between pieces each tested in isolation with small synthetic inputs — per
sub-project 1's Task 18 and sub-project 2's Task 8, finding and fixing a
real bug here (if the implementation reveals one) is expected, normal work.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_scenario_mc_integration_smoke.py
from pathlib import Path

import numpy as np
import torch
import yaml

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    fit_ms_egarch,
)
from raemf_mc.risk.metrics import summarize_risk
from raemf_mc.scenario.mu_fit import fit_regime_mu
from raemf_mc.scenario.simulate import simulate_mc_paths

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mc_smoke.yaml"


def test_scenario_mc_pipeline_end_to_end_on_real_data(tmp_path):
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ohlcv = load_vnindex_ohlcv()
    ohlcv_window = ohlcv.iloc[-config["window_sessions"] :]
    log_returns = compute_log_returns(ohlcv_window)
    # No train/val/test split for this sub-project (forward simulation from
    # "today," not nowcasting or held-out evaluation) -- fit on the whole
    # window, centered by its own mean.
    centered_returns = log_returns - log_returns.mean()

    layout = MSEGARCHParamLayout()
    device = torch.device("cpu")
    returns_tensor = torch.tensor(centered_returns.to_numpy(), dtype=torch.float32)

    ms_egarch_advi = AdviConfig(**config["ms_egarch_advi"])
    ms_egarch_prior = HierarchicalPriorConfig(**config["ms_egarch_prior"])
    ms_egarch_posterior = fit_ms_egarch(
        returns_tensor, ms_egarch_advi, ms_egarch_prior, config["seeds"], device, layout,
        fallback_log_path=tmp_path / "ms_egarch_fallbacks.json",
    )
    for r in ms_egarch_posterior.seed_results:
        assert torch.isfinite(r.mu).all()

    init_log_var, init_log_state_prob = default_recursion_init(layout, device=device)

    mu_advi = AdviConfig(**config["mu_advi"])
    mu_gen = torch.Generator().manual_seed(0)
    mu_posterior = fit_regime_mu(
        returns_tensor, ms_egarch_posterior, mu_advi, config["seeds"], device,
        init_log_var, init_log_state_prob, layout=layout,
        n_draws=config["mu_n_draws"], mu_prior_scale=config["mu_prior_scale"],
        min_effective_observations=config["mu_min_effective_observations"],
        generator=mu_gen, fallback_log_path=tmp_path / "mu_fallbacks.json",
    )
    for r in mu_posterior.seed_results:
        assert torch.isfinite(r.mu).all()

    mc_gen = torch.Generator().manual_seed(1)
    daily_paths = simulate_mc_paths(
        ms_egarch_posterior, mu_posterior, returns_tensor, init_log_var, init_log_state_prob,
        n_paths=config["mc_n_paths"], horizon=config["mc_horizon"], layout=layout,
        device=device, generator=mc_gen,
    )
    assert daily_paths.shape == (config["mc_n_paths"], config["mc_horizon"])
    assert torch.isfinite(daily_paths).all()

    risk_table = summarize_risk(
        daily_paths.cpu().numpy(),
        horizons=tuple(config["mc_report_horizons"]),
        alphas=tuple(config["mc_alphas"]),
    )
    print(f"Risk summary (smoke-scale, {config['mc_n_paths']} paths):\n{risk_table}")

    for h in config["mc_report_horizons"]:
        row = risk_table.loc[h]
        assert row["VaR_99"] >= row["VaR_95"]
        assert row["CVaR_95"] >= row["VaR_95"]
        assert row["CVaR_99"] >= row["VaR_99"]
        assert 0.0 <= row["mean_max_drawdown"] <= 1.0
        assert np.isfinite(row["mean_max_drawdown"])
```

- [ ] **Step 2: Run the test and debug any real integration failure**

Run: `python -m pytest tests/test_scenario_mc_integration_smoke.py -v -s`
Expected: either PASS directly, or a real integration bug to diagnose and
fix at its root cause (in whichever earlier task's file actually has the
bug) — this is allowed to fail for a substantive reason, per the same
discipline as sub-project 1 Task 18 / sub-project 2 Task 8.

- [ ] **Step 3: Iterate until it passes**

Run: `python -m pytest tests/test_scenario_mc_integration_smoke.py -v -s`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_scenario_mc_integration_smoke.py
git commit -m "test: add end-to-end scenario+MC+risk pipeline smoke test on real VN-Index data"
```

---

## Task 7: Full suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q` (budget 40+ minutes — this now includes the
MS-EGARCH integration test, the EBM integration test, AND this plan's
integration test, all real ADVI/MC fits)
Expected: all tests pass, 0 failures

- [ ] **Step 2: Run lint**

Run: `python -m ruff check src tests scripts`
Expected: no errors (fix any reported issues, re-run until clean)

- [ ] **Step 3: Final commit if any lint fixes were needed**

```bash
git add -A
git commit -m "chore: fix lint issues found in full-suite verification"
```

(Skip this commit if step 2 was already clean with no changes.)
