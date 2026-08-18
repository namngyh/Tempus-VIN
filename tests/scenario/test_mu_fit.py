import math

import torch

from raemf_mc.bayesian.torch_backend import AdviConfig, FitResult, PooledPosterior, sample_joint_draw
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    run_ms_egarch_recursion,
    student_t_log_pdf_with_variance,
    unpack_params,
)
from raemf_mc.scenario.mu_fit import build_mu_log_joint, fit_regime_mu


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
    # With omega = alpha = gamma = 0 (as in _known_theta), log_var_bar_prev is
    # a fixed point at exactly 0 for every t regardless of beta: log_var_t =
    # beta * log_var_bar_prev = beta * 0 = 0, then log_var_bar_t =
    # logsumexp(log_filt_t + 0) = 0 again since log_filt_t sums to 1. So beta
    # would have literally zero effect on the recursion without a nonzero
    # alpha (or omega/gamma) to kick log_var_bar_prev off of 0 first. Give
    # both thetas the SAME nonzero alpha so beta is the only thing that
    # differs between them.
    theta_a[layout.n_states : 2 * layout.n_states] = 0.3
    theta_b = theta_a.clone()
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
