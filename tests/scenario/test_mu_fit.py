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


def _rare_state_theta(layout: MSEGARCHParamLayout) -> torch.Tensor:
    """`_known_theta` with state 3 (Stress, index 3) made structurally
    rare: push the transition-logit column feeding it very negative so the
    Hamilton filter assigns it near-zero mass, giving it low effective_obs
    and thus (via `state_shrinkage_weight`) a tight, ~zero-centered prior."""
    theta = _known_theta(layout)
    n = layout.n_states
    trans_start = 4 * n
    theta[trans_start : trans_start + n * (n - 1)] = 0.0
    theta_reshaped = theta[trans_start : trans_start + n * (n - 1)].view(n, n - 1)
    theta_reshaped[:, min(2, n - 2)] = -8.0  # column feeding state index 3 stays near-zero prob
    return theta


def test_fit_regime_mu_shrinks_rare_state_more_than_a_well_identified_control():
    """A/B comparison, not an absolute-magnitude threshold: fit mu TWICE
    from otherwise-identical setup (same centered_returns, seeds, advi_config,
    mu_prior_scale, min_effective_observations), differing only in the
    MS-EGARCH theta used to generate the (log_filtered_prob, log_var) draws
    that condition the fit --

      * "rare":    `_rare_state_theta` -- state 3 gets near-zero filtered mass.
      * "control": plain `_known_theta` -- the transition-logit block is left
        at all-zero, which softmaxes to an exactly uniform row every step
        (see `transition_matrix`), so all 4 states get equal ~T/4 effective
        observations and the shrinkage weight sits at its unclamped max (1.0)
        for every state, including index 3.

    `centered_returns` carries a real, nonzero drift (not pure noise) so a
    well-identified state's ADVI fit has a genuine pull away from 0 to
    demonstrate. Under `_known_theta`'s degenerate recursion (omega = alpha =
    gamma = 0 for every state, so log-variance is pinned at its t=0 fixed
    point of exactly 0 for the whole series -- see the fixed-point note in
    `test_mu_log_joint_averages_draws_via_logsumexp_not_naive_mean`), that
    variance path is IDENTICAL across states and across the rare/control
    fixtures, so filtered occupancy is driven purely by the transition
    structure, not by the data -- an apples-to-apples comparison.

    An absolute-threshold assertion (as the previous version of this test
    used) can't distinguish "the shrinkage mechanism singled out the
    structurally-rare state" from "ADVI barely moved anything in this many
    steps" -- both look like "mu_3 stayed small". Comparing state 3's fitted
    mu head-to-head against the SAME index under the control fixture (where
    it is not rare) can't pass that way: state 3 has to end up measurably
    closer to 0 under the rare-occupancy scenario than under the
    comparable-occupancy control for the assertion to hold.

    n_steps=300 (vs Task 1/2's original 100) and drift=0.03 were chosen
    empirically: at n_steps=100 both fits are dominated by the ADVI init and
    the separation is small; at n_steps=300 the control's state 3 has had
    room to pick up the drift signal while the rare state's stays pinned
    near 0, giving a consistently large (~8-10x), seed-robust margin. Total
    runtime for both fits is a few seconds -- well under the "under a
    minute" budget for synthetic-data ADVI tests.
    """
    layout = MSEGARCHParamLayout()
    n = layout.n_states

    torch.manual_seed(2)
    centered_returns = torch.randn(200) * 0.01 + 0.03  # shared, nonzero-drift signal
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    advi_config = AdviConfig(n_steps=300, learning_rate=0.05, warmup_steps=10,
                              elbo_ma_window=10, early_stop_patience=100)

    def _fit(theta: torch.Tensor) -> tuple[torch.Tensor, PooledPosterior]:
        posterior = _fake_posterior(theta, layout)
        gen = torch.Generator().manual_seed(3)
        mu_posterior = fit_regime_mu(
            centered_returns, posterior, advi_config, seeds=[0], device=torch.device("cpu"),
            init_log_var=init_log_var, init_log_state_prob=init_log_state_prob, layout=layout,
            n_draws=1, mu_prior_scale=0.05, min_effective_observations=50.0, generator=gen,
        )
        assert len(mu_posterior.seed_results) == 1
        fitted_mu = mu_posterior.seed_results[0].mu
        assert torch.isfinite(fitted_mu).all()
        assert fitted_mu.shape == (n,)
        return fitted_mu, mu_posterior

    fitted_mu_rare, mu_posterior_rare = _fit(_rare_state_theta(layout))
    fitted_mu_control, _ = _fit(_known_theta(layout))

    draw = sample_joint_draw(mu_posterior_rare, generator=torch.Generator().manual_seed(4))
    assert draw.shape == (n,)

    # Sanity check: the control's state 3 actually picked up a real, nonzero
    # pull from the drift signal -- otherwise the comparison below could
    # pass vacuously because BOTH fits stayed near 0.
    assert abs(float(fitted_mu_control[3])) > 0.008

    # The core claim: the structurally-rare state's fitted mu is pulled
    # measurably closer to the zero hyper-mean under the rare-occupancy
    # scenario than the SAME state index is under the comparable-occupancy
    # control -- a direct demonstration of the shrinkage mechanism.
    assert abs(float(fitted_mu_rare[3])) < abs(float(fitted_mu_control[3]))
