import math

import torch
from raemf_mc.regime.ms_egarch import (
    default_recursion_init,
    transition_logit_prior_loc,
    transition_matrix,
    MSEGARCHParamLayout,
    MSEGARCHParams,
    build_ms_egarch_log_joint,
    fit_ms_egarch,
    sample_ms_egarch_draw,
)
from raemf_mc.bayesian.torch_backend import AdviConfig, PooledPosterior
from raemf_mc.bayesian.priors import HierarchicalPriorConfig


def _small_returns(T=40, seed=0):
    torch.manual_seed(seed)
    return torch.randn(T) * 0.01


def test_build_log_joint_returns_finite_scalar_for_valid_theta():
    layout = MSEGARCHParamLayout()
    returns = _small_returns()
    n = layout.n_states
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 0.25))
    prior_config = HierarchicalPriorConfig()
    log_joint = build_ms_egarch_log_joint(
        returns, init_log_var, init_log_state_prob, prior_config, layout
    )
    theta = torch.zeros(layout.total)
    value = log_joint(theta)
    assert value.shape == ()
    assert torch.isfinite(value)


def test_fit_ms_egarch_runs_end_to_end_on_small_window(tmp_path):
    returns = _small_returns(T=30)
    layout = MSEGARCHParamLayout()
    advi_config = AdviConfig(n_steps=15, learning_rate=0.02, warmup_steps=2,
                               elbo_ma_window=5, early_stop_patience=15, n_mc_samples=2)
    prior_config = HierarchicalPriorConfig()
    posterior = fit_ms_egarch(
        returns, advi_config, prior_config, seeds=[0, 1],
        device=torch.device("cpu"), layout=layout,
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert isinstance(posterior, PooledPosterior)
    assert len(posterior.seed_results) == 2
    for r in posterior.seed_results:
        assert r.mu.shape == (layout.total,)
        assert torch.isfinite(r.mu).all()


def test_sample_ms_egarch_draw_returns_structured_params_with_fixed_generator(tmp_path):
    returns = _small_returns(T=25)
    layout = MSEGARCHParamLayout()
    advi_config = AdviConfig(n_steps=10, learning_rate=0.02, warmup_steps=1,
                               elbo_ma_window=3, early_stop_patience=10, n_mc_samples=2)
    prior_config = HierarchicalPriorConfig()
    posterior = fit_ms_egarch(
        returns, advi_config, prior_config, seeds=[0], device=torch.device("cpu"),
        layout=layout, fallback_log_path=tmp_path / "fallbacks.json",
    )
    gen1 = torch.Generator().manual_seed(7)
    gen2 = torch.Generator().manual_seed(7)
    draw1 = sample_ms_egarch_draw(posterior, layout, generator=gen1)
    draw2 = sample_ms_egarch_draw(posterior, layout, generator=gen2)
    assert isinstance(draw1, MSEGARCHParams)
    assert torch.equal(draw1.omega, draw2.omega)
    assert draw1.omega.shape == (4,)
    assert draw1.transition_logits.shape == (4, 3)


def _log_joint_value(prior_config, beta_value, layout, returns):
    """Evaluate log_joint at a fixed theta. The likelihood term does not
    depend on prior_config at all, so differences across configs at the same
    theta isolate the prior exactly."""
    init_log_var, init_log_state_prob = default_recursion_init(layout)
    theta = torch.zeros(layout.total)
    n = layout.n_states
    theta[2 * n : 3 * n] = beta_value  # the beta block
    log_joint = build_ms_egarch_log_joint(
        returns, init_log_var, init_log_state_prob, prior_config, layout
    )
    return log_joint(theta)


def test_level_prior_penalizes_explosive_beta():
    """The hierarchical term alone is translation-invariant: it penalizes the
    SPREAD of the four states' beta, so adding a constant to all four costs
    nothing and beta > 1 (explosive) is unconstrained. The level prior on the
    hyper-mean is what supplies that missing force."""
    layout = MSEGARCHParamLayout()
    returns = _small_returns(T=30)
    with_level = HierarchicalPriorConfig()
    without_level = HierarchicalPriorConfig(beta_hyper_mean_scale=1e6)

    stationary = _log_joint_value(with_level, 0.9, layout, returns) - _log_joint_value(
        without_level, 0.9, layout, returns
    )
    explosive = _log_joint_value(with_level, 1.6, layout, returns) - _log_joint_value(
        without_level, 1.6, layout, returns
    )
    assert explosive < stationary


def test_nu_prior_makes_realistic_degrees_of_freedom_plausible():
    """nu = 2.05 + softplus(nu_raw), so a Normal(0, 1) prior on nu_raw — the
    obvious choice — confines nu below ~4.2 and effectively mandates
    near-infinite kurtosis. Realistic daily-index df of 5-15 must be
    reachable under the prior."""
    from torch.distributions import Normal

    config = HierarchicalPriorConfig()
    prior = Normal(config.nu_raw_loc, config.nu_raw_scale)
    # inverse of nu = 2.05 + softplus(nu_raw)
    def raw_for(nu):
        return math.log(math.exp(nu - 2.05) - 1.0)

    p_above_5 = 1.0 - prior.cdf(torch.tensor(raw_for(5.0)))
    p_above_15 = 1.0 - prior.cdf(torch.tensor(raw_for(15.0)))
    assert float(p_above_5) > 0.5
    assert float(p_above_15) > 0.02
    # the old Normal(0, 1) put nu > 5 essentially out of reach
    old = Normal(0.0, 1.0)
    assert float(1.0 - old.cdf(torch.tensor(raw_for(5.0)))) < 0.01


def test_transition_logit_prior_centers_on_a_sticky_regime():
    """Zero logits mean a UNIFORM transition row (p_stay = 0.25), so a
    Normal(0, 1) prior put a realistic sticky regime ~4 sd out."""
    config = HierarchicalPriorConfig()
    loc = transition_logit_prior_loc(4, config.transition_stay_prob)
    implied = transition_matrix(loc)
    assert torch.allclose(
        torch.diagonal(implied),
        torch.full((4,), config.transition_stay_prob),
        atol=1e-5,
    )
    assert torch.allclose(implied.sum(dim=1), torch.ones(4), atol=1e-6)
    # a p_stay = 0.95 row is now well within one prior sd
    sticky = transition_logit_prior_loc(4, 0.95)
    assert float((sticky - loc).abs().max()) < config.transition_logit_scale


def test_effective_obs_is_detached_from_the_gradient_path():
    """Occupancy weights the prior as a fixed empirical-Bayes quantity. Left
    attached, the prior becomes a function of theta and the model gains a
    gradient incentive to starve a regime purely to lighten its own prior
    penalty."""
    layout = MSEGARCHParamLayout()
    returns = _small_returns(T=25)
    init_log_var, init_log_state_prob = default_recursion_init(layout)
    log_joint = build_ms_egarch_log_joint(
        returns, init_log_var, init_log_state_prob, HierarchicalPriorConfig(), layout
    )
    theta = torch.zeros(layout.total, requires_grad=True)
    value = log_joint(theta)
    grad = torch.autograd.grad(value, theta)[0]
    assert torch.isfinite(grad).all()

    from raemf_mc.regime.ms_egarch import run_ms_egarch_recursion, unpack_params

    params = unpack_params(theta, layout)
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
    effective_obs = torch.exp(result["log_filtered_prob"]).sum(dim=0).detach()
    assert effective_obs.requires_grad is False
    assert torch.allclose(effective_obs.sum(), torch.tensor(float(returns.shape[0])), atol=1e-3)


def test_fit_ms_egarch_moves_returns_to_the_requested_device(tmp_path):
    """Every tensor meeting theta must live on theta's device; a CPU-resident
    `returns` or prior constant is a device-mismatch crash on CUDA."""
    returns = _small_returns(T=25)
    layout = MSEGARCHParamLayout()
    advi_config = AdviConfig(n_steps=5, learning_rate=0.02, warmup_steps=1,
                             elbo_ma_window=3, early_stop_patience=5, n_mc_samples=2)
    device = torch.device("cpu")
    posterior = fit_ms_egarch(
        returns, advi_config, HierarchicalPriorConfig(), seeds=[0], device=device,
        layout=layout, fallback_log_path=tmp_path / "fallbacks.json",
    )
    for r in posterior.seed_results:
        assert r.mu.device.type == device.type
        assert r.log_sigma.device.type == device.type
