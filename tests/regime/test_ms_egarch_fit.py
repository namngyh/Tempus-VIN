import torch
from raemf_mc.regime.ms_egarch import (
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
