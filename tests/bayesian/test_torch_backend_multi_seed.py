import torch
from raemf_mc.bayesian.torch_backend import (
    AdviConfig,
    PooledPosterior,
    fit_multi_seed_advi,
    sample_joint_draw,
)


def _gaussian_log_joint(theta: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(theta**2)


def test_fit_multi_seed_advi_pools_all_seeds_equally(tmp_path):
    config = AdviConfig(n_steps=100, learning_rate=0.05, warmup_steps=10, elbo_ma_window=10, early_stop_patience=100)
    init_mu = torch.full((2,), 2.0)
    init_log_sigma = torch.zeros(2)
    posterior = fit_multi_seed_advi(
        _gaussian_log_joint, init_mu, init_log_sigma, config, seeds=[0, 1, 2],
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert isinstance(posterior, PooledPosterior)
    assert len(posterior.seed_results) == 3
    assert {r.seed for r in posterior.seed_results} == {0, 1, 2}
    assert posterior.n_fallback_seeds == 0
    assert posterior.fallback_fraction == 0.0


def _diverging_log_joint(theta: torch.Tensor) -> torch.Tensor:
    return -torch.exp(theta.sum() * 50.0)


def test_pooled_posterior_exposes_fallback_seed_count(tmp_path):
    log_path = tmp_path / "fallbacks.json"
    config = AdviConfig(n_steps=20, learning_rate=5.0, warmup_steps=0, max_retries=1)
    posterior = fit_multi_seed_advi(
        _diverging_log_joint, torch.full((1,), 10.0), torch.zeros(1), config,
        seeds=[0, 1], device=torch.device("cpu"), fallback_log_path=log_path,
    )
    # fallback seeds still participate in the pool at equal weight (no
    # cherry-picking), but the pool must be able to say so.
    assert len(posterior.seed_results) == 2
    assert posterior.n_fallback_seeds == 2
    assert posterior.fallback_fraction == 1.0
    summary = posterior.fallback_summary()
    assert summary["n_seeds"] == 2
    assert {s["seed"] for s in summary["fallback_seeds"]} == {0, 1}


def test_sample_joint_draw_is_deterministic_given_generator(tmp_path):
    config = AdviConfig(n_steps=50, learning_rate=0.05, warmup_steps=5, elbo_ma_window=10, early_stop_patience=50)
    init_mu = torch.zeros(2)
    init_log_sigma = torch.zeros(2)
    posterior = fit_multi_seed_advi(
        _gaussian_log_joint, init_mu, init_log_sigma, config, seeds=[0, 1],
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    draw1 = sample_joint_draw(posterior, generator=gen1)
    draw2 = sample_joint_draw(posterior, generator=gen2)
    assert torch.equal(draw1, draw2)
    assert draw1.shape == (2,)


def test_sample_joint_draw_varies_across_calls_without_fixed_generator(tmp_path):
    config = AdviConfig(n_steps=20, learning_rate=0.05, warmup_steps=2, elbo_ma_window=5, early_stop_patience=20)
    init_mu = torch.zeros(2)
    init_log_sigma = torch.zeros(2)
    posterior = fit_multi_seed_advi(
        _gaussian_log_joint, init_mu, init_log_sigma, config, seeds=[0, 1, 2, 3],
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    draws = [sample_joint_draw(posterior) for _ in range(20)]
    assert not all(torch.equal(draws[0], d) for d in draws[1:])
