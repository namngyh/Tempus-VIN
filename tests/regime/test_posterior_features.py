import math

import pandas as pd
import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init
from raemf_mc.regime.posterior_features import (
    canonical_theta,
    compute_posterior_volatility_features,
)


def _fake_posterior(mus: list[torch.Tensor], layout: MSEGARCHParamLayout) -> PooledPosterior:
    results = [
        FitResult(
            mu=mu, log_sigma=torch.full((layout.total,), -3.0), elbo_trace=[0.0],
            completed_without_divergence=True, fallback_used=False,
            fallback_reason=None, n_retries=0, seed=i,
        )
        for i, mu in enumerate(mus)
    ]
    return PooledPosterior(seed_results=results)


def test_canonical_theta_is_mean_of_seed_mus():
    layout = MSEGARCHParamLayout()
    mu_a = torch.zeros(layout.total)
    mu_b = torch.full((layout.total,), 2.0)
    posterior = _fake_posterior([mu_a, mu_b], layout)
    theta = canonical_theta(posterior)
    assert torch.allclose(theta, torch.full((layout.total,), 1.0))


def test_posterior_volatility_features_shape_and_index():
    layout = MSEGARCHParamLayout()
    torch.manual_seed(0)
    mus = [torch.randn(layout.total) * 0.1 for _ in range(3)]
    posterior = _fake_posterior(mus, layout)
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    returns = pd.Series(torch.randn(15).numpy() * 0.01, index=dates)
    init_log_var, init_log_state_prob = default_recursion_init(layout)
    gen = torch.Generator().manual_seed(1)
    features = compute_posterior_volatility_features(
        posterior, returns, init_log_var, init_log_state_prob,
        layout=layout, n_draws=8, generator=gen,
    )
    assert list(features.columns) == ["posterior_mean_sigma", "posterior_sd_sigma"]
    assert list(features.index) == list(dates)
    assert (features["posterior_mean_sigma"] > 0).all()
    assert (features["posterior_sd_sigma"] >= 0).all()
    assert not features.isna().any().any()


def test_posterior_volatility_features_reproducible_with_fixed_generator():
    layout = MSEGARCHParamLayout()
    torch.manual_seed(2)
    mus = [torch.randn(layout.total) * 0.1 for _ in range(2)]
    posterior = _fake_posterior(mus, layout)
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    returns = pd.Series(torch.randn(10).numpy() * 0.01, index=dates)
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    gen1 = torch.Generator().manual_seed(42)
    features_1 = compute_posterior_volatility_features(
        posterior, returns, init_log_var, init_log_state_prob,
        layout=layout, n_draws=5, generator=gen1,
    )
    gen2 = torch.Generator().manual_seed(42)
    features_2 = compute_posterior_volatility_features(
        posterior, returns, init_log_var, init_log_state_prob,
        layout=layout, n_draws=5, generator=gen2,
    )
    pd.testing.assert_frame_equal(features_1, features_2)
