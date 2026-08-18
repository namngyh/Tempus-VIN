
import pandas as pd
import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init
from raemf_mc.regime.posterior_features import (
    canonical_theta,
    compute_posterior_volatility_features,
)
from raemf_mc.regime.state_alignment import STATE_NAMES
from raemf_mc.regime.posterior_features import compute_regime_labels


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


def test_regime_labels_are_valid_state_names_and_reproducible():
    layout = MSEGARCHParamLayout()
    torch.manual_seed(3)
    mus = [torch.randn(layout.total) * 0.05 for _ in range(2)]
    posterior = _fake_posterior(mus, layout)
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    returns = pd.Series(torch.randn(40).numpy() * 0.01, index=dates)
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    labels_1 = compute_regime_labels(
        posterior, returns, init_log_var, init_log_state_prob, n_train=30, layout=layout
    )
    labels_2 = compute_regime_labels(
        posterior, returns, init_log_var, init_log_state_prob, n_train=30, layout=layout
    )
    assert labels_1.name == "regime_label"
    assert list(labels_1.index) == list(dates)
    assert set(labels_1.unique()).issubset(set(STATE_NAMES))
    pd.testing.assert_series_equal(labels_1, labels_2)  # deterministic, no sampling


def test_regime_labels_train_slice_unaffected_by_val_test_perturbation():
    """Regression test for the align_states leakage bug: align_states used to
    be fit over the FULL series, so the state->name permutation — and hence
    the TRAINING targets — depended on val/test rows. Same train prefix +
    different tail must yield byte-identical train-region labels.

    Verified to FAIL against the pre-fix code path (with this seed the
    full-series permutation moves from [2, 1, 0, 3] to [0, 3, 2, 1] when only
    the tail changes) and to PASS after the fix.
    """
    layout = MSEGARCHParamLayout()
    torch.manual_seed(4)
    mus = [torch.randn(layout.total) * 0.05 for _ in range(2)]
    posterior = _fake_posterior(mus, layout)
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    n_train = 30
    base_returns = torch.randn(50) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    returns_a = pd.Series(base_returns.numpy(), index=dates)
    perturbed_tail = base_returns.clone()
    perturbed_tail[n_train:] = torch.randn(50 - n_train) * 0.05  # different tail
    returns_b = pd.Series(perturbed_tail.numpy(), index=dates)

    # The tail really does differ — otherwise this test proves nothing.
    assert not returns_a.iloc[n_train:].equals(returns_b.iloc[n_train:])
    pd.testing.assert_series_equal(returns_a.iloc[:n_train], returns_b.iloc[:n_train])

    labels_a = compute_regime_labels(
        posterior, returns_a, init_log_var, init_log_state_prob,
        n_train=n_train, layout=layout,
    )
    labels_b = compute_regime_labels(
        posterior, returns_b, init_log_var, init_log_state_prob,
        n_train=n_train, layout=layout,
    )
    pd.testing.assert_series_equal(
        labels_a.iloc[:n_train], labels_b.iloc[:n_train]
    )
