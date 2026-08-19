
import pandas as pd
import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init
from raemf_mc.regime.posterior_features import (
    canonical_theta,
    compute_posterior_volatility_features,
    regime_health_report,
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


def _degenerate_posterior(layout, dominant_state=3):
    """Theta tao ra mot fit ma mot che do ap dao: omega cua che do do thap
    hon han (bien dong nho nhat) va ma tran chuyen thien ve o lai do."""
    theta = torch.zeros(layout.total)
    n = layout.n_states
    theta[0:n] = -1.0                      # omega
    theta[dominant_state] = -4.0           # omega thap -> sigma nho
    theta[2 * n : 3 * n] = 0.7             # beta, deu trong vung dung
    theta[-1] = -4.0                       # nu_raw -> nu ~ 2.068, sat san
    return PooledPosterior(seed_results=[FitResult(
        mu=theta, log_sigma=torch.full_like(theta, -50.0), elbo_trace=[0.0],
        completed_without_divergence=True, fallback_used=False,
        fallback_reason=None, n_retries=0, seed=0,
    )])


def test_regime_health_report_flags_a_degenerate_fit():
    """Chan doan phai BAT duoc dung lop loi ma cac chan doan cu bo sot:
    tren du lieu that, ca ba seed deu bao `fallback_used=False`,
    `clamp_saturation_fraction=0` va ELBO trace binh thuong, trong khi nhan
    thi 1049/1049 phien cung mot lop va nu = 2.15-2.21 sat san. Neu
    `healthy` van True trong tinh huong nhu vay thi ham nay vo dung.
    """
    layout = MSEGARCHParamLayout()
    posterior = _degenerate_posterior(layout)
    torch.manual_seed(4)
    returns = pd.Series(torch.randn(400).numpy() * 0.01)
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    report = regime_health_report(
        posterior, returns, init_log_var, init_log_state_prob,
        n_train=300, layout=layout,
    )
    assert report["healthy"] is False
    assert report["problems"]
    # nu = 2.05 + softplus(-4.0) ~= 2.068, phai bi bat vi sat san
    assert report["nu"] < 2.05 + 0.5
    assert any("nu" in p for p in report["problems"])
    # cac truong so lieu phai co mat de bao cao duoc, khong chi co co bool
    assert len(report["occupancy"]) == layout.n_states
    assert 0.0 <= report["max_occupancy_share"] <= 1.0
    assert report["n_label_classes"] >= 1


def test_regime_health_report_passes_a_healthy_fit():
    """Doi xung voi test tren: mot fit khong vi pham nguong nao phai bao
    healthy=True. Neu thieu test nay, mot ham luon tra ve False cung se
    'pass' test o tren."""
    layout = MSEGARCHParamLayout()
    theta = torch.zeros(layout.total)
    n = layout.n_states
    theta[0:n] = torch.tensor([-2.0, -1.5, -1.0, -0.5])   # omega tach biet
    theta[2 * n : 3 * n] = 0.5
    theta[-1] = 5.0                                        # nu ~ 7.06, xa san
    posterior = PooledPosterior(seed_results=[FitResult(
        mu=theta, log_sigma=torch.full_like(theta, -50.0), elbo_trace=[0.0],
        completed_without_divergence=True, fallback_used=False,
        fallback_reason=None, n_retries=0, seed=0,
    )])
    torch.manual_seed(5)
    returns = pd.Series(torch.randn(400).numpy() * 0.01)
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    report = regime_health_report(
        posterior, returns, init_log_var, init_log_state_prob,
        n_train=300, layout=layout,
    )
    assert report["nu"] > 2.05 + 0.5
    assert not any("nu" in p for p in report["problems"])


def test_compute_regime_labels_logs_a_degenerate_fit(tmp_path):
    """`health_log_path` phai bien fit suy bien thanh mot ban ghi tren dia.
    Day la yeu cau ky luat cua du an ("khong silent fallback"), khong phai
    tien ich: truoc thay doi nay pipeline bao thanh cong sach se trong khi
    sinh ra nhan don lop."""
    layout = MSEGARCHParamLayout()
    posterior = _degenerate_posterior(layout)
    torch.manual_seed(6)
    returns = pd.Series(torch.randn(400).numpy() * 0.01)
    init_log_var, init_log_state_prob = default_recursion_init(layout)
    log_path = tmp_path / "health.json"

    labels = compute_regime_labels(
        posterior, returns, init_log_var, init_log_state_prob,
        n_train=300, layout=layout, health_log_path=log_path,
    )
    assert len(labels) == 400
    assert log_path.exists(), "fit suy bien phai duoc ghi log"
    content = log_path.read_text(encoding="utf-8")
    assert "degenerate_regime_fit" in content
