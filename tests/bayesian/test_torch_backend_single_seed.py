import json
import torch
import pytest
from raemf_mc.bayesian.torch_backend import (
    AdviConfig,
    FitResult,
    assert_no_float16,
    fit_mean_field_advi,
)


def test_assert_no_float16_raises_on_float16_tensor():
    ok = torch.zeros(3, dtype=torch.float32)
    bad = torch.zeros(3, dtype=torch.float16)
    assert_no_float16(ok)  # must not raise
    with pytest.raises(TypeError):
        assert_no_float16(ok, bad)


def _gaussian_log_joint(theta: torch.Tensor) -> torch.Tensor:
    # log density of a standard 2D Gaussian at theta — known analytic
    # posterior: mu -> 0, sigma -> 1 (up to ADVI's mean-field approximation).
    return -0.5 * torch.sum(theta**2) - theta.shape[0] * 0.5 * torch.log(
        torch.tensor(2 * 3.141592653589793)
    )


def test_fit_mean_field_advi_recovers_known_gaussian_mean(tmp_path):
    config = AdviConfig(n_steps=300, learning_rate=0.05, warmup_steps=20, elbo_ma_window=10, early_stop_patience=300)
    init_mu = torch.full((2,), 3.0)
    init_log_sigma = torch.zeros(2)
    result = fit_mean_field_advi(
        _gaussian_log_joint, init_mu, init_log_sigma, config, seed=0,
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert isinstance(result, FitResult)
    assert torch.allclose(result.mu, torch.zeros(2), atol=0.3)
    assert len(result.elbo_trace) > 0
    assert result.fallback_used is False


def test_fit_mean_field_advi_rejects_float16_init(tmp_path):
    config = AdviConfig(n_steps=5)
    bad_mu = torch.zeros(2, dtype=torch.float16)
    init_log_sigma = torch.zeros(2)
    with pytest.raises(TypeError):
        fit_mean_field_advi(
            _gaussian_log_joint, bad_mu, init_log_sigma, config, seed=0,
            device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
        )


def _diverging_log_joint(theta: torch.Tensor) -> torch.Tensor:
    # deliberately produces NaN once theta grows large, to exercise the
    # retry/fallback path.
    huge = torch.exp(theta.sum() * 50.0)
    return -huge


def test_fit_mean_field_advi_logs_fallback_on_persistent_divergence(tmp_path):
    log_path = tmp_path / "fallbacks.json"
    config = AdviConfig(
        n_steps=50, learning_rate=5.0, warmup_steps=0, max_retries=1, retry_lr_factor=0.5,
    )
    init_mu = torch.full((1,), 10.0)
    init_log_sigma = torch.zeros(1)
    result = fit_mean_field_advi(
        _diverging_log_joint, init_mu, init_log_sigma, config, seed=0,
        device=torch.device("cpu"), fallback_log_path=log_path,
    )
    assert result.fallback_used is True
    assert result.fallback_reason is not None
    assert log_path.exists()
    events = json.loads(log_path.read_text())
    assert len(events) >= 1


def _finite_loss_nan_grad_log_joint(theta: torch.Tensor) -> torch.Tensor:
    # Value is always exactly 0.0 (so torch.isfinite(loss) passes), but the
    # backward pass computes 0 * d/dx sqrt(x)|_{x=0} = 0 * inf = NaN, so
    # optimizer.step() writes NaNs into mu/log_sigma. This is the shape of a
    # 0/0 inside a real log_joint recursion.
    x = theta.sum()
    zero = x - x.detach()
    return torch.sqrt(zero) * 0.0


def test_fallback_never_returns_nan_params_when_gradients_poison_state(tmp_path):
    config = AdviConfig(n_steps=20, learning_rate=0.1, warmup_steps=0, max_retries=1)
    init_mu = torch.full((2,), 0.5)
    init_log_sigma = torch.zeros(2)
    result = fit_mean_field_advi(
        _finite_loss_nan_grad_log_joint, init_mu, init_log_sigma, config, seed=0,
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert result.fallback_used is True
    # the reason names a "last finite step" — it must actually be finite
    assert torch.isfinite(result.mu).all()
    assert torch.isfinite(result.log_sigma).all()


def _raising_log_joint(theta: torch.Tensor) -> torch.Tensor:
    raise ValueError("scale must be positive")


def test_exception_inside_log_joint_takes_the_fallback_path_not_the_stack(tmp_path):
    log_path = tmp_path / "nested" / "fallbacks.json"
    config = AdviConfig(n_steps=10, warmup_steps=0, max_retries=1)
    result = fit_mean_field_advi(
        _raising_log_joint, torch.zeros(2), torch.zeros(2), config, seed=0,
        device=torch.device("cpu"), fallback_log_path=log_path,
    )
    assert result.fallback_used is True
    assert torch.isfinite(result.mu).all()
    # append_fallback_log must create missing parent directories
    events = json.loads(log_path.read_text())
    reasons = " ".join(str(e.get("reason", "")) + str(e.get("last_failure", "")) for e in events)
    assert "log_joint_raised" in reasons
    assert "scale must be positive" in reasons
