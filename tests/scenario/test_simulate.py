import math

import pytest
import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init
from raemf_mc.scenario.simulate import (
    SIM_VOL_BAND_MULTIPLE,
    simulate_mc_paths,
    simulation_log_var_band,
)


def _fake_posterior(theta: torch.Tensor, log_sigma_value: float = -50.0) -> PooledPosterior:
    result = FitResult(
        mu=theta, log_sigma=torch.full_like(theta, log_sigma_value), elbo_trace=[0.0],
        completed_without_divergence=True, fallback_used=False, fallback_reason=None,
        n_retries=0, seed=0,
    )
    return PooledPosterior(seed_results=[result])


def test_simulate_mc_paths_shape_and_finiteness(tmp_path):
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
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert paths.shape == (25, 20)
    assert torch.isfinite(paths).all()


def test_simulate_mc_paths_reproducible_with_fixed_generator(tmp_path):
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
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    paths_2 = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob,
        n_paths=10, horizon=5, layout=layout, generator=torch.Generator().manual_seed(42),
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    torch.testing.assert_close(paths_1, paths_2)


def test_simulate_mc_paths_degenerate_single_state_no_crash_and_finite(tmp_path):
    """K=1 edge case: MSEGARCHParamLayout(n_states=1) gives
    transition_logits shape (1, 0), which exercises transition_matrix's
    reference-category softmax at its degenerate boundary (a single-column
    all-zero row that softmaxes to exactly [1.0]), and torch.multinomial
    sampling from a single-category distribution (always index 0 with
    probability 1). This test verifies only that this boundary case runs
    without error and produces finite output of the expected shape -- it
    does NOT compare against a manually-written single-regime recursion."""
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
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert paths.shape == (5, 10)
    assert torch.isfinite(paths).all()


def test_simulate_mc_paths_honors_explicit_device_argument(tmp_path):
    """Regression guard for two device-allocation bugs: (1) daily_returns
    used to be allocated with the raw `device` kwarg instead of
    centered_returns.device, and (2) posterior-sampled params/mu never
    moved to an explicitly-requested device. This test passes an explicit
    device=torch.device("cpu") (the only device available in this
    environment) so both previously-buggy code paths execute, and checks
    the result is on that device and matches the device=None result
    (since everything here is already CPU-resident, moving to "cpu" is a
    no-op numerically -- this only guards the code path, not cross-device
    correctness, which cannot be exercised without a GPU)."""
    layout = MSEGARCHParamLayout()
    theta = torch.zeros(layout.total)
    theta[2 * layout.n_states : 3 * layout.n_states] = -1.0
    theta[-1] = 3.0
    ms_posterior = _fake_posterior(theta)
    mu_posterior = _fake_posterior(torch.tensor([0.001, -0.001, 0.0, -0.002]))

    torch.manual_seed(11)
    centered_returns = torch.randn(50) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    paths_explicit = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob,
        n_paths=10, horizon=5, layout=layout, device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(13),
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    paths_default = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob,
        n_paths=10, horizon=5, layout=layout, device=None,
        generator=torch.Generator().manual_seed(13),
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert paths_explicit.device == centered_returns.device
    torch.testing.assert_close(paths_explicit, paths_default)


def test_simulation_log_var_band_arithmetic():
    """simulation_log_var_band is public and pure (simulate.py:126): given the
    estimation window's realized std, the returned (lo, hi) bounds on
    log-variance must translate back to sigma_window/M and sigma_window*M
    exactly, where M = SIM_VOL_BAND_MULTIPLE and the half-width is 2*log(M)
    (stated in volatility, enforced in log-variance -- see the function's own
    docstring). Uses a 2-point series whose variance is computable by hand
    (mean 0, unbiased/ddof=1 variance = ((-1)^2 + 1^2) / (2-1) = 2, so
    std = sqrt(2)) to keep the expected values independent of torch's own
    var() call rather than circularly re-deriving them from it."""
    centered_returns = torch.tensor([-1.0, 1.0])
    std = math.sqrt(2.0)

    log_var_lo, log_var_hi = simulation_log_var_band(centered_returns)

    sigma_lo = torch.exp(0.5 * log_var_lo).item()
    sigma_hi = torch.exp(0.5 * log_var_hi).item()
    assert sigma_lo == pytest.approx(std / SIM_VOL_BAND_MULTIPLE, rel=1e-5)
    assert sigma_hi == pytest.approx(std * SIM_VOL_BAND_MULTIPLE, rel=1e-5)


def test_simulate_mc_paths_does_not_censor_the_tail(tmp_path):
    """Regression guard for the review finding that no committed test
    distinguishes the winsorized news-impact term `z_impact` (used only
    inside the variance recursion, per the note at the top of simulate.py)
    from the never-winsorized `eps_h` that actually generates the returned
    value. If a future change accidentally clamped `eps_h` too (e.g. reusing
    `z_impact` to build the return, or clamping `r_h` itself to the vol
    band), every simulated |return| would be bounded by sigma_hi and this
    test would fail.

    nu_raw = -4.0 -> nu = nu_from_raw(-4.0) ~= 2.068 (nu = min_nu +
    softplus(nu_raw), min_nu=2.05 -- see ms_egarch.nu_from_raw), i.e. close
    to the model's nu floor where the Student-t tail is heaviest. beta_raw =
    -0.7 keeps every state comfortably stationary (|beta| < 1), so this is
    testing genuine fat-tailed sampling, not the non-stationary/band-
    saturation path exercised elsewhere.

    Parameter choice was swept empirically (n_paths=200, horizon=20, same
    fixture, generator seeds 1-29) before fixing on seed=1: every one of the
    29 seeds produced max|r_h| / sigma_hi ratios from 5.97 to 68.4, so this
    is not a cherry-picked seed -- the effect is robust across the whole
    range tried, and seed=1 (ratio ~8.2) was picked to be comfortably above
    1 without being the single luckiest draw."""
    layout = MSEGARCHParamLayout()
    theta = torch.zeros(layout.total)
    theta[2 * layout.n_states : 3 * layout.n_states] = -0.7  # stationary beta
    theta[-1] = -4.0  # nu_raw -> nu near its 2.05 floor (fat tails)
    ms_posterior = _fake_posterior(theta)
    mu_posterior = _fake_posterior(torch.tensor([0.0, 0.0, 0.0, 0.0]))

    torch.manual_seed(5)
    centered_returns = torch.randn(50) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    _, log_var_hi = simulation_log_var_band(centered_returns)
    sigma_hi = torch.exp(0.5 * log_var_hi).item()

    gen = torch.Generator().manual_seed(1)
    paths = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob,
        n_paths=200, horizon=20, layout=layout, generator=gen,
        fallback_log_path=tmp_path / "fallbacks.json",
    )

    max_abs_return = paths.abs().max().item()
    ratio = max_abs_return / sigma_hi
    assert ratio > 5.0, (
        f"max|r_h|/sigma_hi = {ratio:.3f}: the returned tail should sit far "
        "outside the volatility band, proving eps_h is never winsorized"
    )
