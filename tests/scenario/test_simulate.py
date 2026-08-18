import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init
from raemf_mc.scenario.simulate import simulate_mc_paths


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
