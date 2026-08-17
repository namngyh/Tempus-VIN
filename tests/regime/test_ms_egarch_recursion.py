import torch
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParams,
    run_ms_egarch_recursion,
    expected_abs_standardized_t,
    default_recursion_init,
)


def _single_regime_egarch_reference(
    returns: torch.Tensor, omega, alpha, beta, gamma, nu, init_log_var
) -> torch.Tensor:
    """Minimal single-regime EGARCH recursion used as a ground truth for
    the degenerate K=1 case."""
    T = returns.shape[0]
    log_var = torch.zeros(T)
    e_abs_z = expected_abs_standardized_t(nu)
    log_var_prev = init_log_var
    z_prev = torch.tensor(0.0)
    for t in range(T):
        log_var_t = omega + beta * log_var_prev + alpha * (torch.abs(z_prev) - e_abs_z) + gamma * z_prev
        log_var[t] = log_var_t
        sigma_t = torch.exp(0.5 * log_var_t)
        z_prev = returns[t] / sigma_t
        log_var_prev = log_var_t
    return log_var


def test_recursion_output_shapes_and_finiteness():
    torch.manual_seed(0)
    T, n = 20, 4
    returns = torch.randn(T) * 0.01
    params = MSEGARCHParams(
        omega=torch.full((n,), -0.1),
        alpha=torch.full((n,), 0.1),
        beta=torch.full((n,), 0.9),
        gamma=torch.full((n,), -0.05),
        transition_logits=torch.zeros(n, n - 1),
        nu_raw=torch.tensor(2.0),
    )
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 0.25))
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
    assert result["log_var"].shape == (T, n)
    assert result["log_filtered_prob"].shape == (T, n)
    assert result["log_var_bar"].shape == (T,)
    assert torch.isfinite(result["log_var"]).all()
    assert torch.isfinite(result["log_filtered_prob"]).all()
    assert torch.isfinite(result["total_log_lik"])


def test_filtered_probabilities_form_a_simplex_every_step():
    torch.manual_seed(1)
    T, n = 15, 4
    returns = torch.randn(T) * 0.01
    params = MSEGARCHParams(
        omega=torch.full((n,), -0.2),
        alpha=torch.full((n,), 0.15),
        beta=torch.full((n,), 0.85),
        gamma=torch.full((n,), 0.02),
        transition_logits=torch.randn(n, n - 1) * 0.3,
        nu_raw=torch.tensor(1.5),
    )
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 0.25))
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
    probs = torch.exp(result["log_filtered_prob"])
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(T), atol=1e-4)


def test_degenerate_single_state_matches_reference_egarch():
    torch.manual_seed(2)
    T = 25
    returns = torch.randn(T) * 0.01
    n = 4
    omega, alpha, beta, gamma = -0.15, 0.12, 0.88, -0.03
    nu_val = 6.0
    # push ALL transition mass onto state 0 staying in state 0 -> collapses
    # to a single-regime EGARCH driven purely by state 0's parameters.
    huge = 50.0
    transition_logits = torch.full((n, n - 1), -huge)
    params = MSEGARCHParams(
        omega=torch.tensor([omega, 5.0, 5.0, 5.0]),
        alpha=torch.tensor([alpha, 0.0, 0.0, 0.0]),
        beta=torch.tensor([beta, 0.0, 0.0, 0.0]),
        gamma=torch.tensor([gamma, 0.0, 0.0, 0.0]),
        transition_logits=transition_logits,
        nu_raw=torch.log(torch.exp(torch.tensor(nu_val - 2.05)) - 1.0),  # inverse-softplus
    )
    init_log_var = torch.tensor([0.0, -huge, -huge, -huge])
    init_log_state_prob = torch.log(
        torch.tensor([1.0 - 3e-8, 1e-8, 1e-8, 1e-8])
    )
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)

    ref_log_var = _single_regime_egarch_reference(
        returns, torch.tensor(omega), torch.tensor(alpha), torch.tensor(beta),
        torch.tensor(gamma), torch.tensor(nu_val), torch.tensor(0.0),
    )
    got_log_var_state0 = result["log_var"][:, 0]
    assert torch.allclose(got_log_var_state0, ref_log_var, atol=1e-2)


def test_forward_filter_is_causal_prefix_invariant():
    """Fitting on data[0:T] and data[0:T+k] must produce identical
    log_filtered_prob for t <= T-1 — the filter never looks ahead."""
    torch.manual_seed(3)
    T, extra, n = 15, 5, 4
    returns_full = torch.randn(T + extra) * 0.01
    params = MSEGARCHParams(
        omega=torch.full((n,), -0.2),
        alpha=torch.full((n,), 0.1),
        beta=torch.full((n,), 0.85),
        gamma=torch.full((n,), 0.01),
        transition_logits=torch.randn(n, n - 1) * 0.2,
        nu_raw=torch.tensor(1.0),
    )
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 0.25))

    result_short = run_ms_egarch_recursion(
        returns_full[:T], params, init_log_var, init_log_state_prob
    )
    result_long = run_ms_egarch_recursion(
        returns_full, params, init_log_var, init_log_state_prob
    )
    assert torch.allclose(
        result_short["log_filtered_prob"], result_long["log_filtered_prob"][:T], atol=1e-6
    )
    assert torch.allclose(result_short["log_var"], result_long["log_var"][:T], atol=1e-6)


def test_clamp_saturation_fraction_is_zero_for_well_behaved_params():
    torch.manual_seed(4)
    T, n = 30, 4
    returns = torch.randn(T) * 0.01
    params = MSEGARCHParams(
        omega=torch.full((n,), -0.1),
        alpha=torch.full((n,), 0.1),
        beta=torch.full((n,), 0.9),
        gamma=torch.full((n,), -0.05),
        transition_logits=torch.zeros(n, n - 1),
        nu_raw=torch.tensor(2.0),
    )
    init_log_var, init_log_state_prob = default_recursion_init()
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
    assert result["clamp_saturation_fraction"] == 0.0


def test_clamp_saturation_fraction_reports_explosive_recursion():
    """The clamp keeps an explosive beta from producing NaN, but a clamped
    cell has zero gradient — ADVI sees a flat plateau, not an error. The
    fraction is the only signal that this happened at all."""
    torch.manual_seed(5)
    T, n = 40, 4
    returns = torch.randn(T) * 0.01
    params = MSEGARCHParams(
        omega=torch.full((n,), 2.0),
        alpha=torch.full((n,), 0.1),
        beta=torch.full((n,), 2.5),  # |beta| > 1 -> non-stationary
        gamma=torch.full((n,), 0.0),
        transition_logits=torch.zeros(n, n - 1),
        nu_raw=torch.tensor(2.0),
    )
    init_log_var, init_log_state_prob = default_recursion_init()
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
    assert result["clamp_saturation_fraction"] > 0.5
    assert torch.isfinite(result["log_var"]).all()


def test_default_recursion_init_is_unit_variance_and_uniform_states():
    init_log_var, init_log_state_prob = default_recursion_init()
    assert torch.equal(init_log_var, torch.zeros(4))
    assert torch.allclose(torch.exp(init_log_state_prob), torch.full((4,), 0.25))
