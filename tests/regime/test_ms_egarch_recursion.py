import torch
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    MSEGARCHParams,
    run_ms_egarch_recursion,
    expected_abs_standardized_t,
    default_recursion_init,
    unpack_params,
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


def test_batched_recursion_matches_looping_over_single_thetas():
    """Đường batch phải cho ra ĐÚNG cùng số học với việc gọi lần lượt từng
    theta một — đây là hợp đồng nền tảng của toàn bộ đợt vector hoá: mọi
    tăng tốc sau đó (ADVI nhiều MC sample, Monte Carlo nhiều path) đều dựa
    vào giả định này. Dùng `assert_close` với dung sai mặc định của float32
    chứ không phải bằng nhau tuyệt đối: thứ tự rút gọn (reduction) trong
    logsumexp/softmax trên tensor (B, n) không nhất thiết trùng bit-đối-bit
    với tensor (n,), nên sai khác cỡ ULP là hợp lệ và không phải lỗi.
    """
    layout = MSEGARCHParamLayout()
    torch.manual_seed(11)
    returns = torch.randn(120) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    thetas = torch.randn(6, layout.total) * 0.3
    batched = run_ms_egarch_recursion(
        returns, unpack_params(thetas, layout), init_log_var, init_log_state_prob
    )
    assert batched["log_var"].shape == (6, 120, layout.n_states)
    assert batched["log_filtered_prob"].shape == (6, 120, layout.n_states)
    assert batched["log_var_bar"].shape == (6, 120)
    assert batched["total_log_lik"].shape == (6,)
    assert batched["nu"].shape == (6, layout.n_states)

    for i in range(6):
        single = run_ms_egarch_recursion(
            returns, unpack_params(thetas[i], layout), init_log_var, init_log_state_prob
        )
        torch.testing.assert_close(batched["log_var"][i], single["log_var"])
        torch.testing.assert_close(
            batched["log_filtered_prob"][i], single["log_filtered_prob"]
        )
        torch.testing.assert_close(batched["log_var_bar"][i], single["log_var_bar"])
        torch.testing.assert_close(batched["total_log_lik"][i], single["total_log_lik"])
        torch.testing.assert_close(batched["nu"][i], single["nu"])


def test_batched_recursion_gradients_do_not_leak_across_batch_rows():
    """Mỗi hàng batch phải hoàn toàn độc lập: gradient của total_log_lik của
    hàng i theo theta của hàng j != i phải bằng đúng 0. Nếu một phép
    broadcast nào đó vô tình trộn các hàng (ví dụ rút gọn nhầm chiều), test
    này bắt được — trong khi test đối chiếu giá trị ở trên thì KHÔNG, vì
    giá trị forward vẫn có thể đúng khi chỉ đường gradient bị hỏng. Đây là
    điều kiện sống còn cho việc gộp nhiều MC sample vào một bước ADVI.
    """
    layout = MSEGARCHParamLayout()
    torch.manual_seed(12)
    returns = torch.randn(60) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    thetas = (torch.randn(3, layout.total) * 0.3).requires_grad_(True)
    result = run_ms_egarch_recursion(
        returns, unpack_params(thetas, layout), init_log_var, init_log_state_prob
    )
    result["total_log_lik"][1].backward()

    assert thetas.grad is not None
    assert torch.all(thetas.grad[0] == 0.0)
    assert torch.all(thetas.grad[2] == 0.0)
    assert torch.any(thetas.grad[1] != 0.0)


def test_recursion_uses_a_separate_nu_per_regime():
    """nu riêng từng chế độ phải thực sự tới được hàm mật độ, không phải chỉ
    tồn tại trong layout. Dựng một chế độ có đuôi rất nặng (nu ~ 2,3) cạnh
    một chế độ gần Gauss (nu ~ 30): nếu recursion vẫn dùng một nu dùng chung
    thì bốn giá trị trả về sẽ bằng nhau và test này gãy.

    Lý do đổi: với một nu dùng chung, đo được nu rơi xuống sàn 2,05 một cách
    hệ thống (2,15-2,21 trên cả ba seed) vì nó bị buộc phải hấp thụ đuôi của
    những phiên bất thường cho cả bốn chế độ cùng lúc.
    """
    layout = MSEGARCHParamLayout()
    torch.manual_seed(13)
    returns = torch.randn(80) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    theta = torch.zeros(layout.total)
    theta[2 * layout.n_states : 3 * layout.n_states] = 0.5          # beta ổn định
    # nu = 2.05 + softplus(nu_raw): nu_raw = -2 -> ~2.18; nu_raw = 3.3 -> ~5.4
    theta[-layout.n_states :] = torch.tensor([-2.0, 0.0, 1.5, 3.3])

    result = run_ms_egarch_recursion(
        returns, unpack_params(theta, layout), init_log_var, init_log_state_prob
    )
    nu = result["nu"]
    assert nu.shape == (layout.n_states,)
    assert nu[0] < 2.3, "chế độ đầu phải có đuôi rất nặng"
    assert nu[3] > 5.0, "chế độ cuối phải gần Gauss hơn hẳn"
    assert torch.all(nu[:-1] < nu[1:]), "nu phải tăng đúng theo nu_raw đã đặt"
    assert torch.isfinite(result["total_log_lik"]).all()


def test_recursion_still_accepts_a_single_shared_nu():
    """Tương thích ngược: một `nu_raw` vô hướng (layout cũ, và mọi fixture
    test viết trước thay đổi này) vẫn phải chạy, được hiểu là cùng một nu
    cho cả bốn chế độ. Nếu mất tính chất này thì mọi theta lưu từ trước và
    mọi test cũ đều gãy mà không có đường di trú nào.
    """
    layout = MSEGARCHParamLayout()
    torch.manual_seed(14)
    returns = torch.randn(60) * 0.01
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    shared = MSEGARCHParams(
        omega=torch.full((4,), -1.0),
        alpha=torch.full((4,), 0.2),
        beta=torch.full((4,), 0.5),
        gamma=torch.zeros(4),
        transition_logits=torch.zeros(4, 3),
        nu_raw=torch.tensor(2.0),                    # vô hướng, dùng chung
    )
    per_state = MSEGARCHParams(
        omega=shared.omega, alpha=shared.alpha, beta=shared.beta,
        gamma=shared.gamma, transition_logits=shared.transition_logits,
        nu_raw=torch.full((4,), 2.0),                # cùng giá trị, nhưng theo trạng thái
    )
    a = run_ms_egarch_recursion(returns, shared, init_log_var, init_log_state_prob)
    b = run_ms_egarch_recursion(returns, per_state, init_log_var, init_log_state_prob)
    torch.testing.assert_close(a["total_log_lik"], b["total_log_lik"])
    torch.testing.assert_close(a["nu"], b["nu"])
