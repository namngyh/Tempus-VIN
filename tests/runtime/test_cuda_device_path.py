# tests/runtime/test_cuda_device_path.py
"""Test cho đường CUDA thật — trước đợt này repo KHÔNG có một test nào chạm
tới GPU (ledger sub-project 3 ghi rõ giới hạn đó: "inherently untestable
without a CUDA device"). Toàn bộ file này bị skip trên máy không có CUDA,
nên nó không làm gãy CI/CPU-only; trên máy có CUDA nó là thứ duy nhất chứng
minh `device=torch.device("cuda")` thực sự chạy được đầu-cuối.
"""
import pytest
import torch

from raemf_mc.bayesian.torch_backend import (
    AdviConfig,
    FitResult,
    PooledPosterior,
    fit_mean_field_advi,
    sample_joint_draw,
    sample_joint_draws,
)
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    run_ms_egarch_recursion,
    unpack_params,
)
from raemf_mc.scenario.simulate import simulate_mc_paths

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cần GPU CUDA thật"
)


def _posterior(theta: torch.Tensor, device: torch.device) -> PooledPosterior:
    theta = theta.to(device)
    return PooledPosterior(
        seed_results=[
            FitResult(
                mu=theta,
                log_sigma=torch.full_like(theta, -3.0),
                elbo_trace=[0.0],
                completed_without_divergence=True,
                fallback_used=False,
                fallback_reason=None,
                n_retries=0,
                seed=0,
            )
        ]
    )


@cuda_only
def test_sample_joint_draw_works_with_cuda_generator():
    """Guard hồi quy cho một lỗi GPU THẬT đã tái hiện được trên RTX 4060
    trước khi sửa: `torch.randint(..., generator=gen_cuda)` không có
    `device=` cấp phát trên CPU và raise "Expected a 'cpu' device type for
    generator but found 'cuda'". Lỗi này nằm chắn ngang đường đi của
    `simulate_mc_paths` với `device=cuda`, tức toàn bộ tầng Monte Carlo
    không thể chạy trên GPU trước khi sửa.
    """
    device = torch.device("cuda")
    posterior = _posterior(torch.zeros(8), device)
    gen = torch.Generator(device=device).manual_seed(0)

    single = sample_joint_draw(posterior, generator=gen)
    assert single.device.type == "cuda"
    assert single.shape == (8,)

    batched = sample_joint_draws(posterior, 5, generator=gen)
    assert batched.device.type == "cuda"
    assert batched.shape == (5, 8)
    assert torch.isfinite(batched).all()


@cuda_only
def test_recursion_on_cuda_matches_cpu():
    """Cùng theta, cùng dữ liệu, hai thiết bị -> cùng kết quả trong dung sai
    float32. Dung sai nới hơn mặc định vì thứ tự rút gọn của kernel CUDA
    khác CPU; điều cần chứng minh là không có sai lệch mang tính hệ thống,
    không phải trùng bit-đối-bit (điều không thể đòi hỏi giữa hai backend).
    """
    layout = MSEGARCHParamLayout()
    torch.manual_seed(21)
    returns = torch.randn(300) * 0.01
    thetas = torch.randn(4, layout.total) * 0.3

    ilv_cpu, ilsp_cpu = default_recursion_init(layout)
    cpu_result = run_ms_egarch_recursion(
        returns, unpack_params(thetas, layout), ilv_cpu, ilsp_cpu
    )

    device = torch.device("cuda")
    ilv, ilsp = default_recursion_init(layout, device=device)
    cuda_result = run_ms_egarch_recursion(
        returns.to(device), unpack_params(thetas.to(device), layout), ilv, ilsp
    )

    assert cuda_result["total_log_lik"].device.type == "cuda"
    torch.testing.assert_close(
        cuda_result["total_log_lik"].cpu(), cpu_result["total_log_lik"],
        rtol=1e-4, atol=1e-3,
    )
    torch.testing.assert_close(
        cuda_result["log_var_bar"].cpu(), cpu_result["log_var_bar"],
        rtol=1e-4, atol=1e-3,
    )


@cuda_only
def test_simulate_mc_paths_runs_end_to_end_on_cuda(tmp_path):
    """Đường Monte Carlo đầy đủ trên GPU: tensor trả về phải nằm trên CUDA,
    hữu hạn, và đúng hình dạng. Đây chính là kịch bản `gpu_research.yaml`
    nhắm tới và là thứ README ghi là "chưa từng được chạy thật"."""
    layout = MSEGARCHParamLayout()
    device = torch.device("cuda")

    theta = torch.zeros(layout.total)
    theta[2 * layout.n_states : 3 * layout.n_states] = -1.0
    theta[-1] = 3.0
    ms_posterior = _posterior(theta, device)
    mu_posterior = _posterior(torch.tensor([0.001, -0.001, 0.0, -0.002]), device)

    torch.manual_seed(22)
    centered_returns = (torch.randn(200) * 0.01).to(device)
    init_log_var, init_log_state_prob = default_recursion_init(layout, device=device)

    gen = torch.Generator(device=device).manual_seed(7)
    paths = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var,
        init_log_state_prob, n_paths=64, horizon=10, layout=layout,
        device=device, generator=gen,
        fallback_log_path=tmp_path / "cuda_fallbacks.json",
    )
    assert paths.shape == (64, 10)
    assert paths.device.type == "cuda"
    assert torch.isfinite(paths).all()


@cuda_only
def test_simulate_mc_paths_chunking_covers_all_paths_on_cuda(tmp_path):
    """Chia lô phải sinh đủ n_paths hàng kể cả khi n_paths không chia hết cho
    kích thước lô — cơ chế chặn OOM ở quy mô nghiên cứu chỉ có giá trị nếu
    nó không âm thầm cắt mất path."""
    layout = MSEGARCHParamLayout()
    device = torch.device("cuda")
    theta = torch.zeros(layout.total)
    theta[2 * layout.n_states : 3 * layout.n_states] = -1.0
    theta[-1] = 3.0
    ms_posterior = _posterior(theta, device)
    mu_posterior = _posterior(torch.tensor([0.001, -0.001, 0.0, -0.002]), device)

    torch.manual_seed(23)
    centered_returns = (torch.randn(120) * 0.01).to(device)
    init_log_var, init_log_state_prob = default_recursion_init(layout, device=device)

    gen = torch.Generator(device=device).manual_seed(8)
    paths = simulate_mc_paths(
        ms_posterior, mu_posterior, centered_returns, init_log_var,
        init_log_state_prob, n_paths=70, horizon=5, layout=layout,
        device=device, generator=gen, path_chunk_size=32,
        fallback_log_path=tmp_path / "cuda_fallbacks.json",
    )
    assert paths.shape == (70, 5)
    assert torch.isfinite(paths).all()


@cuda_only
def test_recursion_is_cuda_graph_capturable():
    """Guard hồi quy cho ba phép sao chép CPU<->GPU ẩn đã được gỡ bỏ:
    `float(n_saturated)` trong recursion, `torch.as_tensor(min_nu)` trong
    `nu_from_raw`, và `torch.tensor(math.pi)` trong `_student_t_log_pdf`.
    Mỗi cái đều vô hình trong lúc chạy bình thường và mỗi cái đều làm
    `torch.cuda.graph` capture thất bại — đã tái hiện đủ cả ba lần khi thử.

    CUDA graph đáng giá ở đây vì đo được: overhead phát kernel trên card
    GeForce chạy WDDM là ~42 us/kernel ở chế độ eager và ~1,5 us khi phát
    lại graph, và recursion T=1500 phát khoảng 37 500 kernel. Nếu một thay
    đổi tương lai lén đưa lại một phép đồng bộ, cả đường đó sập về eager mà
    không có dấu hiệu nào ngoài việc chậm đi ~30 lần — test này là thứ duy
    nhất biến chuyện đó thành lỗi nhìn thấy được.
    """
    layout = MSEGARCHParamLayout()
    device = torch.device("cuda")
    torch.manual_seed(41)
    returns = (torch.randn(64) * 0.01).to(device)
    thetas = (torch.randn(3, layout.total) * 0.3).to(device)
    params = unpack_params(thetas, layout)
    init_log_var, init_log_state_prob = default_recursion_init(layout, device=device)

    with torch.no_grad():
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                run_ms_egarch_recursion(
                    returns, params, init_log_var, init_log_state_prob
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        reference = run_ms_egarch_recursion(
            returns, params, init_log_var, init_log_state_prob
        )["total_log_lik"].clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run_ms_egarch_recursion(
                returns, params, init_log_var, init_log_state_prob
            )
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

    torch.testing.assert_close(
        captured["total_log_lik"], reference, rtol=1e-4, atol=1e-4
    )


@cuda_only
def test_clamp_saturation_fraction_is_a_tensor_not_a_python_float():
    """Hệ quả trực tiếp của việc gỡ phép đồng bộ: trường chẩn đoán này giờ
    là tensor, và người gọi phải tự đổi sang float ở ngoài vùng capture.
    Test này chốt hợp đồng đó — nếu ai đó đổi lại thành float thì recursion
    hết capture được, và test ở trên sẽ gãy vì lý do trông chẳng liên quan
    gì; test này chỉ thẳng vào nguyên nhân."""
    layout = MSEGARCHParamLayout()
    device = torch.device("cuda")
    torch.manual_seed(42)
    returns = (torch.randn(32) * 0.01).to(device)
    params = unpack_params((torch.randn(layout.total) * 0.2).to(device), layout)
    init_log_var, init_log_state_prob = default_recursion_init(layout, device=device)

    result = run_ms_egarch_recursion(
        returns, params, init_log_var, init_log_state_prob
    )
    frac = result["clamp_saturation_fraction"]
    assert isinstance(frac, torch.Tensor)
    assert frac.device.type == "cuda"
    assert 0.0 <= float(frac) <= 1.0


def _graph_advi_fixture(device, n_steps=25, n_mc=8):
    """Bộ dựng dùng chung cho các test ADVI + CUDA graph."""
    from raemf_mc.bayesian.priors import HierarchicalPriorConfig
    from raemf_mc.regime.ms_egarch import build_ms_egarch_log_joint

    layout = MSEGARCHParamLayout()
    torch.manual_seed(3)
    returns = (torch.randn(250) * 0.01).to(device)
    init_log_var, init_log_state_prob = default_recursion_init(
        layout, dtype=returns.dtype, device=device
    )
    prior = HierarchicalPriorConfig(
        hyper_mean_scale=1.0, min_effective_observations=30.0,
        min_effective_fraction=0.05,
    )
    log_joint = build_ms_egarch_log_joint(
        returns, init_log_var, init_log_state_prob, prior, layout
    )
    kwargs = dict(
        n_steps=n_steps, learning_rate=0.05, warmup_steps=5, grad_clip_norm=10.0,
        elbo_ma_window=5, early_stop_patience=10_000, min_delta=0.0,
        retry_lr_factor=0.5, max_retries=1, n_mc_samples=n_mc,
    )
    init_mu = torch.zeros(layout.total, device=device)
    init_log_sigma = torch.full((layout.total,), -1.0, device=device)
    return log_joint, kwargs, init_mu, init_log_sigma


@cuda_only
def test_cuda_graph_advi_matches_eager_trajectory(tmp_path):
    """Đường CUDA graph phải đi CÙNG quỹ đạo với đường eager khi cùng seed.

    Đây là test đã tìm ra một lỗi thật trong chính đợt này: bản đầu dùng
    `optimizer.state.clear()` để đặt lại Adam sau các bước làm nóng, nhưng
    xoá dict không zero các tensor moment mà graph đã ghi cứng địa chỉ. Hệ
    quả rất dễ bị bỏ qua nếu chỉ nhìn kết quả cuối: `elbo[0]` trùng khít ở
    cả hai đường (nên mọi test "graph chạy được" đều pass), rồi quỹ đạo mới
    tách ra và `elbo` cuối lệch 6.5e2.

    Dung sai đặt ở mức của tích luỹ float32 qua vài chục bước với thứ tự
    kernel khác nhau, KHÔNG đòi trùng bit — điều đó không thể yêu cầu giữa
    một đồ thị đã ghi và một chuỗi lệnh eager.
    """
    device = torch.device("cuda")
    log_joint, kwargs, init_mu, init_log_sigma = _graph_advi_fixture(device)

    results = {}
    for label, use_graph in (("eager", False), ("graph", True)):
        config = AdviConfig(**kwargs, use_cuda_graph=use_graph)
        results[label] = fit_mean_field_advi(
            log_joint, init_mu, init_log_sigma, config, seed=11, device=device,
            fallback_log_path=tmp_path / f"fb_{label}.json",
        )

    eager, graphed = results["eager"], results["graph"]
    assert not eager.fallback_used and not graphed.fallback_used
    assert len(eager.elbo_trace) == len(graphed.elbo_trace)
    # elbo[0] trùng gần như tuyệt đối: cùng eps, cùng tham số ban đầu.
    assert abs(eager.elbo_trace[0] - graphed.elbo_trace[0]) < 1e-2
    torch.testing.assert_close(eager.mu, graphed.mu, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        eager.log_sigma, graphed.log_sigma, rtol=1e-2, atol=1e-2
    )


@cuda_only
def test_cuda_graph_advi_resets_adam_moments_not_just_the_state_dict(tmp_path):
    """Chốt riêng cơ chế đã hỏng một lần: sau khi dựng graph, các tensor
    moment của Adam phải THỰC SỰ bằng 0, chứ không chỉ bị bỏ tham chiếu.

    Test ở trên bắt được hệ quả (quỹ đạo lệch) nhưng chỉ khi chạy đủ nhiều
    bước; test này bắt được nguyên nhân trực tiếp và sẽ chỉ thẳng vào chỗ
    hỏng nếu ai đó đổi lại cách đặt lại trạng thái.
    """
    device = torch.device("cuda")
    log_joint, kwargs, init_mu, init_log_sigma = _graph_advi_fixture(
        device, n_steps=1
    )
    config = AdviConfig(**kwargs, use_cuda_graph=True)
    result = fit_mean_field_advi(
        log_joint, init_mu, init_log_sigma, config, seed=5, device=device,
        fallback_log_path=tmp_path / "fb.json",
    )
    # Một bước duy nhất sau khi đặt lại: giá trị ELBO phải khớp với cùng
    # một bước chạy trên đường eager, tức Adam đã thực sự khởi hành từ 0.
    eager_config = AdviConfig(**kwargs, use_cuda_graph=False)
    eager = fit_mean_field_advi(
        log_joint, init_mu, init_log_sigma, eager_config, seed=5, device=device,
        fallback_log_path=tmp_path / "fb_e.json",
    )
    assert abs(result.elbo_trace[0] - eager.elbo_trace[0]) < 1e-2
    torch.testing.assert_close(result.mu, eager.mu, rtol=1e-3, atol=1e-4)


@cuda_only
def test_cuda_graph_refuses_unbatched_log_joint(tmp_path):
    """Không im lặng tụt xuống đường eager: người gọi đã yêu cầu graph, và
    chạy chậm hơn 30 lần mà không báo gì là đúng kiểu thất bại âm thầm mà
    dự án này cấm."""
    device = torch.device("cuda")
    layout = MSEGARCHParamLayout()

    def plain_log_joint(theta):
        return -(theta**2).sum()

    config = AdviConfig(n_steps=2, n_mc_samples=2, use_cuda_graph=True)
    with pytest.raises(ValueError, match="supports_batched_theta"):
        fit_mean_field_advi(
            plain_log_joint,
            torch.zeros(layout.total, device=device),
            torch.full((layout.total,), -1.0, device=device),
            config, seed=0, device=device,
            fallback_log_path=tmp_path / "fb.json",
        )


def test_cuda_graph_refuses_cpu_device(tmp_path):
    """Chạy được cả trên máy không có CUDA: yêu cầu graph với device CPU
    phải báo lỗi rõ ràng thay vì âm thầm bỏ qua cờ."""
    layout = MSEGARCHParamLayout()

    def plain_log_joint(theta):
        return -(theta**2).sum()

    plain_log_joint.supports_batched_theta = True
    config = AdviConfig(n_steps=2, n_mc_samples=2, use_cuda_graph=True)
    with pytest.raises(ValueError, match="khong phai CUDA"):
        fit_mean_field_advi(
            plain_log_joint,
            torch.zeros(layout.total),
            torch.full((layout.total,), -1.0),
            config, seed=0, device=torch.device("cpu"),
            fallback_log_path=tmp_path / "fb.json",
        )
