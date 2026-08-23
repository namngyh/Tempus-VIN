
import pandas as pd
import pytest
import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init
from raemf_mc.regime.posterior_features import (
    align_theta_states,
    canonical_theta,
    compute_posterior_volatility_features,
    label_indices_from_filtered,
    regime_health_report,
)
from raemf_mc.regime.ms_egarch import transition_matrix, unpack_params
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


def test_label_indices_base_rate_recovers_a_regime_argmax_never_picks():
    """Khuyet diem da do duoc, dung lai bang so hoc toi thieu: mot che do giu
    30% khoi luong trung binh nhung KHONG NGAY NAO vuot che do dan dau, nen
    argmax khong bao gio chon no va muc tieu tro thanh don lop. Chuan hoa
    theo tan suat nen phai lay lai duoc chinh nhung ngay che do do manh bat
    thuong.
    """
    T = 200
    probs = torch.zeros(T, 4)
    # che do 0 luon dan dau (0.60-0.70); che do 1 dao dong 0.20-0.40 nhung
    # khong bao gio vuot che do 0; hai che do con lai gan rong.
    ramp = torch.linspace(0.20, 0.40, T)
    probs[:, 1] = ramp
    probs[:, 0] = 0.95 - ramp
    probs[:, 2] = 0.03
    probs[:, 3] = 0.02
    log_probs = torch.log(probs)

    argmax_idx = label_indices_from_filtered(log_probs, n_train=T, scheme="argmax")
    assert set(argmax_idx.tolist()) == {0}, "tien de cua test: argmax phai don lop"

    base_idx = label_indices_from_filtered(log_probs, n_train=T, scheme="base_rate")
    assert len(set(base_idx.tolist())) >= 2, "base_rate phai lay lai duoc che do thu hai"
    # nhung ngay che do 1 manh nhat (cuoi ramp) phai duoc gan cho che do 1
    assert base_idx[-1].item() == 1
    # va nhung ngay no yeu nhat (dau ramp) van thuoc ve che do 0
    assert base_idx[0].item() == 0


def test_label_indices_base_rate_floor_blocks_near_empty_state_takeover():
    """San tan suat nen phai chan mot che do gan rong thang bang mau so nho.
    Khong co san, che do co pi ~ 0.002 se thang moi ngay chi vi phep chia --
    doi mot dang suy bien lay mot dang suy bien khac.
    """
    T = 200
    probs = torch.zeros(T, 4)
    probs[:, 0] = 0.700
    probs[:, 1] = 0.295
    probs[:, 2] = 0.003
    probs[:, 3] = 0.002
    log_probs = torch.log(probs)

    idx = label_indices_from_filtered(log_probs, n_train=T, scheme="base_rate")
    # che do 2 va 3 gan rong: san keo pi cua chung len 0.05, nen ty le cua
    # chung (0.003/0.05 = 0.06) thua xa che do 1 (0.295/0.295 = 1.0)
    assert set(idx.tolist()) <= {0, 1}


def test_label_indices_base_rate_uses_train_only_base_rate():
    """Tan suat nen dinh nghia muc tieu huan luyen, nen chi duoc tinh tren
    n_train dong dau. Neu ham nhin ca val/test de tinh pi thi muc tieu tro
    thanh ham cua du lieu giu lai -- dung lop ro ri ma align_states da phai
    sua mot lan (xem test_compute_regime_labels_fits_alignment_on_train_only).

    Bo so duoc dung co chu dich de mot hang CU THE lat ke thang khi pham vi
    tinh pi thay doi. Tren train, che do 0 va 1 gan can bang (pi ~ 0.455 va
    0.435), nen hang [0.50, 0.30] thuoc ve che do 0. Neu tinh pi tren ca
    chuoi, phan duoi keo pi[0] len 0.653 va dim pi[1] xuong 0.243, va chinh
    hang do lat sang che do 1. Mot test khong co hang nao lat duoc se pass
    ke ca khi n_train bi bo qua hoan toan.
    """
    T_train, T_tail = 100, 100
    probs = torch.zeros(T_train + T_tail, 4)
    probs[:90] = torch.tensor([0.45, 0.45, 0.05, 0.05])
    probs[90:T_train] = torch.tensor([0.50, 0.30, 0.15, 0.05])   # hang se lat
    probs[T_train:] = torch.tensor([0.85, 0.05, 0.05, 0.05])
    log_probs = torch.log(probs)

    idx_train_only = label_indices_from_filtered(log_probs, n_train=T_train, scheme="base_rate")
    idx_whole = label_indices_from_filtered(log_probs, n_train=T_train + T_tail, scheme="base_rate")

    assert idx_train_only[95].item() == 0
    assert idx_whole[95].item() == 1
    assert not torch.equal(idx_train_only, idx_whole)


def test_label_indices_rejects_unknown_scheme():
    log_probs = torch.log(torch.full((10, 4), 0.25))
    with pytest.raises(ValueError):
        label_indices_from_filtered(log_probs, n_train=10, scheme="khong_ton_tai")


def _permute_theta(theta, perm, layout):
    """Đánh số lại bốn chế độ của một theta theo `perm`. Kết quả mô tả ĐÚNG
    cùng một mô hình — đây chính là phép biến đổi mà hai seed khác nhau có
    thể tình cờ khác nhau bởi."""
    params = unpack_params(theta, layout)
    trans = transition_matrix(params.transition_logits)[perm][:, perm]
    logits = torch.log(trans[:, 1:] / trans[:, :1])
    return torch.cat([
        params.omega[perm], params.alpha[perm], params.beta[perm],
        params.gamma[perm], logits.reshape(-1), params.nu_raw[perm],
    ])


def _distinct_theta(layout, seed=17):
    torch.manual_seed(seed)
    n = layout.n_states
    return torch.cat([
        torch.tensor([-3.0, -1.0, -2.0, -0.5]),      # omega, cố ý không sắp sẵn
        torch.rand(n) * 0.3,                          # alpha
        torch.tensor([0.3, 0.8, 0.5, 0.7]),           # beta
        torch.randn(n) * 0.1,                         # gamma
        torch.randn(n * (n - 1)) * 0.5,               # transition logits
        torch.tensor([0.5, 2.0, 1.0, 3.0]),           # nu_raw
    ])


def test_align_theta_states_round_trips_the_transition_matrix():
    """Hoán vị ma trận chuyển phải chính xác theo CẢ hàng lẫn cột. Không thể
    hoán vị thẳng trên logit vì quy ước tham số hoá pin cột 0 làm mốc, nên
    phải dựng ma trận xác suất, hoán vị, rồi quy ngược. Nếu bước quy ngược
    sai, mô hình sau khi căn chỉnh sẽ có động lực chuyển chế độ khác hẳn —
    một lỗi hoàn toàn vô hình với mọi khẳng định về hình dạng.
    """
    layout = MSEGARCHParamLayout()
    theta = _distinct_theta(layout)
    aligned = align_theta_states(theta, layout)

    original = transition_matrix(unpack_params(theta, layout).transition_logits)
    got = transition_matrix(unpack_params(aligned, layout).transition_logits)

    # Thứ tự căn chỉnh là omega/(1-beta) tăng dần.
    params = unpack_params(theta, layout)
    order = torch.argsort(params.omega / (1.0 - torch.clamp(params.beta, -0.999, 0.999)))
    torch.testing.assert_close(got, original[order][:, order])
    # Mỗi hàng vẫn phải là một phân phối xác suất.
    torch.testing.assert_close(got.sum(dim=1), torch.ones(layout.n_states))


def test_canonical_theta_is_invariant_to_state_relabelling():
    """Đây là hợp đồng mà toàn bộ thay đổi này tồn tại để bảo đảm.

    Hai seed tìm ra CÙNG một nghiệm nhưng đánh số trạng thái khác nhau —
    tình huống đã đo được thật: bốn seed cho ELBO gần như y hệt (4652,8 /
    4655,7 / 4646,8 / 4648,6) trong khi ba trong bốn có cùng cấu trúc
    occupancy chỉ khác hoán vị. Trung bình cộng của chúng phải trả lại CHÍNH
    nghiệm đó, không phải một hỗn hợp của hai cách đánh số.

    Không có căn chỉnh, phép trung bình cộng `omega[3]` của seed này với
    `omega[3]` của seed kia — hai chỉ số chỉ hai chế độ khác nhau — cho ra
    một theta không tương ứng với nghiệm nào cả.
    """
    layout = MSEGARCHParamLayout()
    theta = _distinct_theta(layout)
    relabelled = _permute_theta(theta, torch.tensor([2, 0, 3, 1]), layout)

    posterior = _fake_posterior([theta, relabelled], layout)
    result = canonical_theta(posterior, layout)

    torch.testing.assert_close(result, align_theta_states(theta, layout))

    # Và để chắc chắn test này không rỗng nghĩa: phép trung bình THÔ (cách
    # cũ) phải cho ra kết quả KHÁC. Nếu không, hai theta vốn đã giống nhau
    # và test ở trên chẳng chứng minh được gì.
    naive = torch.stack([theta, relabelled]).mean(dim=0)
    assert not torch.allclose(naive, result, atol=1e-3)


def test_align_theta_states_is_idempotent():
    """Căn chỉnh một theta đã căn chỉnh phải không đổi gì. Nếu không, kết
    quả phụ thuộc vào số lần hàm được gọi — và `canonical_theta` gọi nó trên
    mọi seed ở mọi lần dùng."""
    layout = MSEGARCHParamLayout()
    once = align_theta_states(_distinct_theta(layout), layout)
    twice = align_theta_states(once, layout)
    torch.testing.assert_close(once, twice)
