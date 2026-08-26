# tests/test_evaluation_integration_smoke.py
"""Walk-forward OOS chạy THẬT trên dữ liệu VN-Index, quy mô smoke.

Khác mọi integration test của các sub-project trước ở đúng một điểm quyết
định: tại mỗi mốc, mô hình chỉ nhìn thấy dữ liệu TRƯỚC mốc đó, rồi bị chấm
điểm bằng lợi suất đã thực sự xảy ra sau đó. Các test trước đều fit và đánh
giá trên cùng một cửa sổ.

Mọi con số in ra ở đây là để xác nhận đường ống nối đúng, KHÔNG phải kết quả
nghiên cứu — xem `configs/eval_smoke.yaml` và docstring của
`run_walk_forward_evaluation`.
"""
from pathlib import Path

import numpy as np
import torch
import yaml

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.evaluation.var_backtest import CHI2_1DF_95
from raemf_mc.evaluation.walk_forward import run_walk_forward_evaluation
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout
from raemf_mc.runtime.hardware import select_device

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "eval_smoke.yaml"


def test_walk_forward_evaluation_end_to_end_on_real_data(tmp_path):
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ohlcv = load_vnindex_ohlcv().iloc[-config["window_sessions"] :]
    log_returns = compute_log_returns(ohlcv)

    horizon = config["horizon"]
    n_cutoffs = config["n_cutoffs"]
    # Trải đều các mốc trên nửa sau của cửa sổ: nửa đầu để dành làm dữ liệu
    # fit cho mốc sớm nhất, và mốc cuối phải còn đủ `horizon` phiên phía sau
    # để chấm điểm.
    first = len(log_returns) // 2
    last = len(log_returns) - horizon
    cutoffs = np.linspace(first, last, n_cutoffs, dtype=int).tolist()
    assert cutoffs[-1] + horizon <= len(log_returns)

    layout = MSEGARCHParamLayout()
    device = select_device(config.get("device_preference", "auto"))

    results_df, kupiec = run_walk_forward_evaluation(
        log_returns,
        cutoffs=cutoffs,
        horizon=horizon,
        ms_egarch_advi=AdviConfig(**config["ms_egarch_advi"]),
        ms_egarch_prior=HierarchicalPriorConfig(**config["ms_egarch_prior"]),
        mu_advi=AdviConfig(**config["mu_advi"]),
        mu_n_draws=config["mu_n_draws"],
        mu_prior_scale=config["mu_prior_scale"],
        mu_min_effective_observations=config["mu_min_effective_observations"],
        mu_min_effective_fraction=config["mu_min_effective_fraction"],
        mc_n_paths=config["mc_n_paths"],
        layout=layout,
        seeds=config["seeds"],
        device=device,
        fallback_log_dir=tmp_path,
        var_alphas=tuple(config["var_alphas"]),
        generator=torch.Generator(device=device).manual_seed(0),
    )

    print(f"\nWalk-forward OOS (smoke-scale, {n_cutoffs} moc, horizon={horizon}):")
    print(results_df.to_string(index=False))
    print(f"Kupiec: {kupiec}")
    n_breach_95 = int(results_df["breached_95"].sum())
    print(
        f"Vi pham VaR_95: {n_breach_95}/{n_cutoffs} "
        f"(ky vong {0.05 * n_cutoffs:.1f}) -- co mau nay KHONG du de ket luan"
    )

    assert len(results_df) == n_cutoffs
    assert list(results_df["cutoff"]) == cutoffs
    assert results_df["crps"].notna().all()
    assert results_df["wis"].notna().all()
    assert (results_df["crps"] >= 0).all()
    assert (results_df["wis"] >= 0).all()
    assert (results_df["var_99"] >= results_df["var_95"]).all()
    assert np.isfinite(results_df["realized_return"]).all()

    for key, value in kupiec.items():
        assert np.isfinite(value), f"{key} phải hữu hạn"
        assert value >= 0.0

    # KHÔNG khẳng định rằng VaR được hiệu chỉnh đúng. Với 8 mốc và p=0.05, số
    # vi phạm kỳ vọng là 0.4: mọi kết quả từ 0 đến 2 vi phạm đều nằm trong
    # vùng không phân biệt được, nên một khẳng định kiểu "kupiec < 3.841"
    # sẽ pass vì kiểm định không có sức mạnh, chứ không phải vì mô hình đúng.
    # Ghi lại con số để đọc, và ghi rõ nó chưa kết luận được gì.
    print(
        f"CHI2_1DF_95 = {CHI2_1DF_95} (nguong bac bo o muc 95%, chi co y nghia "
        f"khi so moc lon hon nhieu)"
    )


def test_walk_forward_realized_returns_match_the_historical_record():
    """Lợi suất "thực tế" mà harness chấm điểm phải đúng bằng tổng lợi suất
    log của đúng những phiên đó trong lịch sử — không lệch chỉ số, không lệch
    một phiên.

    Lệch một phiên là loại lỗi không làm gãy gì cả: mọi bất biến vẫn đúng,
    CRPS/WIS vẫn hữu hạn, và kết quả vẫn trông hợp lý. Nó chỉ âm thầm chấm
    điểm dự báo bằng một khoảng thời gian khác khoảng đã mô phỏng. Test này
    không chạy fit nào nên rẻ, và nó khoá đúng chỗ đó.
    """
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ohlcv = load_vnindex_ohlcv().iloc[-config["window_sessions"] :]
    log_returns = compute_log_returns(ohlcv)
    horizon = config["horizon"]

    # `compute_log_returns` BỎ dòng đầu (r_t cần P_{t-1}), nên
    # `log_returns.iloc[k]` ứng với `ohlcv.iloc[k+1]` so với `ohlcv.iloc[k]`.
    # Lệch một phiên ở đây chính là cái test này tồn tại để bắt — và nó đã
    # bắt được ngay lần viết đầu tiên, khi tôi dùng `iloc[cutoff-1]`.
    assert len(log_returns) == len(ohlcv) - 1

    for cutoff in (600, 900, 1200):
        expected = float(log_returns.iloc[cutoff : cutoff + horizon].sum())
        # Đối chiếu độc lập qua giá: tổng log-return của h phiên bằng
        # log(P_cuoi / P_dau) của chính khoảng đó.
        price_start = ohlcv["close"].iloc[cutoff]
        price_end = ohlcv["close"].iloc[cutoff + horizon]
        assert abs(expected - float(np.log(price_end / price_start))) < 1e-6

        # Và ngày tháng phải khớp, không chỉ giá trị: phiên đầu tiên được
        # cộng vào phải là phiên NGAY SAU ngày của giá mở đầu.
        assert log_returns.index[cutoff] == ohlcv.index[cutoff + 1]
        assert log_returns.index[cutoff + horizon - 1] == ohlcv.index[cutoff + horizon]


def test_eval_smoke_config_cutoffs_leave_room_for_the_horizon():
    """Cấu hình phải tự nhất quán: mốc cuối cộng `horizon` không được vượt
    quá cửa sổ. Nếu vượt, `run_walk_forward_evaluation` sẽ raise — nhưng chỉ
    sau khi đã fit xong mọi mốc trước đó, tức lãng phí hàng chục phút mới
    biết config sai."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    n = config["window_sessions"]
    horizon = config["horizon"]
    first = n // 2
    last = n - horizon
    cutoffs = np.linspace(first, last, config["n_cutoffs"], dtype=int).tolist()
    assert cutoffs[0] > 0
    assert cutoffs[-1] + horizon <= n
    assert len(set(cutoffs)) == len(cutoffs), "các mốc phải phân biệt"
