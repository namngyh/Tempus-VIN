# tests/evaluation/test_walk_forward.py
from pathlib import Path

import pandas as pd
import pytest
import torch

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.evaluation.walk_forward import run_walk_forward_evaluation
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout


def test_run_walk_forward_evaluation_shape_and_invariants(tmp_path):
    layout = MSEGARCHParamLayout()
    torch.manual_seed(0)
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    log_returns = pd.Series(torch.randn(120).numpy() * 0.01, index=dates)

    advi_config = AdviConfig(n_steps=20, learning_rate=0.05, warmup_steps=5,
                             elbo_ma_window=5, early_stop_patience=20)
    prior_config = HierarchicalPriorConfig()
    fallback_dir = tmp_path / "fallbacks"
    fallback_dir.mkdir()

    results_df, kupiec = run_walk_forward_evaluation(
        log_returns, cutoffs=[80, 100], horizon=5,
        ms_egarch_advi=advi_config, ms_egarch_prior=prior_config,
        mu_advi=advi_config, mu_n_draws=3, mu_prior_scale=0.01,
        mu_min_effective_observations=10.0, mu_min_effective_fraction=0.05,
        mc_n_paths=50, layout=layout, seeds=[0], device=torch.device("cpu"),
        fallback_log_dir=fallback_dir, generator=torch.Generator().manual_seed(1),
    )

    assert list(results_df["cutoff"]) == [80, 100]
    assert results_df["crps"].ge(0).all()
    assert results_df["wis"].ge(0).all()
    for col in ("var_95", "var_99"):
        assert results_df[col].notna().all()
    assert set(kupiec.keys()) == {"kupiec_lr_95", "kupiec_lr_99"}
    for v in kupiec.values():
        assert v == v  # khong phai NaN
        assert v >= 0.0


def test_run_walk_forward_evaluation_rejects_cutoff_too_close_to_end():
    layout = MSEGARCHParamLayout()
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    log_returns = pd.Series([0.01] * 30, index=dates)
    advi_config = AdviConfig(n_steps=5)
    prior_config = HierarchicalPriorConfig()
    with pytest.raises(ValueError, match="horizon"):
        run_walk_forward_evaluation(
            log_returns, cutoffs=[28], horizon=5,
            ms_egarch_advi=advi_config, ms_egarch_prior=prior_config,
            mu_advi=advi_config, mu_n_draws=2, mu_prior_scale=0.01,
            mu_min_effective_observations=10.0, mu_min_effective_fraction=0.05,
            mc_n_paths=10, layout=layout, seeds=[0], device=torch.device("cpu"),
            fallback_log_dir=Path("."),
        )


def _tiny_walk_forward(log_returns, tmp_path, cutoffs=(60,), horizon=5):
    layout = MSEGARCHParamLayout()
    advi_config = AdviConfig(n_steps=15, learning_rate=0.05, warmup_steps=3,
                             elbo_ma_window=5, early_stop_patience=100)
    fallback_dir = tmp_path / "fb"
    fallback_dir.mkdir(exist_ok=True)
    return run_walk_forward_evaluation(
        log_returns, cutoffs=list(cutoffs), horizon=horizon,
        ms_egarch_advi=advi_config, ms_egarch_prior=HierarchicalPriorConfig(),
        mu_advi=advi_config, mu_n_draws=3, mu_prior_scale=0.01,
        mu_min_effective_observations=10.0, mu_min_effective_fraction=0.05,
        mc_n_paths=40, layout=layout, seeds=[0], device=torch.device("cpu"),
        fallback_log_dir=fallback_dir,
        generator=torch.Generator().manual_seed(1),
    )


def test_walk_forward_ignores_data_beyond_the_scored_horizon(tmp_path):
    """Không được nhìn thấy gì sau `cutoff + horizon`.

    Thay toàn bộ phần đuôi bằng số vô lý: MỌI cột phải giữ nguyên bit-đối-bit.
    Nếu một thay đổi tương lai vô tình tính `train_mean` trên cả chuỗi, hay
    chuẩn hoá bằng thống kê toàn cục, test này gãy — trong khi test hình dạng
    thì không. Rò rỉ kiểu đó chỉ qua ĐÚNG MỘT con số vẫn đủ làm mọi kết luận
    OOS mất giá trị, và nó vô hình với mắt thường.
    """
    torch.manual_seed(4)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    base = pd.Series(torch.randn(100).numpy() * 0.01, index=dates)

    polluted = base.copy()
    polluted.iloc[65:] = 5.0          # sau cutoff(60) + horizon(5)

    clean_df, clean_kupiec = _tiny_walk_forward(base, tmp_path)
    dirty_df, dirty_kupiec = _tiny_walk_forward(polluted, tmp_path)

    pd.testing.assert_frame_equal(clean_df, dirty_df)
    assert clean_kupiec == dirty_kupiec


def test_walk_forward_var_comes_from_the_fit_not_from_the_realized_future(tmp_path):
    """Phân biệt sắc hơn: đổi dữ liệu NẰM TRONG chân trời được chấm điểm.

    `realized_return` PHẢI đổi (nó chính là dữ liệu đó), nhưng `var_95` và
    `var_99` PHẢI GIỮ NGUYÊN — chúng đến từ phân phối mô phỏng, mà phân phối
    đó chỉ được fit trên dữ liệu trước mốc. Nếu VaR cũng đổi thì con số dự
    báo đang được tính từ chính thứ nó phải dự báo, và toàn bộ kiểm định
    Kupiec trở thành vòng lặp tự khẳng định.
    """
    torch.manual_seed(5)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    base = pd.Series(torch.randn(100).numpy() * 0.01, index=dates)

    shifted = base.copy()
    shifted.iloc[60:65] = -0.08       # trong chan troi cua cutoff=60

    clean_df, _ = _tiny_walk_forward(base, tmp_path)
    shifted_df, _ = _tiny_walk_forward(shifted, tmp_path)

    assert clean_df["realized_return"].iloc[0] != shifted_df["realized_return"].iloc[0]
    assert clean_df["var_95"].iloc[0] == shifted_df["var_95"].iloc[0]
    assert clean_df["var_99"].iloc[0] == shifted_df["var_99"].iloc[0]
    # KHONG khang dinh rang lo -40% trong 5 ngay se thung VaR_95. Da do:
    # o quy mo smoke nay (15 buoc ADVI), var_95 = 1.66 -- mo hinh noi "5% kha
    # nang lo tu 166% tro len trong 5 ngay". Mot khoan lo 40% khong thung
    # noi. Do la giai han da biet cua fit chua hoi tu, khong phai loi cua
    # harness, va no duoc ghi vao model card. Khang dinh dieu nguoc lai o day
    # se la ma hoa mot khiem khuyet thanh ky vong: khi fit duoc sua, test se
    # gay vi ly do sai.


def test_walk_forward_var_99_is_never_below_var_95(tmp_path):
    """Bất biến giữa hai mức: phân vị xa hơn ở đuôi không thể cho khoản lỗ
    nhỏ hơn. Bất biến này đã được chứng minh trong `risk/metrics.py` nhưng
    chưa từng được kiểm ở tầng walk-forward, nơi nó đi qua một đường cộng
    drift riêng."""
    torch.manual_seed(6)
    dates = pd.date_range("2020-01-01", periods=110, freq="D")
    returns = pd.Series(torch.randn(110).numpy() * 0.012, index=dates)
    df, _ = _tiny_walk_forward(returns, tmp_path, cutoffs=(70, 90))
    assert (df["var_99"] >= df["var_95"]).all()
