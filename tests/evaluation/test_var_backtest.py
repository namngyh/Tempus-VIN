# tests/evaluation/test_var_backtest.py
import math

import pytest

from raemf_mc.evaluation.var_backtest import CHI2_1DF_95, compute_kupiec_lr


def test_compute_kupiec_lr_hand_computed():
    # n=10 quan sat, x=1 vi pham, p=0.05 (khop ty le vi pham ky vong cua VaR_95).
    # Tinh tay (log tu nhien):
    # log_L0 = 9*ln(0.95) + 1*ln(0.05) = -3.4573720
    # log_L1 = 9*ln(0.9)  + 1*ln(0.1)  = -3.2508296   (phat = 1/10 = 0.1)
    # LR = -2*(log_L0 - log_L1) = 0.4130848
    lr = compute_kupiec_lr(n=10, x=1, p=0.05)
    assert abs(lr - 0.4130848) < 1e-5


def test_compute_kupiec_lr_zero_violations_edge_case():
    # x=0: log_L1 = 0 chinh xac (mo hinh bao hoa khop hoan hao phat=0).
    # log_L0 = 5*ln(0.95) = -0.2564665
    # LR = -2*(-0.2564665 - 0) = 0.512933
    lr = compute_kupiec_lr(n=5, x=0, p=0.05)
    assert abs(lr - 0.512933) < 1e-5
    assert math.isfinite(lr)


def test_compute_kupiec_lr_all_violations_edge_case():
    # x=n: log_L1 = 0 chinh xac (mo hinh bao hoa khop hoan hao phat=1).
    lr = compute_kupiec_lr(n=5, x=5, p=0.05)
    assert math.isfinite(lr)
    assert lr > 0


def test_kupiec_lr_is_near_zero_when_observed_rate_matches_p():
    """Neo thang đo: khi tỷ lệ vi phạm quan sát bằng đúng `p`, hai mô hình
    trùng nhau nên LR phải bằng 0. Nếu thiếu điều kiện biên này, một lỗi dấu
    hoặc hệ số 2 đặt sai vẫn có thể lọt qua ba test tính tay ở trên."""
    assert abs(compute_kupiec_lr(n=100, x=5, p=0.05)) < 1e-12
    assert abs(compute_kupiec_lr(n=20, x=2, p=0.10)) < 1e-12


def test_kupiec_lr_grows_as_observed_rate_departs_from_p():
    """LR phải tăng đơn điệu khi tỷ lệ quan sát rời xa `p`. Đây là tính chất
    khiến nó dùng được làm kiểm định, và nó không suy ra được từ các giá trị
    tính tay đơn lẻ."""
    lrs = [compute_kupiec_lr(n=100, x=k, p=0.05) for k in (5, 8, 12, 20, 30)]
    assert all(a < b for a, b in zip(lrs, lrs[1:]))


def test_kupiec_lr_rejects_a_clearly_miscalibrated_var():
    """Một VaR 95% vi phạm 30/100 lần phải bị bác bỏ dứt khoát ở mức 95%.
    Đây là ca duy nhất trong file này mà kiểm định thực sự KẾT LUẬN được điều
    gì đó — và nó cần n lớn. Ở n=10 như quy mô smoke của sub-project này thì
    không ca nào làm được vậy, đó chính là lý do docstring cảnh báo."""
    assert compute_kupiec_lr(n=100, x=30, p=0.05) > CHI2_1DF_95
    # Doi lai: o n=10, ngay ca ty le vi pham gap 4 lan ky vong cung KHONG bi
    # bac bo -- kiem dinh khong co suc manh o co mau do.
    assert compute_kupiec_lr(n=10, x=2, p=0.05) < CHI2_1DF_95


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0, "x": 0, "p": 0.05},
        {"n": 10, "x": 11, "p": 0.05},
        {"n": 10, "x": -1, "p": 0.05},
        {"n": 10, "x": 1, "p": 0.0},
        {"n": 10, "x": 1, "p": 1.0},
    ],
)
def test_kupiec_lr_rejects_invalid_inputs(kwargs):
    """Đầu vào vô nghĩa phải báo lỗi thay vì trả về NaN lặng lẽ chảy xuống
    bảng kết quả cuối."""
    with pytest.raises(ValueError):
        compute_kupiec_lr(**kwargs)
