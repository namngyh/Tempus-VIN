# tests/evaluation/test_probabilistic_metrics.py
import numpy as np

from raemf_mc.evaluation.probabilistic_metrics import compute_crps, compute_wis


def test_compute_crps_hand_computed_two_point_distribution():
    # Phan phoi thuc nghiem: hai diem dong xac suat {1, 3}. Gia tri thuc y=2.
    # Tich phan tay cua (F(x) - 1{x>=y})^2: 0.25 tren [1,2) + 0.25 tren [2,3) = 0.5.
    samples = np.array([1.0, 3.0])
    crps = compute_crps(samples, y=2.0)
    assert abs(crps - 0.5) < 1e-9


def test_compute_wis_realized_inside_interval():
    # 101 diem cach deu 0..100 de phan vi roi dung so nguyen:
    # quantile(0.25)=25, quantile(0.75)=75 (vi tri = (101-1)*q).
    samples = np.linspace(0, 100, 101)
    wis = compute_wis(samples, y=50.0, alpha_levels=(0.5,))
    # median=50, IS_0.5 = (75-25) + 0 (y nam trong [25,75]) = 50.
    # WIS = (1/(1+0.5)) * (0.5*|50-50| + (0.5/2)*50) = (1/1.5)*12.5 = 8.3333...
    assert abs(wis - 8.3333) < 1e-3


def test_compute_wis_realized_outside_interval():
    samples = np.linspace(0, 100, 101)
    wis = compute_wis(samples, y=90.0, alpha_levels=(0.5,))
    # l=25, u=75, y=90 > u: IS = (75-25) + (2/0.5)*(90-75) = 50 + 60 = 110.
    # WIS = (1/1.5) * (0.5*|90-50| + 0.25*110) = (1/1.5) * (20 + 27.5) = 31.6667
    assert abs(wis - 31.6667) < 1e-3


def test_crps_of_a_degenerate_forecast_is_absolute_error():
    """Dự báo tất định (mọi mẫu bằng nhau) phải cho CRPS đúng bằng sai số
    tuyệt đối: khi F là hàm bậc thang tại c, số hạng E|X-X'| bằng 0 và CRPS
    thu về |c - y|. Đây là điều kiện biên neo thang đo của CRPS vào một đại
    lượng đọc được, và nó bắt được lỗi hệ số 1/2 đặt sai chỗ — thứ mà test
    tính tay hai điểm không phân biệt được vì ở đó cả hai số hạng đều khác 0.
    """
    samples = np.full(50, 3.0)
    assert abs(compute_crps(samples, y=3.0) - 0.0) < 1e-12
    assert abs(compute_crps(samples, y=5.0) - 2.0) < 1e-12
    assert abs(compute_crps(samples, y=-1.0) - 4.0) < 1e-12


def test_wis_prefers_the_sharper_of_two_intervals_that_both_cover():
    """Tính sắc nét (sharpness): trong hai dự báo cùng bao phủ giá trị thực,
    dự báo hẹp hơn phải được điểm TỐT HƠN (WIS nhỏ hơn). Nếu thiếu tính chất
    này thì một mô hình dự báo khoảng rộng vô tận sẽ luôn thắng, và toàn bộ
    phần đánh giá xác suất của sub-project này mất ý nghĩa.
    """
    tight = np.linspace(45.0, 55.0, 101)
    wide = np.linspace(0.0, 100.0, 101)
    y = 50.0
    assert compute_wis(tight, y) < compute_wis(wide, y)


def test_wis_penalizes_a_miss_more_than_a_cover():
    """Giá trị thực nằm ngoài khoảng phải bị phạt nặng hơn nằm trong, với
    cùng một dự báo. Cùng với test sharpness ở trên, hai test này chốt cả
    hai chiều đánh đổi mà WIS tồn tại để cân bằng."""
    samples = np.linspace(0.0, 100.0, 101)
    assert compute_wis(samples, y=50.0) < compute_wis(samples, y=150.0)
