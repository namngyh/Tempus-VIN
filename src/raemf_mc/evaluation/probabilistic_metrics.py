from __future__ import annotations

import numpy as np


def compute_crps(samples: np.ndarray, y: float) -> float:
    """CRPS của phân phối thực nghiệm cho bởi `samples` đối chiếu với giá trị
    thực `y`, qua dạng đóng chính xác cho CDF thực nghiệm:
    CRPS(F, y) = E|X - y| - 0.5*E|X - X'|, với X, X' iid ~ F.

    KHÔNG phải xấp xỉ — với phân phối rời rạc thực nghiệm, biểu thức này
    bằng đúng định nghĩa tích phân.

    Chi phí O(M^2) theo số mẫu, do ma trận |X_i - X_j| đầy đủ. Chấp nhận
    được ở M cỡ vài trăm đến một nghìn path Monte Carlo; ở M = 20 000 (quy
    mô mà GPU làm được, xem docs/perf_batching_notes.md) ma trận này là 4e8
    phần tử ~ 3,2 GB float64 và sẽ vỡ bộ nhớ. Nếu cần M lớn, phải đổi sang
    dạng sắp xếp O(M log M):
      CRPS = (2/M^2) * sum_i (x_(i) - y) * (M*1{y < x_(i)} - i + 0.5)
    Chưa cài đặt vì quy mô hiện tại chưa cần, và một công thức chưa được
    dùng là một công thức chưa được kiểm chứng.
    """
    samples = np.asarray(samples, dtype=np.float64)
    term1 = float(np.mean(np.abs(samples - y)))
    diffs = np.abs(samples[:, None] - samples[None, :])
    term2 = 0.5 * float(np.mean(diffs))
    return term1 - term2


def compute_wis(
    samples: np.ndarray,
    y: float,
    alpha_levels: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02),
) -> float:
    """Weighted Interval Score (Bracher và cộng sự, 2021) của phân phối thực
    nghiệm cho bởi `samples` đối chiếu với giá trị thực `y`.

    `alpha_levels` là mức của khoảng TRUNG TÂM HAI PHÍA (0.5 -> khoảng 50%,
    tức phân vị [0.25, 0.75]) — KHÔNG liên quan tới quy ước alpha một phía
    của VaR trong dự án này (`compute_var(alpha=0.95)` nghĩa là đuôi 5%).
    Hai quy ước này dùng cùng một chữ cái cho hai thứ khác nhau; không được
    lẫn. Đây là lý do tham số ở đây tên là `alpha_levels` chứ không phải
    `alphas` như trong `risk/metrics.py`.
    """
    samples = np.asarray(samples, dtype=np.float64)
    median = float(np.median(samples))
    k = len(alpha_levels)
    total = 0.5 * abs(y - median)
    for alpha in alpha_levels:
        lo = float(np.quantile(samples, alpha / 2))
        hi = float(np.quantile(samples, 1 - alpha / 2))
        interval_score = hi - lo
        if y < lo:
            interval_score += (2.0 / alpha) * (lo - y)
        elif y > hi:
            interval_score += (2.0 / alpha) * (y - hi)
        total += (alpha / 2.0) * interval_score
    return total / (k + 0.5)
