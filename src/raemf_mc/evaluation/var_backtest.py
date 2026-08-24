from __future__ import annotations

import math

# Giá trị tới hạn của phân phối chi-bình-phương 1 bậc tự do ở mức 95%. Đặt
# thành hằng số có tên vì con số 3.841 xuất hiện trong mọi diễn giải của
# thống kê Kupiec và một số ma thuật rải rác thì không ai kiểm được.
CHI2_1DF_95 = 3.841


def compute_kupiec_lr(n: int, x: int, p: float) -> float:
    """Thống kê tỷ số hợp lý (likelihood-ratio) của Kupiec (1995) cho kiểm
    định độ phủ vô điều kiện của VaR.

    `n`: số quan sát. `x`: số lần vi phạm VaR (lỗ thực tế vượt VaR).
    `p`: xác suất vi phạm kỳ vọng của mức VaR đó (0.05 cho VaR 95%).

    Dưới giả thuyết H0 (VaR hiệu chỉnh đúng), LR ~ chi-bình-phương(1); so
    với `CHI2_1DF_95` = 3.841 để kiểm tra xem tỷ lệ vi phạm quan sát được có
    khác `p` một cách có ý nghĩa hay không.

    CẢNH BÁO VỀ SỨC MẠNH THỐNG KÊ: với `n` nhỏ (5-10 mốc walk-forward mà
    sub-project này chạy ở quy mô smoke), kiểm định này gần như KHÔNG có sức
    mạnh — nó xác nhận CÔNG THỨC chạy đúng, không phải một kết luận đáng tin
    về hiệu chỉnh. Với p=0.05 và n=10, số vi phạm kỳ vọng là 0.5: quan sát 0
    hay 1 lần vi phạm đều không phân biệt được một VaR đúng với một VaR sai
    hệ thống. Mọi báo cáo dùng con số này phải nói rõ điều đó.
    """
    if n <= 0:
        raise ValueError(f"n phải dương, nhận {n}")
    if not 0 <= x <= n:
        raise ValueError(f"x phải nằm trong [0, n], nhận x={x}, n={n}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p phải nằm trong (0, 1), nhận {p}")

    log_l0 = (n - x) * math.log(1 - p) + x * math.log(p)
    if x == 0 or x == n:
        # Mô hình không ràng buộc (bão hoà) với phat đúng bằng 0 hoặc 1 khớp
        # hoàn hảo: hợp lý bằng 1, log-hợp-lý bằng 0. Tính thẳng
        # x*log(phat) hoặc (n-x)*log(1-phat) sẽ cho 0*log(0) = NaN trong dấu
        # phẩy động, dù giới hạn giải tích đúng bằng 0.
        log_l1 = 0.0
    else:
        phat = x / n
        log_l1 = (n - x) * math.log(1 - phat) + x * math.log(phat)
    return -2.0 * (log_l0 - log_l1)
