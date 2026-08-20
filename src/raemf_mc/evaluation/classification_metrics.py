from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from raemf_mc.regime.state_alignment import STATE_NAMES


def compute_classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = STATE_NAMES
) -> dict:
    """Macro F1, recall từng lớp, và ma trận nhầm lẫn có nhãn. Dùng lại
    scikit-learn thay vì tự cài đặt — chỉ phần gán nhãn/đóng gói là riêng
    của dự án này.

    `labels` phải được truyền tường minh ở mọi lời gọi có thể có lớp vắng
    mặt, và phải liệt kê ĐỦ bốn chế độ: nếu để scikit-learn tự suy ra tập
    nhãn từ dữ liệu, một lớp không xuất hiện trong y_true lẫn y_pred sẽ bị
    loại khỏi mẫu số của macro F1, và con số thu được sẽ là macro trên 3
    lớp trong khi báo cáo nói 4 — im lặng thổi phồng kết quả.

    Điều này KHÔNG phải giả định lý thuyết trên dữ liệu VN-Index: đo được
    trên cửa sổ ebm_smoke, lớp `Bull` chỉ có 2 mẫu train và 0 mẫu val/test,
    nên đây đúng là tình huống sẽ xảy ra ngay lần chạy đầu tiên.

    `zero_division=0` cho lớp vắng mặt điểm 0 chứ không phải NaN: một lớp
    không bao giờ được dự đoán đúng thì recall của nó là 0, và macro F1 phải
    chịu hình phạt đó thay vì lặng lẽ bỏ qua lớp.
    """
    labels_list = list(labels)
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)
    )
    recalls = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    recall_by_class = {label: float(r) for label, r in zip(labels_list, recalls)}
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)
    return {
        "macro_f1": macro_f1,
        "recall_by_class": recall_by_class,
        "confusion_matrix": cm_df,
    }


def compute_nll(log_probs: torch.Tensor, y_true_idx: torch.Tensor) -> float:
    """Negative log-likelihood trung bình, tính từ LOG-xác suất (ví dụ do
    `apply_temperature_log_prob` sinh ra), không bao giờ từ
    `log(probability)` trực tiếp — đường đó underflow thành -inf ở những dự
    đoán sai một cách tự tin, đúng lỗi mà sub-project 2 đã tìm ra và sửa một
    lần cho chính phép tính này.
    """
    return float(torch.nn.functional.nll_loss(log_probs, y_true_idx).item())
