# tests/evaluation/test_classification_metrics.py
import torch

from raemf_mc.evaluation.classification_metrics import (
    compute_classification_report,
    compute_nll,
)
from raemf_mc.regime.state_alignment import STATE_NAMES


def test_compute_classification_report_hand_computed():
    # 3 classes, one wrong prediction (Bear predicted as Sideway once).
    y_true = ["Bull", "Bull", "Bear", "Bear", "Sideway"]
    y_pred = ["Bull", "Bull", "Bear", "Sideway", "Sideway"]
    labels = ["Bull", "Sideway", "Bear", "Stress"]

    report = compute_classification_report(y_true, y_pred, labels=labels)

    # Bull: 2/2 correct -> recall 1.0. Sideway: 1/1 correct -> recall 1.0.
    # Bear: 1/2 correct -> recall 0.5. Stress: no true instances -> recall 0.0.
    assert report["recall_by_class"]["Bull"] == 1.0
    assert report["recall_by_class"]["Sideway"] == 1.0
    assert abs(report["recall_by_class"]["Bear"] - 0.5) < 1e-9
    assert report["recall_by_class"]["Stress"] == 0.0

    # macro F1 over 4 labels: Bull F1=1.0; Sideway precision=0.5, recall=1.0
    # -> F1=0.6667; Bear precision=1.0, recall=0.5 -> F1=0.6667; Stress 0.
    # macro = (1.0 + 0.6667 + 0.6667 + 0.0) / 4 = 0.5833
    assert abs(report["macro_f1"] - 0.5833) < 1e-3

    cm = report["confusion_matrix"]
    assert list(cm.index) == labels
    assert list(cm.columns) == labels
    assert cm.loc["Bear", "Sideway"] == 1  # the one misclassification
    assert cm.loc["Bull", "Bull"] == 2


def test_compute_nll_uses_log_domain_not_log_of_probability():
    # Two samples, true class index 0 and 1, log-probs already computed
    # (as apply_temperature_log_prob would produce) rather than probabilities.
    log_probs = torch.log_softmax(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), dim=1)
    y_true_idx = torch.tensor([0, 1])
    nll = compute_nll(log_probs, y_true_idx)
    expected = float(-log_probs[0, 0].item() - log_probs[1, 1].item()) / 2
    assert abs(nll - expected) < 1e-6
    assert nll < 0.5  # sanity: confident-correct predictions have low NLL


def test_macro_f1_counts_absent_classes_instead_of_silently_dropping_them():
    """Cái bẫy này KHÔNG phải giả định: đo được trên cửa sổ ebm_smoke, lớp
    `Bull` có 2 mẫu train và 0 mẫu val/test. Nếu tập nhãn được để cho
    scikit-learn tự suy ra từ dữ liệu, lớp vắng mặt biến mất khỏi mẫu số và
    macro F1 trở thành macro trên 3 lớp trong khi báo cáo nói 4 — im lặng
    thổi phồng kết quả đúng ở tình huống sẽ gặp ngay lần chạy thật đầu tiên.

    Ở đây cả `Bull` lẫn `Stress` đều vắng mặt hoàn toàn. Ba lớp có mặt được
    dự đoán đúng tuyệt đối, nên macro-trên-2-lớp = 1.0 còn macro-trên-4-lớp
    = 2/4 = 0.5. Khoảng cách đó chính là thứ test này bảo vệ.
    """
    y_true = ["Sideway", "Sideway", "Bear"]
    y_pred = ["Sideway", "Sideway", "Bear"]
    labels = ["Bull", "Sideway", "Bear", "Stress"]

    report = compute_classification_report(y_true, y_pred, labels=labels)

    assert report["recall_by_class"]["Sideway"] == 1.0
    assert report["recall_by_class"]["Bear"] == 1.0
    assert report["recall_by_class"]["Bull"] == 0.0
    assert report["recall_by_class"]["Stress"] == 0.0
    assert abs(report["macro_f1"] - 0.5) < 1e-9, (
        "macro F1 phải chia cho cả 4 lớp, kể cả lớp vắng mặt"
    )
    assert set(report["recall_by_class"]) == set(labels)


def test_default_labels_cover_all_four_state_names():
    """Mặc định của `labels` phải là đủ bốn chế độ, không phải suy ra từ dữ
    liệu — người gọi quên truyền `labels` vẫn phải nhận macro trên 4 lớp."""
    report = compute_classification_report(["Sideway"], ["Sideway"])
    assert set(report["recall_by_class"]) == set(STATE_NAMES)
    assert abs(report["macro_f1"] - 0.25) < 1e-9


def test_compute_nll_survives_a_confidently_wrong_prediction():
    """Guard cho đúng lỗi underflow mà sub-project 2 đã sửa một lần: một dự
    đoán sai một cách rất tự tin phải cho NLL lớn và HỮU HẠN. Đi qua
    `log(probability)` thì xác suất lớp đúng underflow về 0 và NLL thành
    inf, làm hỏng mọi con số trung bình phía sau."""
    logits = torch.tensor([[60.0, -60.0]])
    log_probs = torch.log_softmax(logits, dim=1)
    nll = compute_nll(log_probs, torch.tensor([1]))
    assert torch.isfinite(torch.tensor(nll))
    assert nll > 100.0
