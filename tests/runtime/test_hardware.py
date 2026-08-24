import torch
import pytest
from raemf_mc.runtime.hardware import select_device, hardware_report


def test_select_device_cpu_explicit():
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_auto_falls_back_to_cpu_without_cuda():
    device = select_device("auto")
    if not torch.cuda.is_available():
        assert device == torch.device("cpu")


def test_select_device_cuda_raises_when_unavailable():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            select_device("cuda")


def test_hardware_report_has_expected_keys():
    report = hardware_report()
    assert set(report.keys()) == {
        "torch_version", "cuda_available", "cpu_count",
        "device_preference", "selected_device",
    }
    assert report["torch_version"] == torch.__version__


def test_hardware_report_honours_the_preference_it_is_given():
    """Bao cao phai noi ve thiet bi NGUOI GOI dung, khong phai ve mac dinh.

    Ban truoc luon goi select_device("auto"), nen tren may co CUDA no in
    `selected_device: cuda` ke ca khi lan chay that su dung CPU -- da xay ra
    that trong lan chay nghien cuu dau tien voi
    `configs/gpu_research.yaml` (dat `device_preference: cpu` co chu dich).
    Mot bao cao phan cung noi sai thiet bi bay nguoi doc log di kiem tra
    sai cho.
    """
    report = hardware_report("cpu")
    assert report["device_preference"] == "cpu"
    assert report["selected_device"] == "cpu"
