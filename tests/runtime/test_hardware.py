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
        "torch_version", "cuda_available", "cpu_count", "selected_device",
    }
    assert report["torch_version"] == torch.__version__
