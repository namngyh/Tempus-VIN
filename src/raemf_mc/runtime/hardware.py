import os
import torch


def select_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine")
        return torch.device("cuda")
    if preference != "auto":
        raise ValueError(f"unknown device preference: {preference!r}")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def hardware_report(preference: str = "auto") -> dict:
    """Bao cao phan cung. `preference` phai la CHINH gia tri ma nguoi goi se
    dung, khong phai mac dinh.

    Ban truoc luon goi `select_device("auto")` bat ke nguoi goi dung gi, nen
    tren may co CUDA no LUON in `selected_device: cuda` -- ke ca khi config
    dat `device_preference: cpu` va lan chay that su dung CPU. Mot bao cao
    phan cung noi sai thiet bi dang dung la thu bay nguoi doc log di kiem
    tra sai cho, va `configs/gpu_research.yaml` dat dung `cpu` co chu dich
    (xem ghi chu trong chinh file do).
    """
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "device_preference": preference,
        "selected_device": str(select_device(preference)),
    }
