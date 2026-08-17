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


def hardware_report() -> dict:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "selected_device": str(select_device("auto")),
    }
