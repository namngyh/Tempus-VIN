import os
import torch


def configure_cpu_threads(num_threads: int | None = None) -> int:
    n = num_threads if num_threads is not None else (os.cpu_count() or 1)
    torch.set_num_threads(n)
    return n
