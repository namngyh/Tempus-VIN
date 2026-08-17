import torch
from raemf_mc.runtime.cpu import configure_cpu_threads


def test_configure_cpu_threads_sets_torch_threads():
    n = configure_cpu_threads(2)
    assert n == 2
    assert torch.get_num_threads() == 2


def test_configure_cpu_threads_defaults_to_cpu_count():
    import os
    n = configure_cpu_threads(None)
    assert n == (os.cpu_count() or 1)
