from __future__ import annotations

import itertools

import torch

from raemf_mc.bayesian.torch_backend import FitResult


def summarize_elbo_trace(elbo_trace: list[float]) -> dict:
    return {
        "final_elbo": elbo_trace[-1],
        "best_elbo": max(elbo_trace),
        "n_steps": len(elbo_trace),
    }


def seed_stability_metrics(seed_results: list[FitResult]) -> dict:
    final_elbos = torch.tensor([r.elbo_trace[-1] for r in seed_results])
    final_elbo_std = float(final_elbos.std(unbiased=False))

    max_diff = 0.0
    for a, b in itertools.combinations(seed_results, 2):
        diff = float(torch.max(torch.abs(a.mu - b.mu)))
        max_diff = max(max_diff, diff)

    return {"final_elbo_std": final_elbo_std, "max_pairwise_mu_diff": max_diff}
