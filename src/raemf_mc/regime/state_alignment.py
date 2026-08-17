from __future__ import annotations

import torch

STATE_NAMES = ("Bull", "Sideway", "Bear", "Stress")


def align_states(returns: torch.Tensor, log_filtered_prob: torch.Tensor) -> list[int]:
    """Compute a fixed permutation mapping raw latent states to economic
    labels (Bull/Sideway/Bear/Stress), fit on train data only. Ranks raw
    states by a composite score: mean return (favors Bull), penalized by
    volatility (favors Stress at the low end) — implemented as a single
    sort key `mean_return - volatility`, which is high for Bull (high
    mean, low vol) and lowest for Stress (very negative mean, high vol).
    """
    probs = torch.exp(log_filtered_prob)
    n_states = probs.shape[1]
    weights = probs / probs.sum(dim=0, keepdim=True)  # (T, n_states)

    mean_return = (weights * returns.unsqueeze(1)).sum(dim=0)
    mean_sq = (weights * (returns.unsqueeze(1) ** 2)).sum(dim=0)
    variance = torch.clamp(mean_sq - mean_return**2, min=0.0)
    volatility = torch.sqrt(variance)

    score = mean_return - volatility
    order = torch.argsort(score, descending=True)  # best (Bull) first
    return [int(order[i]) for i in range(n_states)]


def apply_alignment(log_filtered_prob: torch.Tensor, permutation: list[int]) -> torch.Tensor:
    """Reorder columns of log_filtered_prob so column i corresponds to
    STATE_NAMES[i], using a permutation produced by align_states."""
    idx = torch.tensor(permutation, dtype=torch.long)
    return log_filtered_prob[:, idx]
