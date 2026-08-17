from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HierarchicalPriorConfig:
    hyper_mean_scale: float = 1.0
    min_effective_observations: float = 30.0


def state_shrinkage_weight(
    effective_obs: torch.Tensor, min_effective_observations: float
) -> torch.Tensor:
    """Per-state weight in (0, 1]: states with effective_obs far below
    min_effective_observations get a smaller weight (tighter shrinkage
    toward the hyper-mean); states with ample data approach weight 1."""
    return torch.clamp(effective_obs / min_effective_observations, min=0.05, max=1.0)


def hierarchical_normal_log_prob(
    state_params: torch.Tensor,
    hyper_mean: torch.Tensor,
    base_scale: torch.Tensor,
    shrinkage_weight: torch.Tensor,
) -> torch.Tensor:
    """log p(state_params | hyper_mean) under
    Normal(hyper_mean, base_scale / shrinkage_weight). A smaller
    shrinkage_weight tightens the effective prior std, pulling the state
    parameter harder toward the shared hyper-mean."""
    scale = base_scale / shrinkage_weight
    dist = torch.distributions.Normal(hyper_mean, scale)
    return dist.log_prob(state_params).sum()
