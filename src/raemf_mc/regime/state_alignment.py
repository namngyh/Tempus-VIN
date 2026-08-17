from __future__ import annotations

import torch

STATE_NAMES = ("Bull", "Sideway", "Bear", "Stress")


def _zscore(x: torch.Tensor, rel_floor: float = 1e-6) -> torch.Tensor:
    """Standardize across the state dimension.

    When the four states' values are effectively identical (spread negligible
    relative to their own level) the component carries no information for
    discriminating states, and dividing by that near-zero spread would
    amplify pure float noise into a full-weight ranking signal. Return zeros
    in that case, which says exactly that: this component gets no vote.
    """
    centered = x - x.mean()
    std = float(x.std(unbiased=False))
    level = float(x.abs().mean())
    if std <= 0.0 or std <= rel_floor * level:
        return torch.zeros_like(x)
    return centered / std


def align_states(returns: torch.Tensor, log_filtered_prob: torch.Tensor) -> list[int]:
    """Compute a fixed permutation mapping raw latent states to economic
    labels (Bull/Sideway/Bear/Stress), fit on train data only.

    Ranks raw states by a composite score combining mean return (favors Bull
    at the high end) and volatility (favors Stress at the low end). Both
    components are z-scored ACROSS THE FOUR STATES before combining: on daily
    VN-Index data mean returns are order 1e-3 while volatilities are order
    1e-2, so the raw `mean_return - volatility` difference was effectively
    ranking by volatility alone and the mean-return term contributed almost
    nothing. Standardizing puts them on a comparable footing so neither
    dominates purely because of its natural scale.

    CAVEAT: standardizing removes the unit mismatch, but two z-scored
    components summed with equal weight still do not guarantee a sensible
    ordering when mean return and volatility disagree sharply and one state
    is a volatility outlier — a 2-of-2 vote has no tie-break. Resolving that
    properly needs the third criterion below.

    NOTE: the design spec also names regime DURATION (expected persistence)
    as an alignment criterion. It is not implemented here — it would require
    threading MSEGARCHParams / the transition matrix into a function that
    currently takes only `returns` and `log_filtered_prob`, which is a real
    interface change. Deliberately deferred, not overlooked.
    """
    probs = torch.exp(log_filtered_prob)
    n_states = probs.shape[1]
    total_mass = probs.sum(dim=0, keepdim=True)
    # A state with ~zero total mass would make this division produce NaN and
    # silently corrupt the whole ordering; floor it instead.
    weights = probs / torch.clamp(total_mass, min=1e-12)  # (T, n_states)

    mean_return = (weights * returns.unsqueeze(1)).sum(dim=0)
    mean_sq = (weights * (returns.unsqueeze(1) ** 2)).sum(dim=0)
    variance = torch.clamp(mean_sq - mean_return**2, min=0.0)
    volatility = torch.sqrt(variance)

    score = _zscore(mean_return) - _zscore(volatility)
    order = torch.argsort(score, descending=True)  # best (Bull) first
    return [int(order[i]) for i in range(n_states)]


def apply_alignment(log_filtered_prob: torch.Tensor, permutation: list[int]) -> torch.Tensor:
    """Reorder columns of log_filtered_prob so column i corresponds to
    STATE_NAMES[i], using a permutation produced by align_states."""
    idx = torch.tensor(permutation, dtype=torch.long, device=log_filtered_prob.device)
    return log_filtered_prob[:, idx]
