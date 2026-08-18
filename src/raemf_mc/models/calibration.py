from __future__ import annotations

import torch


def fit_temperature(
    scores_val: torch.Tensor, y_val: torch.Tensor, n_steps: int = 100
) -> float:
    """Fit a single scalar temperature T > 0 minimizing NLL of
    softmax(scores/T) against y_val (integer class indices matching
    scores_val's column order). Parametrized via softplus to keep T
    positive throughout optimization, initialized near T=1.0
    (softplus(0.5413) ~= 1.0, since log(e-1) ~= 0.5413)."""
    raw_t = torch.nn.Parameter(torch.tensor(0.5413))
    optimizer = torch.optim.Adam([raw_t], lr=0.05)
    for _ in range(n_steps):
        optimizer.zero_grad()
        t = torch.nn.functional.softplus(raw_t)
        log_probs = torch.log_softmax(scores_val / t, dim=1)
        loss = torch.nn.functional.nll_loss(log_probs, y_val)
        loss.backward()
        optimizer.step()
    return float(torch.nn.functional.softplus(raw_t).item())


def apply_temperature(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return calibrated class probabilities: softmax(scores / temperature).

    Use this when you need actual probabilities (argmax for a predicted
    class, displaying calibrated probabilities). For NLL, use
    apply_temperature_log_prob instead of log() of this output."""
    return torch.softmax(scores / temperature, dim=1)


def apply_temperature_log_prob(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return calibrated log-probabilities: log_softmax(scores / temperature).
    Prefer this over log(apply_temperature(...)) for any NLL computation —
    log_softmax stays numerically finite even when temperature scaling
    drives probabilities to float32 zero, which log() of an already-computed
    probability cannot recover from."""
    return torch.log_softmax(scores / temperature, dim=1)
