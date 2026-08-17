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
    """Return calibrated class probabilities: softmax(scores / temperature)."""
    return torch.softmax(scores / temperature, dim=1)
