from __future__ import annotations

import numpy as np
import pandas as pd


def compute_var(horizon_returns: np.ndarray, alpha: float) -> float:
    """VaR at confidence alpha, as a positive loss magnitude:
    VaR_alpha = -quantile(horizon_returns, 1 - alpha)."""
    return float(-np.quantile(horizon_returns, 1.0 - alpha))


def compute_cvar(horizon_returns: np.ndarray, alpha: float) -> float:
    """CVaR (Expected Shortfall) at confidence alpha: mean of the tail
    at/below the VaR quantile, as a positive loss magnitude."""
    threshold = np.quantile(horizon_returns, 1.0 - alpha)
    tail = horizon_returns[horizon_returns <= threshold]
    return float(-tail.mean())


def compute_max_drawdown(daily_returns_2d: np.ndarray) -> np.ndarray:
    """Max drawdown per path (fraction in [0,1]) over the given window.
    daily_returns_2d: (n_paths, horizon)."""
    cum_log_return = np.cumsum(daily_returns_2d, axis=1)
    price = np.exp(cum_log_return)
    running_max = np.maximum.accumulate(price, axis=1)
    drawdown = (running_max - price) / running_max
    return drawdown.max(axis=1)


def summarize_risk(
    daily_paths: np.ndarray,
    horizons: tuple[int, ...] = (1, 20),
    alphas: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        horizon_returns = daily_paths[:, :h].sum(axis=1)
        drawdowns = compute_max_drawdown(daily_paths[:, :h])
        row: dict[str, float] = {"horizon": h}
        for alpha in alphas:
            pct = int(round(alpha * 100))
            row[f"VaR_{pct}"] = compute_var(horizon_returns, alpha)
            row[f"CVaR_{pct}"] = compute_cvar(horizon_returns, alpha)
        row["mean_max_drawdown"] = float(drawdowns.mean())
        row["median_max_drawdown"] = float(np.median(drawdowns))
        row["p95_max_drawdown"] = float(np.quantile(drawdowns, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")
