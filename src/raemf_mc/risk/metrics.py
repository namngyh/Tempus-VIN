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
    """Max drawdown per path (fraction in [0,1]) over the given window,
    measured from the entry price P_0 = 1 (log_return = 0 at t=0).
    daily_returns_2d: (n_paths, horizon).

    Computed entirely in log-space: (1 - price/running_max) equals
    (1 - exp(cum - running_max_cum)) exactly, and the exponent here is
    always <= 0, so this form cannot overflow or divide by zero the way
    exponentiating the raw cumulative log return first can.
    """
    n_paths = daily_returns_2d.shape[0]
    cum = np.cumsum(daily_returns_2d, axis=1)
    cum = np.concatenate([np.zeros((n_paths, 1), dtype=cum.dtype), cum], axis=1)
    running_max_cum = np.maximum.accumulate(cum, axis=1)
    drawdown = 1.0 - np.exp(cum - running_max_cum)
    return drawdown.max(axis=1)


def summarize_risk(
    daily_paths: np.ndarray,
    horizons: tuple[int, ...] = (1, 20),
    alphas: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """VaR/CVaR and max-drawdown summary table, one row per horizon.

    `daily_paths` must be (n_paths, horizon) daily log returns on the scale
    the caller wants reported. Note that `simulate_mc_paths` returns
    CENTERED returns — a caller reporting real-world risk must add the
    estimation window's own historical mean return back before passing them
    here, otherwise the market's baseline drift is silently dropped from the
    tail.

    Caveat: a table produced from a fast, few-step ADVI config (see
    configs/mc_smoke.yaml) is for verifying pipeline correctness, not for
    reporting as a real research-grade risk estimate. A meaningful fraction
    of such a fit's posterior draws can sit outside the model's own
    stationarity region (see the non-stationary-draw and volatility-band
    diagnostics `simulate_mc_paths` reports), which that function's bounds
    make SAFE to simulate from but do not make STATISTICALLY
    WELL-IDENTIFIED. Full-scale numbers require a properly converged ADVI
    fit (more steps, more seeds), not just a longer window.
    """
    if max(horizons) > daily_paths.shape[1]:
        raise ValueError(
            f"requested horizon {max(horizons)} exceeds the simulated path "
            f"length {daily_paths.shape[1]}"
        )
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
