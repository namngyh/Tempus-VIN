from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(ohlcv: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Log return r_t = log(P_t / P_{t-1}). The first value has no
    predecessor and is dropped rather than filled."""
    prices = ohlcv[price_col]
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.dropna()
