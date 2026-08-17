from __future__ import annotations

import numpy as np
import pandas as pd

WARMUP_SESSIONS = 60


def compute_causal_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Causal (no-lookahead) features from OHLCV, one row per session.

    Every feature is a pandas rolling/shift computation ending at the
    current row, so it never uses future data. The first WARMUP_SESSIONS
    rows lack enough history for the longest window (drawdown_from_high_60)
    and are dropped explicitly rather than imputed.
    """
    close = ohlcv["close"]
    volume = ohlcv["volume"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    ret_1 = np.log(close / close.shift(1))
    ret_5 = np.log(close / close.shift(5))
    ret_20 = np.log(close / close.shift(20))
    realized_vol_5 = ret_1.rolling(5).std()
    realized_vol_20 = ret_1.rolling(20).std()
    volume_change_1 = np.log(volume / volume.shift(1))
    relative_volume_20 = volume / volume.rolling(20).mean()
    high_low_range = (high - low) / close
    rolling_max_60 = close.rolling(60).max()
    drawdown_from_high_60 = (close - rolling_max_60) / rolling_max_60

    features = pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "realized_vol_5": realized_vol_5,
            "realized_vol_20": realized_vol_20,
            "volume_change_1": volume_change_1,
            "relative_volume_20": relative_volume_20,
            "high_low_range": high_low_range,
            "drawdown_from_high_60": drawdown_from_high_60,
        },
        index=ohlcv.index,
    )
    return features.iloc[WARMUP_SESSIONS:].copy()
