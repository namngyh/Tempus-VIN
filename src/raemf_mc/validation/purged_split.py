from __future__ import annotations

import numpy as np
import pandas as pd


def target_end_dates(dates: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """For each position i, the date at i+horizon (NaT if out of range) —
    the date on which the h-step-ahead target for row i is realized."""
    n = len(dates)
    result = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    valid_n = n - horizon
    if valid_n > 0:
        result.iloc[:valid_n] = dates[horizon:]
    return result


def purged_train_mask(dates: pd.DatetimeIndex, boundary, horizon: int) -> np.ndarray:
    """Boolean mask: True for rows whose target_end_date_h exists and is
    strictly before boundary."""
    targets = target_end_dates(dates, horizon)
    boundary_ts = pd.Timestamp(boundary)
    return (targets.notna() & (targets < boundary_ts)).to_numpy()
