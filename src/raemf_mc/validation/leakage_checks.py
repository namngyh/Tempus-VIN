from __future__ import annotations

import numpy as np
import pandas as pd

from raemf_mc.validation.purged_split import target_end_dates


def assert_no_target_leakage(
    dates: pd.DatetimeIndex, mask: np.ndarray, boundary, horizon: int
) -> None:
    """Raise AssertionError if any row selected by mask has a horizon-h
    target_end_date that is missing or >= boundary."""
    targets = target_end_dates(dates, horizon)
    boundary_ts = pd.Timestamp(boundary)
    selected = targets[np.asarray(mask, dtype=bool)]
    if selected.isna().any():
        raise AssertionError(
            "mask selects rows with no valid horizon target (leakage risk at series tail)"
        )
    leaky = selected[selected >= boundary_ts]
    if len(leaky) > 0:
        raise AssertionError(
            f"target leakage: {len(leaky)} rows have target_end_date >= boundary {boundary_ts}"
        )
