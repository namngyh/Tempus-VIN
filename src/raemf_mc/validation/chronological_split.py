from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/val/test by position,
    preserving chronological order — no shuffling, no purge-by-horizon
    (this is for nowcasting, not h-step-ahead forecasting). test receives
    the remainder after train_frac + val_frac."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test
