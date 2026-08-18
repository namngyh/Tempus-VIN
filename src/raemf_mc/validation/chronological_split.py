from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/val/test by position,
    preserving chronological order — no shuffling, no purge-by-horizon
    (this is for nowcasting, not h-step-ahead forecasting). test receives
    the remainder after train_frac + val_frac."""
    if train_frac + val_frac > 1.0:
        raise ValueError(
            f"train_frac + val_frac must be <= 1.0, got {train_frac} + {val_frac}"
        )
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"chronological_split produced an empty partition "
            f"(train={len(train)}, val={len(val)}, test={len(test)}) "
            f"from n={n} rows with train_frac={train_frac}, val_frac={val_frac}"
        )
    return train, val, test
