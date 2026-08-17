import pandas as pd
from raemf_mc.validation.purged_split import target_end_dates, purged_train_mask


def _dates(n=30):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_target_end_dates_shifts_by_horizon():
    dates = _dates(10)
    targets = target_end_dates(dates, horizon=3)
    assert targets.iloc[0] == dates[3]
    assert pd.isna(targets.iloc[-1])
    assert pd.isna(targets.iloc[-3])
    assert not pd.isna(targets.iloc[-4])


def test_purged_train_mask_excludes_rows_without_valid_target():
    dates = _dates(10)
    boundary = dates[9] + pd.Timedelta(days=100)  # boundary far in the future
    mask = purged_train_mask(dates, boundary, horizon=3)
    # last 3 rows have no horizon-3 target at all -> must be excluded
    assert not mask[-3:].any()
    assert mask[: 10 - 3].all()


def test_purged_train_mask_excludes_rows_whose_target_crosses_boundary():
    dates = _dates(20)
    boundary = dates[10]
    mask = purged_train_mask(dates, boundary, horizon=5)
    targets = target_end_dates(dates, horizon=5)
    included_targets = targets[mask]
    assert (included_targets < boundary).all()
    assert included_targets.notna().all()
    # sanity: at least one row is legitimately included
    assert mask.sum() > 0
