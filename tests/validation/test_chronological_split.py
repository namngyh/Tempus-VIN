import pandas as pd
from raemf_mc.validation.chronological_split import chronological_split


def test_split_sizes_and_order_for_100_rows():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"x": range(100)}, index=dates)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_split_preserves_chronological_order_with_no_overlap():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"x": range(100)}, index=dates)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)
    assert train.index[-1] < val.index[0]
    assert val.index[-1] < test.index[0]
    all_idx = train.index.append(val.index).append(test.index)
    assert all_idx.is_monotonic_increasing
    assert len(all_idx) == len(df)


def test_split_uses_default_fractions():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({"x": range(200)}, index=dates)
    train, val, test = chronological_split(df)
    assert len(train) == 140
    assert len(val) == 30
    assert len(test) == 30
