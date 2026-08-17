import numpy as np
import pandas as pd
import pytest
from raemf_mc.validation.purged_split import purged_train_mask
from raemf_mc.validation.leakage_checks import assert_no_target_leakage


def test_assert_no_target_leakage_passes_for_valid_mask():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    boundary = dates[20]
    mask = purged_train_mask(dates, boundary, horizon=5)
    assert_no_target_leakage(dates, mask, boundary, horizon=5)  # must not raise


def test_assert_no_target_leakage_raises_for_leaky_mask():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    boundary = dates[20]
    leaky_mask = np.ones(len(dates), dtype=bool)  # includes everything, incl. leaky rows
    with pytest.raises(AssertionError):
        assert_no_target_leakage(dates, leaky_mask, boundary, horizon=5)


def test_assert_no_target_leakage_raises_for_missing_target():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    boundary = dates[9] + pd.Timedelta(days=100)
    mask = np.ones(len(dates), dtype=bool)  # last rows have no horizon-5 target
    with pytest.raises(AssertionError):
        assert_no_target_leakage(dates, mask, boundary, horizon=5)
