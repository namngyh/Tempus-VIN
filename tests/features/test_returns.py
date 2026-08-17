import numpy as np
import pandas as pd
from raemf_mc.features.returns import compute_log_returns


def test_compute_log_returns_matches_manual_calculation():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.Series([100.0, 105.0, 103.95, 108.0, 110.0], index=dates, name="close")
    df = pd.DataFrame({"close": prices})
    result = compute_log_returns(df)
    expected = np.log(prices / prices.shift(1)).dropna()
    pd.testing.assert_series_equal(result, expected, check_names=False)
    assert len(result) == 4
    assert result.index[0] == dates[1]


def test_compute_log_returns_uses_given_price_col():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"close": [1.0, 1.0, 1.0], "open": [100.0, 110.0, 121.0]}, index=dates
    )
    result = compute_log_returns(df, price_col="open")
    assert np.isclose(result.iloc[0], np.log(110.0 / 100.0))
