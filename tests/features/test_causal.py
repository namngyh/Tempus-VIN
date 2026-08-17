import numpy as np
import pandas as pd
from raemf_mc.features.causal import compute_causal_features, WARMUP_SESSIONS


def _synthetic_ohlcv(n=80):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_warmup_rows_dropped_and_no_nan_remain():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    assert len(features) == 80 - WARMUP_SESSIONS
    assert features.index[0] == df.index[WARMUP_SESSIONS]
    assert not features.isna().any().any()


def test_ret_1_matches_manual_log_return():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    expected = np.log(df["close"] / df["close"].shift(1))
    pd.testing.assert_series_equal(
        features["ret_1"], expected.iloc[WARMUP_SESSIONS:], check_names=False
    )


def test_ret_5_and_ret_20_use_correct_lookback():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    t = 70
    expected_ret_5 = np.log(df["close"].iloc[t] / df["close"].iloc[t - 5])
    expected_ret_20 = np.log(df["close"].iloc[t] / df["close"].iloc[t - 20])
    row = features.loc[df.index[t]]
    assert np.isclose(row["ret_5"], expected_ret_5)
    assert np.isclose(row["ret_20"], expected_ret_20)


def test_realized_vol_5_matches_manual_rolling_std():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    ret_1 = np.log(df["close"] / df["close"].shift(1))
    expected = ret_1.rolling(5).std()
    pd.testing.assert_series_equal(
        features["realized_vol_5"], expected.iloc[WARMUP_SESSIONS:], check_names=False
    )


def test_drawdown_from_high_60_is_nonpositive():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    assert (features["drawdown_from_high_60"] <= 1e-9).all()


def test_relative_volume_20_matches_manual_ratio():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    expected = df["volume"] / df["volume"].rolling(20).mean()
    pd.testing.assert_series_equal(
        features["relative_volume_20"], expected.iloc[WARMUP_SESSIONS:], check_names=False
    )


def test_output_has_exactly_nine_columns():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    assert list(features.columns) == [
        "ret_1", "ret_5", "ret_20",
        "realized_vol_5", "realized_vol_20",
        "volume_change_1", "relative_volume_20",
        "high_low_range", "drawdown_from_high_60",
    ]
