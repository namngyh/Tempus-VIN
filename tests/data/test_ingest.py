from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from raemf_mc.data.ingest import (
    parse_vnindex_csv,
    drop_exact_duplicates,
    validate_ohlc_invariants,
    clean_vnindex_data,
    check_implausible_daily_moves,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "vnindex_sample.csv"
REAL_DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "VNINDEX_Daily.csv"


def test_parse_fixture_reconstructs_thousands_split_numbers():
    df = parse_vnindex_csv(FIXTURE)
    assert len(df) == 7
    row0 = df.iloc[0]
    assert row0["open"] == 100.0 and row0["volume"] == 4200
    row2 = df.iloc[2]
    assert row2["open"] == 1264.91 and row2["high"] == 1265.45
    assert row2["low"] == 1260.6 and row2["close"] == 1261.72
    row5 = df.iloc[5]
    assert row5["volume"] == 728451840


def test_parse_restores_leading_zeros_stripped_from_thousands_remainder():
    # Mirrors real raw line 1458: "1,24,1,24,1,22.97,1,23.05,8,751,729".
    # The source export strips leading zeros from the remainder group, so
    # 1,024.00 arrives as the pair ("1", "24"). Naive concatenation yields
    # 124.0 — internally consistent (it passes every OHLC invariant) but
    # ~10x wrong, which is exactly why this needs a fixture-level guard.
    df = parse_vnindex_csv(FIXTURE)
    row = df[df["date"] == pd.Timestamp("2007-01-19")].iloc[0]
    assert row["open"] == 1024.0
    assert row["high"] == 1024.0
    assert row["low"] == 1022.97
    assert row["close"] == 1023.05
    assert row["volume"] == 8751729


def test_drop_exact_duplicates_keeps_first_occurrence():
    df = parse_vnindex_csv(FIXTURE)
    deduped, dropped = drop_exact_duplicates(df)
    assert len(deduped) == 6
    assert len(dropped) == 1
    assert dropped[0].reason == "exact_duplicate_of_previous_row"
    assert dropped[0].date == "2024-12-17"


def test_validate_ohlc_invariants_drops_violating_row():
    df = parse_vnindex_csv(FIXTURE)
    deduped, _ = drop_exact_duplicates(df)
    valid, dropped = validate_ohlc_invariants(deduped)
    assert len(valid) == 5
    assert len(dropped) == 1
    assert dropped[0].reason == "ohlc_invariant_violation"
    assert dropped[0].date == "2004-12-10"


def test_clean_vnindex_data_on_fixture_end_to_end():
    # the 7-row fixture packs 2 dropped rows on purpose (~29%) to exercise
    # both drop paths in one file, so it needs a raised threshold here;
    # the real-data test below checks the strict 5% default instead.
    clean, dropped, total = clean_vnindex_data(FIXTURE, max_drop_fraction=0.5)
    assert total == 7
    assert len(clean) == 5
    assert len(dropped) == 2
    assert list(clean.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert clean["date"].is_monotonic_increasing


def test_clean_vnindex_data_raises_when_drop_fraction_too_high():
    with pytest.raises(ValueError, match="dropped fraction"):
        clean_vnindex_data(FIXTURE, max_drop_fraction=0.1)


def _frame(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(closes), freq="D"),
            "close": closes,
        }
    )


def test_check_implausible_daily_moves_passes_on_realistic_series():
    assert check_implausible_daily_moves(_frame([100.0, 103.0, 99.0, 101.0])) == []


def test_check_implausible_daily_moves_reports_but_tolerates_a_few():
    offenders = check_implausible_daily_moves(_frame([100.0, 130.0, 128.0]))
    assert len(offenders) == 1
    assert offenders[0][0] == "2020-01-02"
    assert offenders[0][1] > 0.10


def test_check_implausible_daily_moves_raises_on_systematic_corruption():
    # the parser bug produced exactly this signature: many rows off by ~10x,
    # each internally consistent, showing up only as impossible daily jumps.
    closes = [100.0, 1000.0] * 6
    with pytest.raises(ValueError, match="systematic data corruption"):
        check_implausible_daily_moves(_frame(closes))


@pytest.mark.skipif(not REAL_DATA.exists(), reason="real dataset not present")
def test_clean_vnindex_data_on_real_file_matches_verified_counts():
    clean, dropped, total = clean_vnindex_data(REAL_DATA)
    assert total == 6307
    assert len(dropped) == 43
    assert sum(1 for d in dropped if d.reason == "exact_duplicate_of_previous_row") == 1
    # 42 genuinely unrecoverable source rows. This was 89 before the
    # thousands-remainder leading-zero fix; 47 of those were parser damage,
    # not source damage.
    assert sum(1 for d in dropped if d.reason == "ohlc_invariant_violation") == 42
    assert len(clean) == 6264
    assert clean["date"].is_monotonic_increasing
    assert clean["date"].is_unique
    assert clean["date"].iloc[0] == pd.Timestamp("2000-07-28")
    assert clean["date"].iloc[-1] == pd.Timestamp("2026-07-13")


@pytest.mark.skipif(not REAL_DATA.exists(), reason="real dataset not present")
def test_real_series_has_no_physically_implausible_daily_moves():
    clean, _, _ = clean_vnindex_data(REAL_DATA)
    close = clean["close"].to_numpy()
    log_ret = np.log(close[1:] / close[:-1])
    # VN-Index runs a +/-7% daily price-limit band. Exactly one session in the
    # corrected series exceeds 8% (2025-04-03 -> 2025-04-08, a real global
    # market event); the buggy parser produced 187 such days, topping out at
    # an impossible +453%.
    assert int((np.abs(log_ret) > 0.08).sum()) == 1
    assert float(np.abs(log_ret).max()) < 0.09
