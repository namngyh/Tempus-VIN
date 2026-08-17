from pathlib import Path
import pandas as pd
import pytest
from raemf_mc.data.ingest import (
    parse_vnindex_csv,
    drop_exact_duplicates,
    validate_ohlc_invariants,
    clean_vnindex_data,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "vnindex_sample.csv"
REAL_DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "VNINDEX_Daily.csv"


def test_parse_fixture_reconstructs_thousands_split_numbers():
    df = parse_vnindex_csv(FIXTURE)
    assert len(df) == 6
    row0 = df.iloc[0]
    assert row0["open"] == 100.0 and row0["volume"] == 4200
    row2 = df.iloc[2]
    assert row2["open"] == 1264.91 and row2["high"] == 1265.45
    assert row2["low"] == 1260.6 and row2["close"] == 1261.72
    row5 = df.iloc[5]
    assert row5["volume"] == 728451840


def test_drop_exact_duplicates_keeps_first_occurrence():
    df = parse_vnindex_csv(FIXTURE)
    deduped, dropped = drop_exact_duplicates(df)
    assert len(deduped) == 5
    assert len(dropped) == 1
    assert dropped[0].reason == "exact_duplicate_of_previous_row"
    assert dropped[0].date == "2024-12-17"


def test_validate_ohlc_invariants_drops_violating_row():
    df = parse_vnindex_csv(FIXTURE)
    deduped, _ = drop_exact_duplicates(df)
    valid, dropped = validate_ohlc_invariants(deduped)
    assert len(valid) == 4
    assert len(dropped) == 1
    assert dropped[0].reason == "ohlc_invariant_violation"
    assert dropped[0].date == "2004-12-10"


def test_clean_vnindex_data_on_fixture_end_to_end():
    # the 6-row fixture packs 2 dropped rows on purpose (33%) to exercise
    # both drop paths in one file, so it needs a raised threshold here;
    # the real-data test below checks the strict 5% default instead.
    clean, dropped, total = clean_vnindex_data(FIXTURE, max_drop_fraction=0.5)
    assert total == 6
    assert len(clean) == 4
    assert len(dropped) == 2
    assert list(clean.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert clean["date"].is_monotonic_increasing


def test_clean_vnindex_data_raises_when_drop_fraction_too_high():
    with pytest.raises(ValueError, match="dropped fraction"):
        clean_vnindex_data(FIXTURE, max_drop_fraction=0.1)


@pytest.mark.skipif(not REAL_DATA.exists(), reason="real dataset not present")
def test_clean_vnindex_data_on_real_file_matches_verified_counts():
    clean, dropped, total = clean_vnindex_data(REAL_DATA)
    assert total == 6307
    assert len(dropped) == 90
    assert sum(1 for d in dropped if d.reason == "exact_duplicate_of_previous_row") == 1
    assert sum(1 for d in dropped if d.reason == "ohlc_invariant_violation") == 89
    assert len(clean) == 6217
    assert clean["date"].is_monotonic_increasing
    assert clean["date"].is_unique
    assert clean["date"].iloc[0] == pd.Timestamp("2000-07-28")
    assert clean["date"].iloc[-1] == pd.Timestamp("2026-07-13")
