from raemf_mc.data.loader import (
    load_vnindex_ohlcv,
    load_vnindex_ohlcv_with_report,
    write_ingestion_notes,
    DEFAULT_RAW_PATH,
)


def test_default_raw_path_points_at_real_csv():
    assert DEFAULT_RAW_PATH.name == "VNINDEX_Daily.csv"


def test_load_vnindex_ohlcv_returns_indexed_frame():
    df = load_vnindex_ohlcv()
    assert df.index.name == "date"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert len(df) == 6217


def test_load_vnindex_ohlcv_with_report_returns_dropped_rows():
    df, dropped, total = load_vnindex_ohlcv_with_report()
    assert total == 6307
    assert len(dropped) == 90
    assert len(df) == 6217


def test_write_ingestion_notes_produces_markdown(tmp_path):
    _, dropped, total = load_vnindex_ohlcv_with_report()
    out = tmp_path / "notes.md"
    write_ingestion_notes(dropped, total_parsed=total, final_count=6217, output_path=out)
    text = out.read_text(encoding="utf-8")
    assert "6307" in text
    assert "6217" in text
    assert "2024-12-17" in text
