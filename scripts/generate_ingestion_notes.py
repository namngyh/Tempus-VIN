from pathlib import Path

from raemf_mc.data.loader import load_vnindex_ohlcv_with_report, write_ingestion_notes

if __name__ == "__main__":
    df, dropped, total = load_vnindex_ohlcv_with_report()
    output = Path(__file__).resolve().parents[1] / "docs" / "data_ingestion_notes.md"
    write_ingestion_notes(dropped, total_parsed=total, final_count=len(df), output_path=output)
    print(f"wrote {output} ({len(dropped)} dropped rows logged)")
