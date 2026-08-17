from __future__ import annotations

from pathlib import Path

import pandas as pd

from raemf_mc.data.ingest import DroppedRow, clean_vnindex_data

DEFAULT_RAW_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "VNINDEX_Daily.csv"
)


def load_vnindex_ohlcv_with_report(
    path: str | Path | None = None, max_drop_fraction: float = 0.05
) -> tuple[pd.DataFrame, list[DroppedRow], int]:
    resolved = Path(path) if path is not None else DEFAULT_RAW_PATH
    clean, dropped, total = clean_vnindex_data(resolved, max_drop_fraction=max_drop_fraction)
    indexed = clean.set_index("date")
    indexed.index.name = "date"
    return indexed, dropped, total


def load_vnindex_ohlcv(
    path: str | Path | None = None, max_drop_fraction: float = 0.05
) -> pd.DataFrame:
    """Public API: load cleaned VN-Index OHLCV data indexed by date."""
    df, _dropped, _total = load_vnindex_ohlcv_with_report(path, max_drop_fraction)
    return df


def write_ingestion_notes(
    dropped: list[DroppedRow],
    total_parsed: int,
    final_count: int,
    output_path: str | Path,
) -> None:
    """Write a human-readable Markdown log of every dropped row and why."""
    output_path = Path(output_path)
    lines = [
        "# Ghi chu lam sach du lieu VN-Index",
        "",
        f"Tong so dong parse duoc tu file goc: {total_parsed}",
        f"So dong bi loai: {len(dropped)}",
        f"So dong sach con lai: {final_count}",
        "",
        "## Danh sach dong bi loai",
        "",
        "| Dong (line) | Ngay | Ly do | Gia tri raw (O,H,L,C,V) |",
        "|---|---|---|---|",
    ]
    for d in dropped:
        raw = ", ".join(d.raw_fields)
        lines.append(f"| {d.line_number} | {d.date} | {d.reason} | {raw} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
