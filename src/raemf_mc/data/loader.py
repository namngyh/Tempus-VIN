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
        "# Ghi chú làm sạch dữ liệu VN-Index",
        "",
        f"Tổng số dòng parse được từ file gốc: {total_parsed}",
        f"Số dòng bị loại: {len(dropped)}",
        f"Số dòng sạch còn lại: {final_count}",
        "",
        "## Danh sách dòng bị loại",
        "",
        "| Dòng (line) | Ngày | Lý do | Giá trị raw (O,H,L,C,V) |",
        "|---|---|---|---|",
    ]
    for d in dropped:
        raw = ", ".join(d.raw_fields)
        lines.append(f"| {d.line_number} | {d.date} | {d.reason} | {raw} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
