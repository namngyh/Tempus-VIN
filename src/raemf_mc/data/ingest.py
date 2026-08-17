from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DroppedRow:
    line_number: int
    date: str
    reason: str
    raw_fields: tuple[str, ...]


def _reconstruct_number_fields(parts: list[str]) -> tuple[float, float, float, float, int]:
    idx = 0
    values: list[float] = []
    for _ in range(4):
        current = parts[idx]
        if current == "1":
            merged = current + parts[idx + 1]
            idx += 2
        else:
            merged = current
            idx += 1
        values.append(float(merged))
    volume_parts = parts[idx:]
    if not volume_parts:
        raise ValueError("missing volume field")
    volume = int("".join(volume_parts))
    open_, high, low, close = values
    return open_, high, low, close, volume


def parse_vnindex_csv(path: str | Path) -> pd.DataFrame:
    """Parse the raw VNINDEX_Daily.csv export into an unvalidated DataFrame.

    The source export splits thousands-separated numbers across extra CSV
    columns (e.g. "1,840.69" becomes two fields "1" and "840.69"). This
    reconstructs the intended Open/High/Low/Close/Volume values. Raises
    ValueError on any row that cannot be parsed — never silently skips a
    row at this stage.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = list(reader)

    records = []
    for line_number, raw in enumerate(rows, start=2):
        date_raw = raw[0].strip()
        try:
            date = datetime.strptime(date_raw, "%d/%m/%Y %H:%M")
        except ValueError as exc:
            raise ValueError(f"line {line_number}: cannot parse date {date_raw!r}") from exc
        parts = [p for p in raw[1:] if p != ""]
        try:
            open_, high, low, close, volume = _reconstruct_number_fields(parts)
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"line {line_number}: cannot reconstruct numeric fields from {raw!r}"
            ) from exc
        records.append(
            {
                "line_number": line_number,
                "date": date,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return pd.DataFrame.from_records(records)


def _row_to_dropped(row: pd.Series, reason: str) -> DroppedRow:
    return DroppedRow(
        line_number=int(row["line_number"]),
        date=row["date"].strftime("%Y-%m-%d"),
        reason=reason,
        raw_fields=tuple(
            str(row[c]) for c in ["open", "high", "low", "close", "volume"]
        ),
    )


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[DroppedRow]]:
    """Drop rows that are exact duplicates of the immediately preceding row
    (same date and same OHLCV values). Keeps the first occurrence."""
    keep_mask = []
    dropped: list[DroppedRow] = []
    prev = None
    for _, row in df.iterrows():
        is_dupe = (
            prev is not None
            and row["date"] == prev["date"]
            and row["open"] == prev["open"]
            and row["high"] == prev["high"]
            and row["low"] == prev["low"]
            and row["close"] == prev["close"]
            and row["volume"] == prev["volume"]
        )
        if is_dupe:
            keep_mask.append(False)
            dropped.append(_row_to_dropped(row, "exact_duplicate_of_previous_row"))
        else:
            keep_mask.append(True)
            prev = row
    return df[keep_mask].reset_index(drop=True), dropped


def validate_ohlc_invariants(df: pd.DataFrame) -> tuple[pd.DataFrame, list[DroppedRow]]:
    """Drop rows violating low <= open <= high and low <= close <= high.
    Never guesses a corrected value — only drops and records why."""
    mask = (
        (df["low"] <= df["open"])
        & (df["open"] <= df["high"])
        & (df["low"] <= df["close"])
        & (df["close"] <= df["high"])
        & (df["low"] <= df["high"])
    )
    dropped = [
        _row_to_dropped(row, "ohlc_invariant_violation")
        for _, row in df[~mask].iterrows()
    ]
    return df[mask].reset_index(drop=True), dropped


def clean_vnindex_data(
    path: str | Path, max_drop_fraction: float = 0.05
) -> tuple[pd.DataFrame, list[DroppedRow], int]:
    """Full ingestion pipeline: parse -> drop exact duplicates -> validate
    OHLC invariants. Raises ValueError if the total dropped fraction
    exceeds max_drop_fraction.

    Returns (clean_df, dropped_rows, total_parsed_rows). clean_df has
    columns [date, open, high, low, close, volume], sorted by date
    ascending.
    """
    raw = parse_vnindex_csv(path)
    total = len(raw)
    deduped, dupes = drop_exact_duplicates(raw)
    valid, invalid = validate_ohlc_invariants(deduped)
    dropped = dupes + invalid
    fraction = len(dropped) / total if total else 0.0
    if fraction > max_drop_fraction:
        raise ValueError(
            f"dropped fraction {fraction:.4f} exceeds max_drop_fraction="
            f"{max_drop_fraction}; {len(dropped)}/{total} rows dropped"
        )
    valid = valid.sort_values("date").reset_index(drop=True)
    clean = valid[["date", "open", "high", "low", "close", "volume"]]
    return clean, dropped, total
