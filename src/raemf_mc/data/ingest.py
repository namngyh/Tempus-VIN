from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DroppedRow:
    line_number: int
    date: str
    reason: str
    raw_fields: tuple[str, ...]


def _merge_thousands_remainder(thousands: str, rest: str) -> str:
    """Join a thousands group with its remainder, restoring stripped zeros.

    The source export writes a thousands-separated price as two CSV fields
    ("1", "024.00" for 1,024.00) but strips leading zeros from the remainder
    group, so the second field arrives as "24.00". Naive concatenation then
    yields "124.00" — a self-consistent but ~10x-wrong OHLC quadruple that
    passes every downstream invariant check. Zero-padding the remainder's
    integer part back to 3 digits restores the intended value, and is a
    no-op when the remainder already has its full 3 digits ("264.91").
    """
    if "." in rest:
        int_part, frac_part = rest.split(".", 1)
        return thousands + int_part.rjust(3, "0") + "." + frac_part
    return thousands + rest.rjust(3, "0")


def _reconstruct_number_fields(parts: list[str]) -> tuple[float, float, float, float, int]:
    idx = 0
    values: list[float] = []
    for _ in range(4):
        current = parts[idx]
        if current == "1":
            merged = _merge_thousands_remainder(current, parts[idx + 1])
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


def check_implausible_daily_moves(
    df: pd.DataFrame,
    threshold: float = 0.10,
    max_allowed: int = 5,
) -> list[tuple[str, float]]:
    """Flag daily close-to-close log returns too large to be physically real.

    HOSE enforces a +/-7% daily price-limit band on VN-Index constituents, so
    an index-level daily log return beyond ~7% is already at the edge of what
    the market mechanically permits; `threshold=0.10` leaves a safety margin
    for index-composition effects and the rare genuine gap. This is the check
    that would have caught the thousands-remainder parsing bug (which produced
    187 such "moves", including an impossible 453% single day) at ingestion
    time instead of letting it flow silently into the model.

    Returns the list of (date, log_return) offenders and never fails on a
    small handful of genuinely real large moves; raises ValueError only when
    the count is high enough to indicate systematic corruption rather than
    real market events.
    """
    if len(df) < 2:
        return []
    close = df["close"].to_numpy()
    log_ret = np.log(close[1:] / close[:-1])
    dates = df["date"].to_numpy()[1:]
    offenders = [
        (pd.Timestamp(d).strftime("%Y-%m-%d"), float(r))
        for d, r in zip(dates, log_ret)
        if abs(r) > threshold
    ]
    if offenders:
        formatted = ", ".join(f"{d} ({r:+.2%})" for d, r in offenders[:10])
        message = (
            f"{len(offenders)} daily log return(s) exceed +/-{threshold:.0%} "
            f"(VN-Index price-limit band is +/-7%): {formatted}"
            f"{' ...' if len(offenders) > 10 else ''}"
        )
        if len(offenders) > max_allowed:
            raise ValueError(
                f"implausible daily moves indicate systematic data corruption: {message}"
            )
        logger.warning(message)
    return offenders


def clean_vnindex_data(
    path: str | Path, max_drop_fraction: float = 0.05
) -> tuple[pd.DataFrame, list[DroppedRow], int]:
    """Full ingestion pipeline: parse -> drop exact duplicates -> validate
    OHLC invariants -> sanity-check daily moves. Raises ValueError if the
    total dropped fraction exceeds max_drop_fraction, or if the cleaned
    series contains implausibly many extreme daily moves.

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
    check_implausible_daily_moves(clean)
    return clean, dropped, total
