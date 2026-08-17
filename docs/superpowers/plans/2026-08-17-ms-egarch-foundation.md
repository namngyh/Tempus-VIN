# MS-EGARCH Foundation (Sub-project 1+2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working repo with clean VN-Index data ingestion, a generic Variational-Bayes (ADVI) optimization engine, and a 4-state MS-EGARCH model (Gray's collapsing recursion + Hamilton forward filter) whose full parameter posterior is estimated via ADVI directly on real data.

**Architecture:** `src/raemf_mc/` package with independent modules (`runtime`, `data`, `features`, `validation`, `bayesian`, `regime`) wired together only at the top of each task; the ADVI engine in `bayesian/torch_backend.py` is domain-agnostic (takes a `log_joint(theta) -> Tensor` callable) and `regime/ms_egarch.py` supplies the MS-EGARCH-specific `log_joint`.

**Tech Stack:** Python 3.11, PyTorch 2.13 (CPU-only on this machine), pandas, numpy, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-17-ms-egarch-foundation-design.md`

## Global Constraints

- No leakage: forward filter is strictly causal; every purge check uses `target_end_date_h < boundary`.
- No silent fallback: every ADVI retry/fallback event is appended to a `fallbacks.json`-style log, never swallowed silently.
- No float16 anywhere in ELBO / log-likelihood / logsumexp / quantile computation — enforce with an explicit guard.
- Multi-seed ADVI pools posteriors as an equal-weight mixture across seeds — never keep only the best-ELBO seed.
- No fabricated data: any row dropped during CSV ingestion is logged with its real reason; the drop-fraction is asserted below 5% and raises otherwise.
- MS-EGARCH parameter budget: 28 core params (4-state transition matrix via unconstrained softmax = 12, plus 4 EGARCH params × 4 states = 16) + 1 global `nu` (Student-t degrees of freedom, shared across states, not counted in the 28).
- Gray's collapsing is done in level-space via `torch.logsumexp(log_filtered_prob + log_var)`, never by averaging log-variances directly (see spec §7.2).
- `z[t-1]` for the leverage term is standardized by the collapsed `sigma_bar[t-1]`, not by any single state's sigma (see spec §7.3).
- All exact row counts referenced below (6307 total, 1 duplicate, 89 invariant violations, 6217 final) were verified against the real committed `data/raw/VNINDEX_Daily.csv` — do not treat them as estimates.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/raemf_mc/__init__.py`
- Create: `README.md`
- Create: `AGENTS.md`
- Create: `tests/__init__.py`
- Test: `tests/test_package_smoke.py`

**Interfaces:**
- Produces: importable package `raemf_mc` with `__version__ = "0.1.0"`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "raemf_mc"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "pyyaml",
    "scikit-learn",
    "statsmodels",
    "pyarrow",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Write `src/raemf_mc/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write `tests/__init__.py`** (empty file)

- [ ] **Step 4: Write `README.md`**

```markdown
# RAEMF-VB-MC

Mo hinh du bao xac suat che do thi truong (Bull/Sideway/Bear/Stress) va phan phoi
loi suat VN-Index, dung MS-EGARCH (Markov-Switching EGARCH 4 trang thai) uoc luong
toan phan bang Variational Bayes (ADVI) lam loi tang regime + risk.

Xem `docs/superpowers/specs/` cho thiet ke chi tiet va `docs/ms_egarch_design_decisions.md`
cho cac quyet dinh thiet ke con mo.

## Cai dat

Repo dung truc tiep Python 3.11 he thong hien co (torch, numpy, pandas, scipy,
pytest, scikit-learn, statsmodels, pyarrow da co san). Cai package o che do
editable:

```
pip install -e ".[dev]"
```

## Chay test

```
python -m pytest -q
python -m ruff check src tests
```
```

- [ ] **Step 5: Write `AGENTS.md`**

```markdown
# Ky luat nghien cuu

- Khong bia so lieu — thanh phan khong uoc luong duoc phai ghi ro ly do.
- Khong dung test set de tuning, chon feature, chon calibration hay chon nguong.
- Voi moi horizon, train/validation phai duoc purge bang `target_end_date_h < boundary`.
- Khong silent fallback: moi lan ADVI fail/retry/rot xuong mean-field phai duoc ghi log.
- Khong float16 o bat ky dau trong ELBO, log-likelihood, log-sum-exp, quantile duoi.
- Tai lieu thiet ke va bao cao bang tieng Viet, trung lap, khong dua loi khuyen dau tu.
```

- [ ] **Step 6: Write failing smoke test**

```python
# tests/test_package_smoke.py
import raemf_mc


def test_package_importable_and_versioned():
    assert raemf_mc.__version__ == "0.1.0"
```

- [ ] **Step 7: Install package in editable mode and run test**

Run: `pip install -e ".[dev]"` then `python -m pytest tests/test_package_smoke.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/raemf_mc/__init__.py README.md AGENTS.md tests/__init__.py tests/test_package_smoke.py
git commit -m "chore: scaffold raemf_mc package"
```

---

## Task 2: Runtime hardware detection

**Files:**
- Create: `src/raemf_mc/runtime/__init__.py`
- Create: `src/raemf_mc/runtime/hardware.py`
- Create: `src/raemf_mc/runtime/cpu.py`
- Test: `tests/runtime/test_hardware.py`
- Test: `tests/runtime/test_cpu.py`
- Create: `tests/runtime/__init__.py`

**Interfaces:**
- Produces: `select_device(preference: str = "auto") -> torch.device`, `hardware_report() -> dict`, `configure_cpu_threads(num_threads: int | None = None) -> int`.

- [ ] **Step 1: Write failing tests**

```python
# tests/runtime/test_hardware.py
import torch
import pytest
from raemf_mc.runtime.hardware import select_device, hardware_report


def test_select_device_cpu_explicit():
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_auto_falls_back_to_cpu_without_cuda():
    device = select_device("auto")
    if not torch.cuda.is_available():
        assert device == torch.device("cpu")


def test_select_device_cuda_raises_when_unavailable():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            select_device("cuda")


def test_hardware_report_has_expected_keys():
    report = hardware_report()
    assert set(report.keys()) == {
        "torch_version", "cuda_available", "cpu_count", "selected_device",
    }
    assert report["torch_version"] == torch.__version__
```

```python
# tests/runtime/test_cpu.py
import torch
from raemf_mc.runtime.cpu import configure_cpu_threads


def test_configure_cpu_threads_sets_torch_threads():
    n = configure_cpu_threads(2)
    assert n == 2
    assert torch.get_num_threads() == 2


def test_configure_cpu_threads_defaults_to_cpu_count():
    import os
    n = configure_cpu_threads(None)
    assert n == (os.cpu_count() or 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/runtime -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.runtime'`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/runtime/__init__.py
```

```python
# src/raemf_mc/runtime/hardware.py
import os
import torch


def select_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine")
        return torch.device("cuda")
    if preference != "auto":
        raise ValueError(f"unknown device preference: {preference!r}")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def hardware_report() -> dict:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "selected_device": str(select_device("auto")),
    }
```

```python
# src/raemf_mc/runtime/cpu.py
import os
import torch


def configure_cpu_threads(num_threads: int | None = None) -> int:
    n = num_threads if num_threads is not None else (os.cpu_count() or 1)
    torch.set_num_threads(n)
    return n
```

```python
# tests/runtime/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/runtime -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/runtime tests/runtime
git commit -m "feat: add runtime hardware detection and CPU thread config"
```

---

## Task 3: CSV ingestion — parse, dedup, validate

**Files:**
- Create: `src/raemf_mc/data/__init__.py`
- Create: `src/raemf_mc/data/ingest.py`
- Test: `tests/data/test_ingest.py`
- Create: `tests/data/__init__.py`
- Create: `tests/fixtures/vnindex_sample.csv` (synthetic fixture covering known patterns)

**Interfaces:**
- Produces: `DroppedRow` dataclass (`line_number: int, date: str, reason: str, raw_fields: tuple[str, ...]`); `parse_vnindex_csv(path) -> pd.DataFrame` (columns `line_number, date, open, high, low, close, volume`); `drop_exact_duplicates(df) -> tuple[pd.DataFrame, list[DroppedRow]]`; `validate_ohlc_invariants(df) -> tuple[pd.DataFrame, list[DroppedRow]]`; `clean_vnindex_data(path, max_drop_fraction=0.05) -> tuple[pd.DataFrame, list[DroppedRow], int]` (clean_df has columns `date, open, high, low, close, volume`, `int` is total rows parsed before any drop).

- [ ] **Step 1: Write the fixture CSV**

```csv
# tests/fixtures/vnindex_sample.csv
Date,Open,High,Low,Close,Volume,,,,,,,
28/7/2000 00:00,100,100,100,100,4,200,,,,,,
31/7/2000 00:00,101.5,101.5,101.5,101.5,10,300,,,,,,
17/12/2024 00:00,1,264.91,1,265.45,1,260.6,1,261.72,358,215,808,
17/12/2024 00:00,1,264.91,1,265.45,1,260.6,1,261.72,358,215,808,
10/12/2004 00:00,229.6,230.1,230.1,230.1,510,830,,,,,,
13/7/2026 00:00,1,829.5,1,829.5,1,781.45,1,800.54,728,451,840,
```

This fixture encodes: a plain sub-1000 row (no comma-splitting), a
thousands-split row (`1,264.91` etc.), an exact byte-identical duplicate of
that row, an OHLC-invariant-violating row (`open=229.6 < low=230.1`), and a
row whose volume itself is thousands-split three times (`728,451,840`).

- [ ] **Step 2: Write failing tests**

```python
# tests/data/test_ingest.py
from pathlib import Path
import pandas as pd
import pytest
from raemf_mc.data.ingest import (
    parse_vnindex_csv,
    drop_exact_duplicates,
    validate_ohlc_invariants,
    clean_vnindex_data,
    DroppedRow,
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/data/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.data'`

- [ ] **Step 4: Write implementation**

```python
# src/raemf_mc/data/__init__.py
```

```python
# src/raemf_mc/data/ingest.py
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
```

```python
# tests/data/__init__.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_ingest.py -v`
Expected: PASS (6 tests, or 5 + 1 skip if the real CSV path differs)

- [ ] **Step 6: Commit**

```bash
git add src/raemf_mc/data tests/data tests/fixtures
git commit -m "feat: ingest and validate raw VN-Index OHLCV CSV"
```

---

## Task 4: Public data loader + ingestion notes report

**Files:**
- Create: `src/raemf_mc/data/loader.py`
- Create: `scripts/generate_ingestion_notes.py`
- Test: `tests/data/test_loader.py`

**Interfaces:**
- Consumes: `clean_vnindex_data` from Task 3 (`src/raemf_mc/data/ingest.py`).
- Produces: `load_vnindex_ohlcv(path=None, max_drop_fraction=0.05) -> pd.DataFrame` (DatetimeIndex named `date`, columns `open, high, low, close, volume`); `load_vnindex_ohlcv_with_report(path=None, max_drop_fraction=0.05) -> tuple[pd.DataFrame, list[DroppedRow], int]`; `write_ingestion_notes(dropped, total_parsed, final_count, output_path) -> None`.

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_loader.py
from pathlib import Path
import pandas as pd
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/data/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/data/loader.py
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
```

- [ ] **Step 4: Write the notes-generation script**

```python
# scripts/generate_ingestion_notes.py
from pathlib import Path

from raemf_mc.data.loader import load_vnindex_ohlcv_with_report, write_ingestion_notes

if __name__ == "__main__":
    df, dropped, total = load_vnindex_ohlcv_with_report()
    output = Path(__file__).resolve().parents[1] / "docs" / "data_ingestion_notes.md"
    write_ingestion_notes(dropped, total_parsed=total, final_count=len(df), output_path=output)
    print(f"wrote {output} ({len(dropped)} dropped rows logged)")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_loader.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Generate the real ingestion notes doc and commit everything**

Run: `python scripts/generate_ingestion_notes.py`
Expected output: `wrote .../docs/data_ingestion_notes.md (90 dropped rows logged)`

```bash
git add src/raemf_mc/data/loader.py scripts/generate_ingestion_notes.py tests/data/test_loader.py docs/data_ingestion_notes.md
git commit -m "feat: add VN-Index data loader and generate ingestion notes report"
```

---

## Task 5: Log-return feature

**Files:**
- Create: `src/raemf_mc/features/__init__.py`
- Create: `src/raemf_mc/features/returns.py`
- Test: `tests/features/test_returns.py`
- Create: `tests/features/__init__.py`

**Interfaces:**
- Produces: `compute_log_returns(ohlcv: pd.DataFrame, price_col: str = "close") -> pd.Series`.

- [ ] **Step 1: Write failing test**

```python
# tests/features/test_returns.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/features -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/features/__init__.py
```

```python
# src/raemf_mc/features/returns.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(ohlcv: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Log return r_t = log(P_t / P_{t-1}). The first value has no
    predecessor and is dropped rather than filled."""
    prices = ohlcv[price_col]
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.dropna()
```

```python
# tests/features/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/features tests/features
git commit -m "feat: add log-return feature computation"
```

---

## Task 6: Purged split boundary logic

**Files:**
- Create: `src/raemf_mc/validation/__init__.py`
- Create: `src/raemf_mc/validation/purged_split.py`
- Test: `tests/validation/test_purged_split.py`
- Create: `tests/validation/__init__.py`

**Interfaces:**
- Produces: `target_end_dates(dates: pd.DatetimeIndex, horizon: int) -> pd.Series` (NaT where the h-step-ahead target doesn't exist); `purged_train_mask(dates: pd.DatetimeIndex, boundary, horizon: int) -> np.ndarray` (bool mask, True only where the target exists and is `< boundary`).

- [ ] **Step 1: Write failing tests**

```python
# tests/validation/test_purged_split.py
import numpy as np
import pandas as pd
from raemf_mc.validation.purged_split import target_end_dates, purged_train_mask


def _dates(n=30):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_target_end_dates_shifts_by_horizon():
    dates = _dates(10)
    targets = target_end_dates(dates, horizon=3)
    assert targets.iloc[0] == dates[3]
    assert pd.isna(targets.iloc[-1])
    assert pd.isna(targets.iloc[-3])
    assert not pd.isna(targets.iloc[-4])


def test_purged_train_mask_excludes_rows_without_valid_target():
    dates = _dates(10)
    boundary = dates[9] + pd.Timedelta(days=100)  # boundary far in the future
    mask = purged_train_mask(dates, boundary, horizon=3)
    # last 3 rows have no horizon-3 target at all -> must be excluded
    assert not mask[-3:].any()
    assert mask[: 10 - 3].all()


def test_purged_train_mask_excludes_rows_whose_target_crosses_boundary():
    dates = _dates(20)
    boundary = dates[10]
    mask = purged_train_mask(dates, boundary, horizon=5)
    targets = target_end_dates(dates, horizon=5)
    included_targets = targets[mask]
    assert (included_targets < boundary).all()
    assert included_targets.notna().all()
    # sanity: at least one row is legitimately included
    assert mask.sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/validation -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/validation/__init__.py
```

```python
# src/raemf_mc/validation/purged_split.py
from __future__ import annotations

import numpy as np
import pandas as pd


def target_end_dates(dates: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """For each position i, the date at i+horizon (NaT if out of range) —
    the date on which the h-step-ahead target for row i is realized."""
    n = len(dates)
    result = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    valid_n = n - horizon
    if valid_n > 0:
        result.iloc[:valid_n] = dates[horizon:]
    return result


def purged_train_mask(dates: pd.DatetimeIndex, boundary, horizon: int) -> np.ndarray:
    """Boolean mask: True for rows whose target_end_date_h exists and is
    strictly before boundary."""
    targets = target_end_dates(dates, horizon)
    boundary_ts = pd.Timestamp(boundary)
    return (targets.notna() & (targets < boundary_ts)).to_numpy()
```

```python
# tests/validation/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/validation -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/validation tests/validation
git commit -m "feat: add purged train mask for horizon-based leakage prevention"
```

---

## Task 7: Leakage assertion helper

**Files:**
- Create: `src/raemf_mc/validation/leakage_checks.py`
- Test: `tests/validation/test_leakage_checks.py`

**Interfaces:**
- Consumes: `target_end_dates` from Task 6 (`src/raemf_mc/validation/purged_split.py`).
- Produces: `assert_no_target_leakage(dates: pd.DatetimeIndex, mask: np.ndarray, boundary, horizon: int) -> None` (raises `AssertionError` on any violation).

- [ ] **Step 1: Write failing tests**

```python
# tests/validation/test_leakage_checks.py
import numpy as np
import pandas as pd
import pytest
from raemf_mc.validation.purged_split import purged_train_mask
from raemf_mc.validation.leakage_checks import assert_no_target_leakage


def test_assert_no_target_leakage_passes_for_valid_mask():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    boundary = dates[20]
    mask = purged_train_mask(dates, boundary, horizon=5)
    assert_no_target_leakage(dates, mask, boundary, horizon=5)  # must not raise


def test_assert_no_target_leakage_raises_for_leaky_mask():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    boundary = dates[20]
    leaky_mask = np.ones(len(dates), dtype=bool)  # includes everything, incl. leaky rows
    with pytest.raises(AssertionError):
        assert_no_target_leakage(dates, leaky_mask, boundary, horizon=5)


def test_assert_no_target_leakage_raises_for_missing_target():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    boundary = dates[9] + pd.Timedelta(days=100)
    mask = np.ones(len(dates), dtype=bool)  # last rows have no horizon-5 target
    with pytest.raises(AssertionError):
        assert_no_target_leakage(dates, mask, boundary, horizon=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/validation/test_leakage_checks.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/validation/leakage_checks.py
from __future__ import annotations

import numpy as np
import pandas as pd

from raemf_mc.validation.purged_split import target_end_dates


def assert_no_target_leakage(
    dates: pd.DatetimeIndex, mask: np.ndarray, boundary, horizon: int
) -> None:
    """Raise AssertionError if any row selected by mask has a horizon-h
    target_end_date that is missing or >= boundary."""
    targets = target_end_dates(dates, horizon)
    boundary_ts = pd.Timestamp(boundary)
    selected = targets[np.asarray(mask, dtype=bool)]
    if selected.isna().any():
        raise AssertionError(
            "mask selects rows with no valid horizon target (leakage risk at series tail)"
        )
    leaky = selected[selected >= boundary_ts]
    if len(leaky) > 0:
        raise AssertionError(
            f"target leakage: {len(leaky)} rows have target_end_date >= boundary {boundary_ts}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/validation/test_leakage_checks.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/validation/leakage_checks.py tests/validation/test_leakage_checks.py
git commit -m "feat: add reusable target-leakage assertion helper"
```

---

## Task 8: ADVI engine — single-seed mean-field fit

**Files:**
- Create: `src/raemf_mc/bayesian/__init__.py`
- Create: `src/raemf_mc/bayesian/torch_backend.py`
- Test: `tests/bayesian/test_torch_backend_single_seed.py`
- Create: `tests/bayesian/__init__.py`

**Interfaces:**
- Produces: `AdviConfig` dataclass; `FitResult` dataclass (`mu, log_sigma: Tensor; elbo_trace: list[float]; converged: bool; fallback_used: bool; fallback_reason: str | None; n_retries: int; seed: int`); `assert_no_float16(*tensors: Tensor) -> None`; `append_fallback_log(event: dict, path) -> None`; `fit_mean_field_advi(log_joint_fn, init_mu, init_log_sigma, config, seed, device, fallback_log_path="fallbacks.json") -> FitResult`.

- [ ] **Step 1: Write failing tests**

```python
# tests/bayesian/test_torch_backend_single_seed.py
import json
import torch
import pytest
from raemf_mc.bayesian.torch_backend import (
    AdviConfig,
    FitResult,
    assert_no_float16,
    fit_mean_field_advi,
)


def test_assert_no_float16_raises_on_float16_tensor():
    ok = torch.zeros(3, dtype=torch.float32)
    bad = torch.zeros(3, dtype=torch.float16)
    assert_no_float16(ok)  # must not raise
    with pytest.raises(TypeError):
        assert_no_float16(ok, bad)


def _gaussian_log_joint(theta: torch.Tensor) -> torch.Tensor:
    # log density of a standard 2D Gaussian at theta — known analytic
    # posterior: mu -> 0, sigma -> 1 (up to ADVI's mean-field approximation).
    return -0.5 * torch.sum(theta**2) - theta.shape[0] * 0.5 * torch.log(
        torch.tensor(2 * 3.141592653589793)
    )


def test_fit_mean_field_advi_recovers_known_gaussian_mean(tmp_path):
    config = AdviConfig(n_steps=300, learning_rate=0.05, warmup_steps=20, elbo_ma_window=10, early_stop_patience=300)
    init_mu = torch.full((2,), 3.0)
    init_log_sigma = torch.zeros(2)
    result = fit_mean_field_advi(
        _gaussian_log_joint, init_mu, init_log_sigma, config, seed=0,
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert isinstance(result, FitResult)
    assert torch.allclose(result.mu, torch.zeros(2), atol=0.3)
    assert len(result.elbo_trace) > 0
    assert result.fallback_used is False


def test_fit_mean_field_advi_rejects_float16_init(tmp_path):
    config = AdviConfig(n_steps=5)
    bad_mu = torch.zeros(2, dtype=torch.float16)
    init_log_sigma = torch.zeros(2)
    with pytest.raises(TypeError):
        fit_mean_field_advi(
            _gaussian_log_joint, bad_mu, init_log_sigma, config, seed=0,
            device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
        )


def _diverging_log_joint(theta: torch.Tensor) -> torch.Tensor:
    # deliberately produces NaN once theta grows large, to exercise the
    # retry/fallback path.
    huge = torch.exp(theta.sum() * 50.0)
    return -huge


def test_fit_mean_field_advi_logs_fallback_on_persistent_divergence(tmp_path):
    log_path = tmp_path / "fallbacks.json"
    config = AdviConfig(
        n_steps=50, learning_rate=5.0, warmup_steps=0, max_retries=1, retry_lr_factor=0.5,
    )
    init_mu = torch.full((1,), 10.0)
    init_log_sigma = torch.zeros(1)
    result = fit_mean_field_advi(
        _diverging_log_joint, init_mu, init_log_sigma, config, seed=0,
        device=torch.device("cpu"), fallback_log_path=log_path,
    )
    assert result.fallback_used is True
    assert result.fallback_reason is not None
    assert log_path.exists()
    events = json.loads(log_path.read_text())
    assert len(events) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/bayesian -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/bayesian/__init__.py
```

```python
# src/raemf_mc/bayesian/torch_backend.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch


@dataclass
class AdviConfig:
    n_steps: int = 2000
    learning_rate: float = 0.01
    warmup_steps: int = 100
    grad_clip_norm: float = 10.0
    elbo_ma_window: int = 50
    early_stop_patience: int = 200
    min_delta: float = 1e-3
    retry_lr_factor: float = 0.5
    max_retries: int = 3
    n_mc_samples: int = 4


@dataclass
class FitResult:
    mu: torch.Tensor
    log_sigma: torch.Tensor
    elbo_trace: list[float]
    converged: bool
    fallback_used: bool
    fallback_reason: str | None
    n_retries: int
    seed: int


def assert_no_float16(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if t.dtype == torch.float16:
            raise TypeError(
                "float16 tensors are not allowed in ELBO/log-likelihood/logsumexp computations"
            )


def append_fallback_log(event: dict, path: str | Path = "fallbacks.json") -> None:
    path = Path(path)
    existing = []
    if path.exists():
        existing = json.loads(path.read_text())
    existing.append(event)
    path.write_text(json.dumps(existing, indent=2, default=str))


def _gaussian_entropy(log_sigma: torch.Tensor) -> torch.Tensor:
    # entropy of a diagonal Gaussian: 0.5 * sum(log(2*pi*e) + 2*log_sigma)
    return 0.5 * torch.sum(math.log(2 * math.pi * math.e) + 2 * log_sigma)


def _lr_schedule(step: int, config: AdviConfig) -> float:
    if step < config.warmup_steps and config.warmup_steps > 0:
        return (step + 1) / config.warmup_steps
    remaining = max(config.n_steps - config.warmup_steps, 1)
    progress = min((step - config.warmup_steps) / remaining, 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def _single_attempt(
    log_joint_fn: Callable[[torch.Tensor], torch.Tensor],
    init_mu: torch.Tensor,
    init_log_sigma: torch.Tensor,
    config: AdviConfig,
    seed: int,
    device: torch.device,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, list[float], bool]:
    generator = torch.Generator(device=device).manual_seed(seed)
    mu = init_mu.clone().to(device).requires_grad_(True)
    log_sigma = init_log_sigma.clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([mu, log_sigma], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _lr_schedule(step, config)
    )

    elbo_trace: list[float] = []
    moving_avg = None
    best_moving_avg = -math.inf
    patience_counter = 0
    last_finite_mu, last_finite_log_sigma = mu.detach().clone(), log_sigma.detach().clone()

    for step in range(config.n_steps):
        optimizer.zero_grad()
        sigma = torch.exp(log_sigma)
        elbo_samples = []
        for _ in range(config.n_mc_samples):
            eps = torch.randn(mu.shape, generator=generator, device=device)
            theta = mu + sigma * eps
            elbo_samples.append(log_joint_fn(theta))
        mean_log_joint = torch.stack(elbo_samples).mean()
        elbo = mean_log_joint + _gaussian_entropy(log_sigma)
        loss = -elbo

        if not torch.isfinite(loss):
            return last_finite_mu, last_finite_log_sigma, elbo_trace, False

        loss.backward()
        torch.nn.utils.clip_grad_norm_([mu, log_sigma], config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        elbo_value = elbo.item()
        elbo_trace.append(elbo_value)
        last_finite_mu, last_finite_log_sigma = mu.detach().clone(), log_sigma.detach().clone()

        window = elbo_trace[-config.elbo_ma_window :]
        moving_avg = sum(window) / len(window)
        if moving_avg > best_moving_avg + config.min_delta:
            best_moving_avg = moving_avg
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.early_stop_patience:
            break

    return last_finite_mu, last_finite_log_sigma, elbo_trace, True


def fit_mean_field_advi(
    log_joint_fn: Callable[[torch.Tensor], torch.Tensor],
    init_mu: torch.Tensor,
    init_log_sigma: torch.Tensor,
    config: AdviConfig,
    seed: int,
    device: torch.device,
    fallback_log_path: str | Path = "fallbacks.json",
) -> FitResult:
    assert_no_float16(init_mu, init_log_sigma)

    lr = config.learning_rate
    n_retries = 0
    fallback_used = False
    fallback_reason = None
    mu, log_sigma, elbo_trace, converged = _single_attempt(
        log_joint_fn, init_mu, init_log_sigma, config, seed, device, lr
    )

    while not converged and n_retries < config.max_retries:
        n_retries += 1
        lr = lr * config.retry_lr_factor
        event = {
            "seed": seed,
            "retry_number": n_retries,
            "new_learning_rate": lr,
            "reason": "elbo_diverged_nan_or_inf",
        }
        append_fallback_log(event, fallback_log_path)
        mu, log_sigma, elbo_trace, converged = _single_attempt(
            log_joint_fn, init_mu, init_log_sigma, config, seed, device, lr
        )

    if not converged:
        fallback_used = True
        fallback_reason = "exhausted_retries_reverted_to_last_finite_step"
        append_fallback_log(
            {"seed": seed, "reason": fallback_reason, "n_retries": n_retries},
            fallback_log_path,
        )

    return FitResult(
        mu=mu,
        log_sigma=log_sigma,
        elbo_trace=elbo_trace,
        converged=converged,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        n_retries=n_retries,
        seed=seed,
    )
```

```python
# tests/bayesian/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/bayesian/test_torch_backend_single_seed.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/bayesian tests/bayesian
git commit -m "feat: add single-seed mean-field ADVI engine with retry/fallback logging"
```

---

## Task 9: ADVI engine — multi-seed pooling and joint-draw sampling

**Files:**
- Modify: `src/raemf_mc/bayesian/torch_backend.py`
- Test: `tests/bayesian/test_torch_backend_multi_seed.py`

**Interfaces:**
- Consumes: `FitResult`, `AdviConfig`, `fit_mean_field_advi` from Task 8.
- Produces: `PooledPosterior` dataclass (`seed_results: list[FitResult]`); `fit_multi_seed_advi(log_joint_fn, init_mu, init_log_sigma, config, seeds, device, fallback_log_path="fallbacks.json") -> PooledPosterior`; `sample_joint_draw(posterior: PooledPosterior, generator=None) -> torch.Tensor`.

- [ ] **Step 1: Write failing tests**

```python
# tests/bayesian/test_torch_backend_multi_seed.py
import torch
from raemf_mc.bayesian.torch_backend import (
    AdviConfig,
    PooledPosterior,
    fit_multi_seed_advi,
    sample_joint_draw,
)


def _gaussian_log_joint(theta: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(theta**2)


def test_fit_multi_seed_advi_pools_all_seeds_equally(tmp_path):
    config = AdviConfig(n_steps=100, learning_rate=0.05, warmup_steps=10, elbo_ma_window=10, early_stop_patience=100)
    init_mu = torch.full((2,), 2.0)
    init_log_sigma = torch.zeros(2)
    posterior = fit_multi_seed_advi(
        _gaussian_log_joint, init_mu, init_log_sigma, config, seeds=[0, 1, 2],
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert isinstance(posterior, PooledPosterior)
    assert len(posterior.seed_results) == 3
    assert {r.seed for r in posterior.seed_results} == {0, 1, 2}


def test_sample_joint_draw_is_deterministic_given_generator(tmp_path):
    config = AdviConfig(n_steps=50, learning_rate=0.05, warmup_steps=5, elbo_ma_window=10, early_stop_patience=50)
    init_mu = torch.zeros(2)
    init_log_sigma = torch.zeros(2)
    posterior = fit_multi_seed_advi(
        _gaussian_log_joint, init_mu, init_log_sigma, config, seeds=[0, 1],
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    draw1 = sample_joint_draw(posterior, generator=gen1)
    draw2 = sample_joint_draw(posterior, generator=gen2)
    assert torch.equal(draw1, draw2)
    assert draw1.shape == (2,)


def test_sample_joint_draw_varies_across_calls_without_fixed_generator(tmp_path):
    config = AdviConfig(n_steps=20, learning_rate=0.05, warmup_steps=2, elbo_ma_window=5, early_stop_patience=20)
    init_mu = torch.zeros(2)
    init_log_sigma = torch.zeros(2)
    posterior = fit_multi_seed_advi(
        _gaussian_log_joint, init_mu, init_log_sigma, config, seeds=[0, 1, 2, 3],
        device=torch.device("cpu"), fallback_log_path=tmp_path / "fallbacks.json",
    )
    draws = [sample_joint_draw(posterior) for _ in range(20)]
    assert not all(torch.equal(draws[0], d) for d in draws[1:])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/bayesian/test_torch_backend_multi_seed.py -v`
Expected: FAIL with `ImportError: cannot import name 'PooledPosterior'`

- [ ] **Step 3: Append implementation to `torch_backend.py`**

```python
# append to src/raemf_mc/bayesian/torch_backend.py

@dataclass
class PooledPosterior:
    seed_results: list[FitResult]


def fit_multi_seed_advi(
    log_joint_fn: Callable[[torch.Tensor], torch.Tensor],
    init_mu: torch.Tensor,
    init_log_sigma: torch.Tensor,
    config: AdviConfig,
    seeds: list[int],
    device: torch.device,
    fallback_log_path: str | Path = "fallbacks.json",
) -> PooledPosterior:
    """Fit one mean-field posterior per seed and pool them as an
    equal-weight mixture — never keep only the best-ELBO seed."""
    results = [
        fit_mean_field_advi(
            log_joint_fn, init_mu, init_log_sigma, config, seed, device, fallback_log_path
        )
        for seed in seeds
    ]
    return PooledPosterior(seed_results=results)


def sample_joint_draw(
    posterior: PooledPosterior, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw exactly one flat theta vector from the pooled posterior: pick
    one seed uniformly at random, then reparameterize-sample once from
    that seed's mean-field Gaussian. The caller is responsible for
    reusing the returned tensor for an entire Monte Carlo path rather
    than resampling at each horizon step — this function performs no
    caching itself."""
    n_seeds = len(posterior.seed_results)
    idx = int(torch.randint(n_seeds, (1,), generator=generator).item())
    fr = posterior.seed_results[idx]
    eps = torch.randn(fr.mu.shape, generator=generator)
    return fr.mu + torch.exp(fr.log_sigma) * eps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/bayesian -v`
Expected: PASS (all tests in `tests/bayesian`)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/bayesian/torch_backend.py tests/bayesian/test_torch_backend_multi_seed.py
git commit -m "feat: add multi-seed posterior pooling and single joint-draw sampling"
```

---

## Task 10: Hierarchical shrinkage prior utilities

**Files:**
- Create: `src/raemf_mc/bayesian/priors.py`
- Test: `tests/bayesian/test_priors.py`

**Interfaces:**
- Produces: `HierarchicalPriorConfig` dataclass (`hyper_mean_scale: float = 1.0, min_effective_observations: float = 30.0`); `state_shrinkage_weight(effective_obs: Tensor, min_effective_observations: float) -> Tensor`; `hierarchical_normal_log_prob(state_params: Tensor, hyper_mean: Tensor, base_scale: Tensor, shrinkage_weight: Tensor) -> Tensor`.

- [ ] **Step 1: Write failing tests**

```python
# tests/bayesian/test_priors.py
import torch
from raemf_mc.bayesian.priors import (
    HierarchicalPriorConfig,
    state_shrinkage_weight,
    hierarchical_normal_log_prob,
)


def test_state_shrinkage_weight_is_lower_for_sparse_states():
    effective_obs = torch.tensor([5.0, 30.0, 100.0])
    weight = state_shrinkage_weight(effective_obs, min_effective_observations=30.0)
    assert weight[0] < weight[1] <= weight[2]
    assert torch.all(weight > 0) and torch.all(weight <= 1.0)


def test_hierarchical_normal_log_prob_penalizes_deviation_from_hyper_mean():
    hyper_mean = torch.tensor(0.0)
    base_scale = torch.tensor(1.0)
    weight = torch.tensor(1.0)
    close = hierarchical_normal_log_prob(torch.tensor([0.1]), hyper_mean, base_scale, weight)
    far = hierarchical_normal_log_prob(torch.tensor([5.0]), hyper_mean, base_scale, weight)
    assert close > far


def test_hierarchical_normal_log_prob_tighter_for_low_shrinkage_weight():
    hyper_mean = torch.tensor(0.0)
    base_scale = torch.tensor(1.0)
    deviation = torch.tensor([2.0])
    low_weight_logprob = hierarchical_normal_log_prob(deviation, hyper_mean, base_scale, torch.tensor(0.1))
    high_weight_logprob = hierarchical_normal_log_prob(deviation, hyper_mean, base_scale, torch.tensor(1.0))
    # smaller shrinkage_weight -> tighter effective prior std -> lower
    # density further from the hyper-mean at a fixed deviation
    assert low_weight_logprob < high_weight_logprob


def test_hierarchical_prior_config_defaults():
    config = HierarchicalPriorConfig()
    assert config.hyper_mean_scale == 1.0
    assert config.min_effective_observations == 30.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/bayesian/test_priors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.bayesian.priors'`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/bayesian/priors.py
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HierarchicalPriorConfig:
    hyper_mean_scale: float = 1.0
    min_effective_observations: float = 30.0


def state_shrinkage_weight(
    effective_obs: torch.Tensor, min_effective_observations: float
) -> torch.Tensor:
    """Per-state weight in (0, 1]: states with effective_obs far below
    min_effective_observations get a smaller weight (tighter shrinkage
    toward the hyper-mean); states with ample data approach weight 1."""
    return torch.clamp(effective_obs / min_effective_observations, min=0.05, max=1.0)


def hierarchical_normal_log_prob(
    state_params: torch.Tensor,
    hyper_mean: torch.Tensor,
    base_scale: torch.Tensor,
    shrinkage_weight: torch.Tensor,
) -> torch.Tensor:
    """log p(state_params | hyper_mean) under
    Normal(hyper_mean, base_scale / shrinkage_weight). A smaller
    shrinkage_weight tightens the effective prior std, pulling the state
    parameter harder toward the shared hyper-mean."""
    scale = base_scale / shrinkage_weight
    dist = torch.distributions.Normal(hyper_mean, scale)
    return dist.log_prob(state_params).sum()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/bayesian/test_priors.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/bayesian/priors.py tests/bayesian/test_priors.py
git commit -m "feat: add hierarchical shrinkage prior utilities for sparse regimes"
```

---

## Task 11: ELBO and seed-stability diagnostics

**Files:**
- Create: `src/raemf_mc/bayesian/diagnostics.py`
- Test: `tests/bayesian/test_diagnostics.py`

**Interfaces:**
- Consumes: `FitResult`, `PooledPosterior` from Tasks 8-9.
- Produces: `summarize_elbo_trace(elbo_trace: list[float]) -> dict` (`final_elbo, best_elbo, n_steps`); `seed_stability_metrics(seed_results: list[FitResult]) -> dict` (`final_elbo_std, max_pairwise_mu_diff`).

- [ ] **Step 1: Write failing tests**

```python
# tests/bayesian/test_diagnostics.py
import torch
from raemf_mc.bayesian.torch_backend import FitResult
from raemf_mc.bayesian.diagnostics import summarize_elbo_trace, seed_stability_metrics


def test_summarize_elbo_trace_reports_final_and_best():
    trace = [-10.0, -5.0, -6.0, -4.0, -4.5]
    summary = summarize_elbo_trace(trace)
    assert summary["final_elbo"] == -4.5
    assert summary["best_elbo"] == -4.0
    assert summary["n_steps"] == 5


def test_seed_stability_metrics_zero_for_identical_seeds():
    results = [
        FitResult(mu=torch.zeros(2), log_sigma=torch.zeros(2), elbo_trace=[-1.0, -1.0],
                   converged=True, fallback_used=False, fallback_reason=None, n_retries=0, seed=s)
        for s in range(3)
    ]
    metrics = seed_stability_metrics(results)
    assert metrics["final_elbo_std"] == 0.0
    assert metrics["max_pairwise_mu_diff"] == 0.0


def test_seed_stability_metrics_nonzero_for_divergent_seeds():
    results = [
        FitResult(mu=torch.zeros(2), log_sigma=torch.zeros(2), elbo_trace=[-1.0],
                   converged=True, fallback_used=False, fallback_reason=None, n_retries=0, seed=0),
        FitResult(mu=torch.ones(2) * 5, log_sigma=torch.zeros(2), elbo_trace=[-9.0],
                   converged=True, fallback_used=False, fallback_reason=None, n_retries=0, seed=1),
    ]
    metrics = seed_stability_metrics(results)
    assert metrics["final_elbo_std"] > 0.0
    assert metrics["max_pairwise_mu_diff"] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/bayesian/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/bayesian/diagnostics.py
from __future__ import annotations

import itertools

import torch

from raemf_mc.bayesian.torch_backend import FitResult


def summarize_elbo_trace(elbo_trace: list[float]) -> dict:
    return {
        "final_elbo": elbo_trace[-1],
        "best_elbo": max(elbo_trace),
        "n_steps": len(elbo_trace),
    }


def seed_stability_metrics(seed_results: list[FitResult]) -> dict:
    final_elbos = torch.tensor([r.elbo_trace[-1] for r in seed_results])
    final_elbo_std = float(final_elbos.std(unbiased=False))

    max_diff = 0.0
    for a, b in itertools.combinations(seed_results, 2):
        diff = float(torch.max(torch.abs(a.mu - b.mu)))
        max_diff = max(max_diff, diff)

    return {"final_elbo_std": final_elbo_std, "max_pairwise_mu_diff": max_diff}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/bayesian/test_diagnostics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/bayesian/diagnostics.py tests/bayesian/test_diagnostics.py
git commit -m "feat: add ELBO trace and multi-seed stability diagnostics"
```

---

## Task 12: MS-EGARCH parameter layout and math primitives

**Files:**
- Create: `src/raemf_mc/regime/__init__.py`
- Create: `src/raemf_mc/regime/ms_egarch.py`
- Test: `tests/regime/test_ms_egarch_params.py`
- Create: `tests/regime/__init__.py`

**Interfaces:**
- Produces: `N_STATES = 4`; `MSEGARCHParamLayout` dataclass (`n_states=4`, properties `n_egarch_params, n_transition_params, n_nu_params, total`); `MSEGARCHParams` dataclass (`omega, alpha, beta, gamma, transition_logits, nu_raw: Tensor`); `unpack_params(theta: Tensor, layout=MSEGARCHParamLayout()) -> MSEGARCHParams`; `transition_matrix(transition_logits: Tensor) -> Tensor`; `nu_from_raw(nu_raw: Tensor, min_nu: float = 2.05) -> Tensor`; `expected_abs_standardized_t(nu: Tensor) -> Tensor`; `student_t_log_pdf_with_variance(x, loc, variance, nu) -> Tensor`.

- [ ] **Step 1: Write failing tests**

```python
# tests/regime/test_ms_egarch_params.py
import math
import torch
from scipy import stats
from raemf_mc.regime.ms_egarch import (
    N_STATES,
    MSEGARCHParamLayout,
    unpack_params,
    transition_matrix,
    nu_from_raw,
    expected_abs_standardized_t,
    student_t_log_pdf_with_variance,
)


def test_param_layout_totals_match_spec_budget():
    layout = MSEGARCHParamLayout()
    assert layout.n_states == 4
    assert layout.n_egarch_params == 16
    assert layout.n_transition_params == 12
    assert layout.n_nu_params == 1
    assert layout.total == 29  # 28 core + 1 global nu


def test_unpack_params_round_trip_shapes():
    layout = MSEGARCHParamLayout()
    theta = torch.arange(layout.total, dtype=torch.float32)
    params = unpack_params(theta, layout)
    assert params.omega.shape == (4,)
    assert params.alpha.shape == (4,)
    assert params.beta.shape == (4,)
    assert params.gamma.shape == (4,)
    assert params.transition_logits.shape == (4, 3)
    assert params.nu_raw.shape == ()


def test_transition_matrix_rows_sum_to_one():
    logits = torch.randn(4, 3)
    trans = transition_matrix(logits)
    assert trans.shape == (4, 4)
    row_sums = trans.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-6)
    assert torch.all(trans >= 0)


def test_transition_matrix_zero_logits_gives_uniform_rows():
    logits = torch.zeros(4, 3)
    trans = transition_matrix(logits)
    assert torch.allclose(trans, torch.full((4, 4), 0.25), atol=1e-6)


def test_nu_from_raw_always_above_min():
    raw = torch.tensor([-100.0, 0.0, 100.0])
    nu = nu_from_raw(raw, min_nu=2.05)
    assert torch.all(nu > 2.05)


def test_expected_abs_standardized_t_matches_scipy_reference():
    nu_value = 8.0
    nu = torch.tensor(nu_value)
    computed = expected_abs_standardized_t(nu).item()
    # E|T| for raw Student-t(nu), then rescale to unit variance
    scale_factor = math.sqrt((nu_value - 2) / nu_value)
    raw_t_mean_abs = 2.0 * math.sqrt(nu_value) * math.exp(
        math.lgamma((nu_value + 1) / 2) - math.lgamma(nu_value / 2)
    ) / ((nu_value - 1) * math.sqrt(math.pi))
    expected = raw_t_mean_abs * scale_factor
    assert math.isclose(computed, expected, rel_tol=1e-4)


def test_student_t_log_pdf_with_variance_matches_scipy_for_unit_variance():
    nu_value = 6.0
    variance = torch.tensor(2.5)
    loc = torch.tensor(1.0)
    x = torch.tensor(1.8)
    nu = torch.tensor(nu_value)
    computed = student_t_log_pdf_with_variance(x, loc, variance, nu).item()

    scale_param = math.sqrt(float(variance) * (nu_value - 2) / nu_value)
    expected = stats.t.logpdf(x.item(), df=nu_value, loc=loc.item(), scale=scale_param)
    assert math.isclose(computed, expected, rel_tol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/regime -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/regime/__init__.py
```

```python
# src/raemf_mc/regime/ms_egarch.py
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

N_STATES = 4


@dataclass(frozen=True)
class MSEGARCHParamLayout:
    n_states: int = N_STATES

    @property
    def n_egarch_params(self) -> int:
        return 4 * self.n_states

    @property
    def n_transition_params(self) -> int:
        return self.n_states * (self.n_states - 1)

    @property
    def n_nu_params(self) -> int:
        return 1

    @property
    def total(self) -> int:
        return self.n_egarch_params + self.n_transition_params + self.n_nu_params


@dataclass(frozen=True)
class MSEGARCHParams:
    omega: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    transition_logits: torch.Tensor
    nu_raw: torch.Tensor


def unpack_params(
    theta: torch.Tensor, layout: MSEGARCHParamLayout = MSEGARCHParamLayout()
) -> MSEGARCHParams:
    n = layout.n_states
    idx = 0
    omega = theta[idx : idx + n]
    idx += n
    alpha = theta[idx : idx + n]
    idx += n
    beta = theta[idx : idx + n]
    idx += n
    gamma = theta[idx : idx + n]
    idx += n
    transition_logits = theta[idx : idx + n * (n - 1)].reshape(n, n - 1)
    idx += n * (n - 1)
    nu_raw = theta[idx : idx + 1].squeeze(-1)
    return MSEGARCHParams(omega, alpha, beta, gamma, transition_logits, nu_raw)


def transition_matrix(transition_logits: torch.Tensor) -> torch.Tensor:
    """Each row's simplex = softmax of [0, logits_row] — fixes one free
    reference category per row so an (n_states-1)-length logit vector maps
    onto an n_states-simplex (avoids Dirichlet, which is a poor fit for
    ADVI's Gaussian variational family)."""
    n = transition_logits.shape[0]
    zeros = torch.zeros(n, 1, dtype=transition_logits.dtype, device=transition_logits.device)
    full_logits = torch.cat([zeros, transition_logits], dim=1)
    return torch.softmax(full_logits, dim=1)


def nu_from_raw(nu_raw: torch.Tensor, min_nu: float = 2.05) -> torch.Tensor:
    return min_nu + torch.nn.functional.softplus(nu_raw)


def expected_abs_standardized_t(nu: torch.Tensor) -> torch.Tensor:
    """E|Z| for Z = a unit-variance-standardized Student-t with df=nu
    (nu > 2), used as the E|z| term in the EGARCH leverage recursion."""
    log_numer = math.log(2.0) + 0.5 * torch.log(nu - 2) + torch.lgamma((nu + 1) / 2)
    log_denom = torch.log(nu - 1) + 0.5 * math.log(math.pi) + torch.lgamma(nu / 2)
    return torch.exp(log_numer - log_denom)


def _student_t_log_pdf(
    x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    z = (x - loc) / scale
    log_kernel = -0.5 * (nu + 1) * torch.log1p((z**2) / nu)
    log_norm = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * torch.tensor(math.pi))
        - torch.log(scale)
    )
    return log_norm + log_kernel


def student_t_log_pdf_with_variance(
    x: torch.Tensor, loc: torch.Tensor, variance: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    """Student-t log-density parametrized so Var(X) = variance exactly
    (for nu > 2), rather than the raw location-scale parametrization where
    Var(X) = scale^2 * nu / (nu - 2)."""
    scale = torch.sqrt(variance * (nu - 2) / nu)
    return _student_t_log_pdf(x, loc, scale, nu)
```

```python
# tests/regime/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/regime/test_ms_egarch_params.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/regime tests/regime
git commit -m "feat: add MS-EGARCH parameter layout and Student-t math primitives"
```

---

## Task 13: Gray's collapsing recursion + Hamilton forward filter

**Files:**
- Modify: `src/raemf_mc/regime/ms_egarch.py`
- Test: `tests/regime/test_ms_egarch_recursion.py`

**Interfaces:**
- Consumes: `MSEGARCHParams`, `transition_matrix`, `nu_from_raw`, `expected_abs_standardized_t`, `student_t_log_pdf_with_variance` from Task 12.
- Produces: `run_ms_egarch_recursion(returns: Tensor, params: MSEGARCHParams, init_log_var: Tensor, init_log_state_prob: Tensor) -> dict` with keys `log_var (T,n), log_filtered_prob (T,n), log_var_bar (T,), total_log_lik (scalar), nu (scalar)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/regime/test_ms_egarch_recursion.py
import math
import torch
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParams,
    run_ms_egarch_recursion,
    expected_abs_standardized_t,
    student_t_log_pdf_with_variance,
)


def _single_regime_egarch_reference(
    returns: torch.Tensor, omega, alpha, beta, gamma, nu, init_log_var
) -> torch.Tensor:
    """Minimal single-regime EGARCH recursion used as a ground truth for
    the degenerate K=1 case."""
    T = returns.shape[0]
    log_var = torch.zeros(T)
    e_abs_z = expected_abs_standardized_t(nu)
    log_var_prev = init_log_var
    z_prev = torch.tensor(0.0)
    for t in range(T):
        log_var_t = omega + beta * log_var_prev + alpha * (torch.abs(z_prev) - e_abs_z) + gamma * z_prev
        log_var[t] = log_var_t
        sigma_t = torch.exp(0.5 * log_var_t)
        z_prev = returns[t] / sigma_t
        log_var_prev = log_var_t
    return log_var


def test_recursion_output_shapes_and_finiteness():
    torch.manual_seed(0)
    T, n = 20, 4
    returns = torch.randn(T) * 0.01
    params = MSEGARCHParams(
        omega=torch.full((n,), -0.1),
        alpha=torch.full((n,), 0.1),
        beta=torch.full((n,), 0.9),
        gamma=torch.full((n,), -0.05),
        transition_logits=torch.zeros(n, n - 1),
        nu_raw=torch.tensor(2.0),
    )
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 0.25))
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
    assert result["log_var"].shape == (T, n)
    assert result["log_filtered_prob"].shape == (T, n)
    assert result["log_var_bar"].shape == (T,)
    assert torch.isfinite(result["log_var"]).all()
    assert torch.isfinite(result["log_filtered_prob"]).all()
    assert torch.isfinite(result["total_log_lik"])


def test_filtered_probabilities_form_a_simplex_every_step():
    torch.manual_seed(1)
    T, n = 15, 4
    returns = torch.randn(T) * 0.01
    params = MSEGARCHParams(
        omega=torch.full((n,), -0.2),
        alpha=torch.full((n,), 0.15),
        beta=torch.full((n,), 0.85),
        gamma=torch.full((n,), 0.02),
        transition_logits=torch.randn(n, n - 1) * 0.3,
        nu_raw=torch.tensor(1.5),
    )
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 0.25))
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
    probs = torch.exp(result["log_filtered_prob"])
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(T), atol=1e-4)


def test_degenerate_single_state_matches_reference_egarch():
    torch.manual_seed(2)
    T = 25
    returns = torch.randn(T) * 0.01
    n = 4
    omega, alpha, beta, gamma = -0.15, 0.12, 0.88, -0.03
    nu_val = 6.0
    # push ALL transition mass onto state 0 staying in state 0 -> collapses
    # to a single-regime EGARCH driven purely by state 0's parameters.
    huge = 50.0
    transition_logits = torch.full((n, n - 1), -huge)
    params = MSEGARCHParams(
        omega=torch.tensor([omega, 5.0, 5.0, 5.0]),
        alpha=torch.tensor([alpha, 0.0, 0.0, 0.0]),
        beta=torch.tensor([beta, 0.0, 0.0, 0.0]),
        gamma=torch.tensor([gamma, 0.0, 0.0, 0.0]),
        transition_logits=transition_logits,
        nu_raw=torch.log(torch.exp(torch.tensor(nu_val - 2.05)) - 1.0),  # inverse-softplus
    )
    init_log_var = torch.tensor([0.0, -huge, -huge, -huge])
    init_log_state_prob = torch.log(
        torch.tensor([1.0 - 3e-8, 1e-8, 1e-8, 1e-8])
    )
    result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)

    ref_log_var = _single_regime_egarch_reference(
        returns, torch.tensor(omega), torch.tensor(alpha), torch.tensor(beta),
        torch.tensor(gamma), torch.tensor(nu_val), torch.tensor(0.0),
    )
    got_log_var_state0 = result["log_var"][:, 0]
    assert torch.allclose(got_log_var_state0, ref_log_var, atol=1e-2)


def test_forward_filter_is_causal_prefix_invariant():
    """Fitting on data[0:T] and data[0:T+k] must produce identical
    log_filtered_prob for t <= T-1 — the filter never looks ahead."""
    torch.manual_seed(3)
    T, extra, n = 15, 5, 4
    returns_full = torch.randn(T + extra) * 0.01
    params = MSEGARCHParams(
        omega=torch.full((n,), -0.2),
        alpha=torch.full((n,), 0.1),
        beta=torch.full((n,), 0.85),
        gamma=torch.full((n,), 0.01),
        transition_logits=torch.randn(n, n - 1) * 0.2,
        nu_raw=torch.tensor(1.0),
    )
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 0.25))

    result_short = run_ms_egarch_recursion(
        returns_full[:T], params, init_log_var, init_log_state_prob
    )
    result_long = run_ms_egarch_recursion(
        returns_full, params, init_log_var, init_log_state_prob
    )
    assert torch.allclose(
        result_short["log_filtered_prob"], result_long["log_filtered_prob"][:T], atol=1e-6
    )
    assert torch.allclose(result_short["log_var"], result_long["log_var"][:T], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/regime/test_ms_egarch_recursion.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_ms_egarch_recursion'`

- [ ] **Step 3: Append implementation to `ms_egarch.py`**

```python
# append to src/raemf_mc/regime/ms_egarch.py

def run_ms_egarch_recursion(
    returns: torch.Tensor,
    params: MSEGARCHParams,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
) -> dict:
    """Gray's-collapsing MS-EGARCH recursion fused with a causal Hamilton
    forward filter, entirely in log-domain for numerical stability.

    `returns` must already be centered (MS-EGARCH models the innovation
    process, not the conditional mean).
    """
    T = returns.shape[0]
    n = params.omega.shape[0]
    trans = transition_matrix(params.transition_logits)
    log_trans = torch.log(trans + 1e-12)
    nu = nu_from_raw(params.nu_raw)
    e_abs_z = expected_abs_standardized_t(nu)

    log_var = torch.zeros(T, n, dtype=returns.dtype, device=returns.device)
    log_filtered = torch.zeros(T, n, dtype=returns.dtype, device=returns.device)
    log_var_bar = torch.zeros(T, dtype=returns.dtype, device=returns.device)

    log_var_bar_prev = torch.logsumexp(init_log_state_prob + init_log_var, dim=0)
    log_filt_prev = init_log_state_prob
    z_prev = torch.zeros((), dtype=returns.dtype, device=returns.device)
    total_log_lik = torch.zeros((), dtype=returns.dtype, device=returns.device)

    for t in range(T):
        log_var_t = (
            params.omega
            + params.beta * log_var_bar_prev
            + params.alpha * (torch.abs(z_prev) - e_abs_z)
            + params.gamma * z_prev
        )
        log_var[t] = log_var_t

        # Hamilton predict step: log P(S_t=k | F_{t-1})
        log_pred = torch.logsumexp(log_trans + log_filt_prev.unsqueeze(1), dim=0)

        variance_t = torch.exp(log_var_t)
        loglik_t = student_t_log_pdf_with_variance(
            returns[t], torch.zeros_like(variance_t), variance_t, nu
        )
        log_joint_t = log_pred + loglik_t
        log_norm = torch.logsumexp(log_joint_t, dim=0)
        total_log_lik = total_log_lik + log_norm

        log_filt_t = log_joint_t - log_norm
        log_filtered[t] = log_filt_t

        log_var_bar_t = torch.logsumexp(log_filt_t + log_var_t, dim=0)
        log_var_bar[t] = log_var_bar_t

        sigma_bar_t = torch.exp(0.5 * log_var_bar_t)
        z_prev = returns[t] / sigma_bar_t
        log_var_bar_prev = log_var_bar_t
        log_filt_prev = log_filt_t

    return {
        "log_var": log_var,
        "log_filtered_prob": log_filtered,
        "log_var_bar": log_var_bar,
        "total_log_lik": total_log_lik,
        "nu": nu,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/regime/test_ms_egarch_recursion.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/regime/ms_egarch.py tests/regime/test_ms_egarch_recursion.py
git commit -m "feat: implement Gray's collapsing MS-EGARCH recursion with causal Hamilton filter"
```

---

## Task 14: MS-EGARCH log-joint, ADVI fit wrapper, and joint-draw sampling

**Files:**
- Modify: `src/raemf_mc/regime/ms_egarch.py`
- Test: `tests/regime/test_ms_egarch_fit.py`

**Interfaces:**
- Consumes: `MSEGARCHParamLayout, unpack_params, run_ms_egarch_recursion` (Tasks 12-13); `HierarchicalPriorConfig, state_shrinkage_weight, hierarchical_normal_log_prob` (Task 10); `AdviConfig, PooledPosterior, fit_multi_seed_advi, sample_joint_draw` (Tasks 8-9).
- Produces: `build_ms_egarch_log_joint(returns, init_log_var, init_log_state_prob, prior_config, layout=MSEGARCHParamLayout()) -> Callable[[Tensor], Tensor]`; `fit_ms_egarch(returns, advi_config, prior_config, seeds, device, layout=MSEGARCHParamLayout(), fallback_log_path="fallbacks.json") -> PooledPosterior`; `sample_ms_egarch_draw(posterior: PooledPosterior, layout=MSEGARCHParamLayout(), generator=None) -> MSEGARCHParams`.

- [ ] **Step 1: Write failing tests**

```python
# tests/regime/test_ms_egarch_fit.py
import torch
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    MSEGARCHParams,
    build_ms_egarch_log_joint,
    fit_ms_egarch,
    sample_ms_egarch_draw,
)
from raemf_mc.bayesian.torch_backend import AdviConfig, PooledPosterior
from raemf_mc.bayesian.priors import HierarchicalPriorConfig


def _small_returns(T=40, seed=0):
    torch.manual_seed(seed)
    return torch.randn(T) * 0.01


def test_build_log_joint_returns_finite_scalar_for_valid_theta():
    layout = MSEGARCHParamLayout()
    returns = _small_returns()
    n = layout.n_states
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 0.25))
    prior_config = HierarchicalPriorConfig()
    log_joint = build_ms_egarch_log_joint(
        returns, init_log_var, init_log_state_prob, prior_config, layout
    )
    theta = torch.zeros(layout.total)
    value = log_joint(theta)
    assert value.shape == ()
    assert torch.isfinite(value)


def test_fit_ms_egarch_runs_end_to_end_on_small_window(tmp_path):
    returns = _small_returns(T=30)
    layout = MSEGARCHParamLayout()
    advi_config = AdviConfig(n_steps=15, learning_rate=0.02, warmup_steps=2,
                               elbo_ma_window=5, early_stop_patience=15, n_mc_samples=2)
    prior_config = HierarchicalPriorConfig()
    posterior = fit_ms_egarch(
        returns, advi_config, prior_config, seeds=[0, 1],
        device=torch.device("cpu"), layout=layout,
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert isinstance(posterior, PooledPosterior)
    assert len(posterior.seed_results) == 2
    for r in posterior.seed_results:
        assert r.mu.shape == (layout.total,)
        assert torch.isfinite(r.mu).all()


def test_sample_ms_egarch_draw_returns_structured_params_with_fixed_generator(tmp_path):
    returns = _small_returns(T=25)
    layout = MSEGARCHParamLayout()
    advi_config = AdviConfig(n_steps=10, learning_rate=0.02, warmup_steps=1,
                               elbo_ma_window=3, early_stop_patience=10, n_mc_samples=2)
    prior_config = HierarchicalPriorConfig()
    posterior = fit_ms_egarch(
        returns, advi_config, prior_config, seeds=[0], device=torch.device("cpu"),
        layout=layout, fallback_log_path=tmp_path / "fallbacks.json",
    )
    gen1 = torch.Generator().manual_seed(7)
    gen2 = torch.Generator().manual_seed(7)
    draw1 = sample_ms_egarch_draw(posterior, layout, generator=gen1)
    draw2 = sample_ms_egarch_draw(posterior, layout, generator=gen2)
    assert isinstance(draw1, MSEGARCHParams)
    assert torch.equal(draw1.omega, draw2.omega)
    assert draw1.omega.shape == (4,)
    assert draw1.transition_logits.shape == (4, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/regime/test_ms_egarch_fit.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_ms_egarch_log_joint'`

- [ ] **Step 3: Append implementation to `ms_egarch.py`**

```python
# append to src/raemf_mc/regime/ms_egarch.py

from typing import Callable  # add to top-level imports in the real file

from raemf_mc.bayesian.priors import (
    HierarchicalPriorConfig,
    hierarchical_normal_log_prob,
    state_shrinkage_weight,
)
from raemf_mc.bayesian.torch_backend import (
    AdviConfig,
    PooledPosterior,
    fit_multi_seed_advi,
    sample_joint_draw,
)


def build_ms_egarch_log_joint(
    returns: torch.Tensor,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    prior_config: HierarchicalPriorConfig,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the MS-EGARCH log_joint(theta) callable consumed by the
    generic ADVI engine: log-likelihood from the Gray's-collapsing
    recursion plus a hierarchical shrinkage prior on the per-state EGARCH
    parameters (heavier shrinkage for states with fewer effective
    observations, e.g. Stress) and weakly-informative Normal(0,1) priors
    on the transition logits and nu.
    """

    def log_joint(theta: torch.Tensor) -> torch.Tensor:
        params = unpack_params(theta, layout)
        result = run_ms_egarch_recursion(returns, params, init_log_var, init_log_state_prob)
        log_lik = result["total_log_lik"]

        effective_obs = torch.exp(result["log_filtered_prob"]).sum(dim=0)
        weight = state_shrinkage_weight(effective_obs, prior_config.min_effective_observations)
        scale = torch.tensor(prior_config.hyper_mean_scale, dtype=theta.dtype)

        log_prior = torch.zeros((), dtype=theta.dtype, device=theta.device)
        for state_params in (params.omega, params.alpha, params.beta, params.gamma):
            hyper_mean = state_params.mean()
            log_prior = log_prior + hierarchical_normal_log_prob(
                state_params, hyper_mean, scale, weight
            )
        log_prior = log_prior + torch.distributions.Normal(0.0, 1.0).log_prob(
            params.transition_logits
        ).sum()
        log_prior = log_prior + torch.distributions.Normal(0.0, 1.0).log_prob(
            params.nu_raw
        ).sum()

        return log_lik + log_prior

    return log_joint


def fit_ms_egarch(
    returns: torch.Tensor,
    advi_config: AdviConfig,
    prior_config: HierarchicalPriorConfig,
    seeds: list[int],
    device: torch.device,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    fallback_log_path: str = "fallbacks.json",
) -> PooledPosterior:
    n = layout.n_states
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 1.0 / n))
    log_joint = build_ms_egarch_log_joint(
        returns, init_log_var, init_log_state_prob, prior_config, layout
    )
    init_mu = torch.zeros(layout.total)
    init_log_sigma = torch.full((layout.total,), -1.0)
    return fit_multi_seed_advi(
        log_joint, init_mu, init_log_sigma, advi_config, seeds, device, fallback_log_path
    )


def sample_ms_egarch_draw(
    posterior: PooledPosterior,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    generator: torch.Generator | None = None,
) -> MSEGARCHParams:
    theta = sample_joint_draw(posterior, generator=generator)
    return unpack_params(theta, layout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/regime/test_ms_egarch_fit.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/regime/ms_egarch.py tests/regime/test_ms_egarch_fit.py
git commit -m "feat: wire MS-EGARCH log-joint into ADVI fit and joint-draw sampling"
```

---

## Task 15: State alignment (Bull/Sideway/Bear/Stress)

**Files:**
- Create: `src/raemf_mc/regime/state_alignment.py`
- Test: `tests/regime/test_state_alignment.py`

**Interfaces:**
- Produces: `STATE_NAMES = ("Bull", "Sideway", "Bear", "Stress")`; `align_states(returns: Tensor, log_filtered_prob: Tensor) -> list[int]` (permutation: `permutation[i]` = raw state index assigned to `STATE_NAMES[i]`); `apply_alignment(log_filtered_prob: Tensor, permutation: list[int]) -> Tensor` (reorders columns).

- [ ] **Step 1: Write failing tests**

```python
# tests/regime/test_state_alignment.py
import torch
from raemf_mc.regime.state_alignment import STATE_NAMES, align_states, apply_alignment


def test_align_states_orders_by_mean_return_and_volatility():
    # 4 raw states with clearly separated economic character:
    # state 0: high positive mean, low vol -> Bull
    # state 1: near-zero mean, low vol -> Sideway
    # state 2: negative mean, moderate vol -> Bear
    # state 3: very negative mean, high vol -> Stress
    torch.manual_seed(0)
    T = 400
    raw_state_returns = {
        0: torch.randn(T) * 0.003 + 0.0015,
        1: torch.randn(T) * 0.004 + 0.0001,
        2: torch.randn(T) * 0.01 - 0.003,
        3: torch.randn(T) * 0.03 - 0.01,
    }
    # build a returns series and a hard filtered-prob assignment that
    # spends ~T/4 timesteps in each raw state, in a fixed known order
    returns = torch.cat([raw_state_returns[k] for k in range(4)])
    log_filtered_prob = torch.full((4 * T, 4), -30.0)
    for k in range(4):
        log_filtered_prob[k * T : (k + 1) * T, k] = 0.0

    permutation = align_states(returns, log_filtered_prob)
    assert sorted(permutation) == [0, 1, 2, 3]
    assert permutation[STATE_NAMES.index("Bull")] == 0
    assert permutation[STATE_NAMES.index("Sideway")] == 1
    assert permutation[STATE_NAMES.index("Bear")] == 2
    assert permutation[STATE_NAMES.index("Stress")] == 3


def test_apply_alignment_reorders_columns():
    log_filtered_prob = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).log()
    permutation = [2, 0, 3, 1]
    reordered = apply_alignment(log_filtered_prob, permutation)
    expected = torch.tensor([[0.3, 0.1, 0.4, 0.2]]).log()
    assert torch.allclose(reordered, expected, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/regime/test_state_alignment.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/regime/state_alignment.py
from __future__ import annotations

import torch

STATE_NAMES = ("Bull", "Sideway", "Bear", "Stress")


def align_states(returns: torch.Tensor, log_filtered_prob: torch.Tensor) -> list[int]:
    """Compute a fixed permutation mapping raw latent states to economic
    labels (Bull/Sideway/Bear/Stress), fit on train data only. Ranks raw
    states by a composite score: mean return (favors Bull), penalized by
    volatility (favors Stress at the low end) — implemented as a single
    sort key `mean_return - volatility`, which is high for Bull (high
    mean, low vol) and lowest for Stress (very negative mean, high vol).
    """
    probs = torch.exp(log_filtered_prob)
    n_states = probs.shape[1]
    weights = probs / probs.sum(dim=0, keepdim=True)  # (T, n_states)

    mean_return = (weights * returns.unsqueeze(1)).sum(dim=0)
    mean_sq = (weights * (returns.unsqueeze(1) ** 2)).sum(dim=0)
    variance = torch.clamp(mean_sq - mean_return**2, min=0.0)
    volatility = torch.sqrt(variance)

    score = mean_return - volatility
    order = torch.argsort(score, descending=True)  # best (Bull) first
    return [int(order[i]) for i in range(n_states)]


def apply_alignment(log_filtered_prob: torch.Tensor, permutation: list[int]) -> torch.Tensor:
    """Reorder columns of log_filtered_prob so column i corresponds to
    STATE_NAMES[i], using a permutation produced by align_states."""
    idx = torch.tensor(permutation, dtype=torch.long)
    return log_filtered_prob[:, idx]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/regime/test_state_alignment.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/regime/state_alignment.py tests/regime/test_state_alignment.py
git commit -m "feat: add train-only economic state alignment for MS-EGARCH regimes"
```

---

## Task 16: Config profiles

**Files:**
- Create: `configs/cpu_smoke.yaml`
- Create: `configs/gpu_research.yaml`

**Interfaces:**
- Produces: two YAML files consumed directly (via `yaml.safe_load`) by Task 18's integration test — no config-loader module in this sub-project's scope.

- [ ] **Step 1: Write `configs/cpu_smoke.yaml`**

```yaml
# Profile debug nhanh tren CPU: cua so du lieu ngan, it buoc ADVI, 1 seed.
# Dung de kiem tra dung/sai truoc khi chay full-scale.
device_preference: auto
window_sessions: 500        # chi dung 500 phien gan nhat de lap nhanh
seeds: [0]
advi:
  n_steps: 300
  learning_rate: 0.05
  warmup_steps: 20
  grad_clip_norm: 10.0
  elbo_ma_window: 20
  early_stop_patience: 100
  min_delta: 0.001
  retry_lr_factor: 0.5
  max_retries: 3
  n_mc_samples: 4
prior:
  hyper_mean_scale: 1.0
  min_effective_observations: 30.0
```

- [ ] **Step 2: Write `configs/gpu_research.yaml`**

```yaml
# Profile scale-up nhieu seed/nhieu buoc ADVI cho may co CUDA.
# Chua chay - chuan bi san de dung khi chuyen sang may co GPU.
device_preference: auto
window_sessions: null        # dung toan bo chuoi du lieu da lam sach
seeds: [0, 1, 2, 3, 4, 5, 6, 7]
advi:
  n_steps: 20000
  learning_rate: 0.01
  warmup_steps: 500
  grad_clip_norm: 10.0
  elbo_ma_window: 200
  early_stop_patience: 2000
  min_delta: 0.0001
  retry_lr_factor: 0.5
  max_retries: 3
  n_mc_samples: 8
prior:
  hyper_mean_scale: 1.0
  min_effective_observations: 30.0
```

- [ ] **Step 3: Commit**

```bash
git add configs/cpu_smoke.yaml configs/gpu_research.yaml
git commit -m "chore: add cpu_smoke and gpu_research ADVI config profiles"
```

---

## Task 17: Design decisions document

**Files:**
- Create: `docs/ms_egarch_design_decisions.md`

- [ ] **Step 1: Write the document**

```markdown
# Quyet dinh thiet ke MS-EGARCH (a-d)

Ngay: 2026-08-17
Tham chieu: docs/superpowers/specs/2026-08-17-ms-egarch-foundation-design.md

Tai lieu nay ghi lai 4 quyet dinh thiet ke con mo, kem ly do va danh doi da
can nhac. Day la lua chon co lap luan rieng, khong phai ket luan da duoc
kiem chung boi tai lieu hoc thuat chuan muc cho truong hop EGARCH cu the -
se can bang chung OOS truoc khi khang dinh mo hinh tot hon.

## (a) Gop o level-space, khong phai log-space

**Lua chon:** gop tai `torch.logsumexp(log_filtered_prob + log_var)`, tuong
duong voi gop tren level-space (trung binh co trong so cua sigma^2 theo
tung trang thai) roi lay log, khong phai trung binh truc tiep tren
log-variance.

**Ly do:** gop level-space bao toan dung mot moment that cua phan phoi tron
- `E[sigma_t^2 | F_{t-1}] = sum_k P_k * sigma_k,t^2` theo dinh ly phuong
sai toan phan. Gop truc tiep tren log-variance khong phai mot moment that,
va do bat dang thuc Jensen (log la ham lom) se luon thien lech xuong so
voi gia tri dung - sai lech cang lon khi xac suat filtered cang phan tan
giua cac trang thai co bien dong chenh lech nhau nhieu, dung luc can phan
biet regime ro nhat. Ve tinh toan, `torch.logsumexp` cho ra chinh xac ket
qua nay ma khong can exponentiate tuong minh, on dinh so hoc hon.

**Danh doi:** khong co tai lieu hoc thuat chuan muc xac nhan lua chon nay
cho truong hop EGARCH cu the (Gray's paper goc la cho GARCH thuong, tren
level-space thuan, khong co de quy log). Da viet test suy bien K=1
(`test_degenerate_single_state_matches_reference_egarch`) de xac nhan
khong co sai khac khi khong con bat dinh giua cac trang thai.

## (b) Chuan hoa z[t-1] bang sigma_bar[t-1] da gop

**Lua chon:** `z[t-1] = eps[t-1] / sigma_bar[t-1]`, voi sigma_bar[t-1] lay
tu chinh gia tri gop o quyet dinh (a).

**Ly do:** `eps[t-1]` la mot so quan sat duoc duy nhat (da thuc su xay ra),
can mot sigma tham chieu duy nhat de chuan hoa. sigma_bar gop la lua chon
tu nhien nhat vi day chinh la gia tri duoc dung lam dau vao cho de quy
log-variance cua buoc ke tiep, giu toan bo he nhat quan noi bo.

**Danh doi:** Gray's paper goc khong co so hang bat doi xung nen khong co
tien le truc tiep cho lua chon nay trong boi canh multi-regime.

## (c) Bo c_k o tang scenario-return

**Lua chon:** khong giu he so nhan `c_k` rieng theo regime o tang
scenario-return (se trien khai o sub-project 3) - chi giu mu_k va nu_k,
vi MS-EGARCH da cung cap sigma_k,t rieng theo tung regime.

**Ly do:** giu them c_k lam he so nhan tu do khac len tren co nguy co
non-identifiability - hai tham so cung giai thich mot scale, ELBO co song
nui phang doc theo huong c_k * sigma, posterior mean-field co the trong
"tu tin" gia tao o tung bien trong khi thuc chat khong dinh danh duoc.

**Danh doi:** neu sau nay phat hien phan tan trong-regime khong duoc giai
thich het boi de quy EGARCH (vi du qua phan tich residual thuc te), co the
can them lai mot tham so scale bo sung - nhung chua co bang chung thuc
nghiem cho dieu nay tai thoi diem viet tai lieu, nen chua them.

## (d) Hai giai doan Bayesian tach biet, khong mot ELBO chung

**Lua chon:** fit posterior MS-EGARCH doc lap truoc (khong can scenario/
return labels); tang scenario-return (sub-project 3) se dieu kien theo
posterior draws cua MS-EGARCH thong qua `sample_ms_egarch_draw`.

**Ly do:** day la mot dang modular/"cut" Bayesian inference - cat duong
phan hoi nguoc tu mot mo-dun co the bi misspecify (tang scenario) sang
mo-dun truoc no (MS-EGARCH), mot ky thuat co co so thong ke chu khong chi
vi tien trien khai. Cung khop tu nhien voi viec tach sub-project 2 va
sub-project 3 theo quan ly pham vi da thong nhat, giup moi mo-dun test doc
lap duoc.

**Danh doi:** khong lan truyen bat dinh cua mu_k/nu_k nguoc lai vao fit
MS-EGARCH (dieu kien mot chieu) - chap nhan duoc vi day la xap xi pho bien
trong modular Bayes, va la doi lay duoc su don gian/de debug ro rang.

## He qua interface

`sample_ms_egarch_draw(posterior, layout, generator)` tra ve dung mot
`MSEGARCHParams` cho moi lan goi - noi dung ma sub-project 3 se dung de lay
"mot joint draw duy nhat cho moi Monte Carlo path, giu co dinh suot
horizon" (khong resample ngam trong vong lap simulate).
```

- [ ] **Step 2: Commit**

```bash
git add docs/ms_egarch_design_decisions.md
git commit -m "docs: record MS-EGARCH open design decisions (a-d) with rationale"
```

---

## Task 18: End-to-end integration smoke test

**Files:**
- Test: `tests/test_integration_smoke.py`

**Interfaces:**
- Consumes: `load_vnindex_ohlcv` (Task 4), `compute_log_returns` (Task 5), `fit_ms_egarch`, `sample_ms_egarch_draw`, `MSEGARCHParamLayout` (Task 14), `align_states`, `apply_alignment`, `STATE_NAMES` (Task 15), `AdviConfig` (Task 8), `HierarchicalPriorConfig` (Task 10), `configs/cpu_smoke.yaml` (Task 16).

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration_smoke.py
from pathlib import Path

import torch
import yaml

from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, fit_ms_egarch, sample_ms_egarch_draw
from raemf_mc.regime.state_alignment import STATE_NAMES, align_states, apply_alignment
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.bayesian.priors import HierarchicalPriorConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cpu_smoke.yaml"


def test_full_pipeline_smoke_on_real_data(tmp_path):
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ohlcv = load_vnindex_ohlcv()
    log_returns = compute_log_returns(ohlcv)
    window = log_returns.iloc[-config["window_sessions"] :]
    returns = torch.tensor(window.to_numpy(), dtype=torch.float32)
    returns = returns - returns.mean()

    advi_config = AdviConfig(**config["advi"])
    prior_config = HierarchicalPriorConfig(**config["prior"])
    layout = MSEGARCHParamLayout()

    posterior = fit_ms_egarch(
        returns, advi_config, prior_config, seeds=config["seeds"],
        device=torch.device("cpu"), layout=layout,
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert len(posterior.seed_results) == len(config["seeds"])
    for r in posterior.seed_results:
        assert torch.isfinite(r.mu).all()
        assert torch.isfinite(r.log_sigma).all()

    draw = sample_ms_egarch_draw(posterior, layout)
    assert draw.omega.shape == (4,)

    from raemf_mc.regime.ms_egarch import run_ms_egarch_recursion

    n = layout.n_states
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 1.0 / n))
    result = run_ms_egarch_recursion(returns, draw, init_log_var, init_log_state_prob)
    assert torch.isfinite(result["log_filtered_prob"]).all()

    permutation = align_states(returns, result["log_filtered_prob"])
    aligned = apply_alignment(result["log_filtered_prob"], permutation)
    assert aligned.shape == result["log_filtered_prob"].shape
    assert len(STATE_NAMES) == 4
```

- [ ] **Step 2: Run test to verify it fails first (before this task existed it can't fail for the right reason — instead run it directly and confirm it currently errors only if any prior task is broken)**

Run: `python -m pytest tests/test_integration_smoke.py -v -s`
Expected: either PASS directly (since all dependencies already exist from Tasks 1-17), or a clear failure pointing at a real integration bug to fix before proceeding — this is the first test in the plan allowed to fail for a substantive reason rather than "module not found".

- [ ] **Step 3: Fix any integration issues found, then re-run until it passes**

Run: `python -m pytest tests/test_integration_smoke.py -v -s`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_smoke.py
git commit -m "test: add end-to-end smoke test fitting MS-EGARCH on real VN-Index data"
```

---

## Task 19: Full suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass, 0 failures

- [ ] **Step 2: Run lint**

Run: `python -m ruff check src tests scripts`
Expected: no errors (fix any reported issues, re-run until clean)

- [ ] **Step 3: Confirm no float16 usage anywhere in bayesian/regime code**

Run: `python -c "import pathlib; import re; [print(p) for p in pathlib.Path('src/raemf_mc/bayesian').rglob('*.py') if 'float16' in p.read_text()]; [print(p) for p in pathlib.Path('src/raemf_mc/regime').rglob('*.py') if 'float16' in p.read_text()]"`
Expected: no output (float16 only referenced inside the guard function's own error message check, if at all)

- [ ] **Step 4: Final commit if any lint fixes were needed**

```bash
git add -A
git commit -m "chore: fix lint issues found in full-suite verification"
```

(Skip this commit if step 2 was already clean with no changes.)
