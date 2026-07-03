"""Tests for the Phase 1 data loader."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DataValidationError, load_payments  # noqa: E402
from src.utils import ID_COLUMN, N_WEEKS, WEEK_COLUMNS  # noqa: E402


def _write_csv(tmp_path: Path, rows: list[str], header: str | None = None) -> Path:
    if header is None:
        weeks = ",".join(f"Week{i}" for i in range(1, N_WEEKS + 1))
        header = f"Farmer No.,{weeks}"
    path = tmp_path / "payments.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def _row(farmer_no: int, fill: str = "100") -> str:
    return ",".join([str(farmer_no), *([fill] * N_WEEKS)])


def test_loads_and_standardizes(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [_row(1), _row(2)])
    df = load_payments(path)
    assert df.index.name == ID_COLUMN
    assert list(df.columns) == WEEK_COLUMNS
    assert len(df) == 2


def test_parses_dash_as_nan_and_strips_separators(tmp_path: Path) -> None:
    weeks = ",".join(f"Week{i}" for i in range(1, N_WEEKS + 1))
    header = f"Farmer No.,{weeks}"
    values = ['"1,034"', "-   "] + ["500"] * (N_WEEKS - 2)
    row = ",".join(["1", *values])
    path = tmp_path / "payments.csv"
    path.write_text(header + "\n" + row + "\n")

    df, report = load_payments(path, return_report=True)
    assert df.loc[1, "week1"] == 1034.0
    assert pd.isna(df.loc[1, "week2"])
    assert report.n_missing_cells == 1


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [_row(1), _row(1)])
    with pytest.raises(DataValidationError):
        load_payments(path)


def test_rejects_wrong_week_count(tmp_path: Path) -> None:
    header = "Farmer No.,Week1,Week2"
    path = _write_csv(tmp_path, ["1,100,200"], header=header)
    with pytest.raises(DataValidationError):
        load_payments(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_payments(tmp_path / "does_not_exist.csv")
