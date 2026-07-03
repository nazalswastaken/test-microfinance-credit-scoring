"""Loading and validating the raw payments CSV.

Just gets the CSV into a clean DataFrame -- no imputation or feature work here,
that comes later. It standardizes the column names, checks there are exactly 52
week columns, converts the payments to numbers (the file uses "-" for skipped
weeks and thousands separators like "1,034"), flags missing values, and refuses
to load if farmer IDs are duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Let this file be run directly (python src/data.py) as a quick check, while
# still importing cleanly as part of the package. Run directly there's no parent
# package for the "from .utils" import, so we put the repo root on the path and
# name the package ourselves.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .utils import ID_COLUMN, N_WEEKS, RAW_PAYMENTS_PATH, WEEK_COLUMNS

# Placeholder tokens that represent "no payment recorded" in the raw file.
# These are parsed to NaN; whether they mean zero or unknown is decided in
# preprocessing, not here.
_MISSING_TOKENS = {"-", "", "nan", "na", "n/a", "none", "null"}


class DataValidationError(ValueError):
    """Raised when the raw data violates an expected schema invariant."""


@dataclass
class DataQualityReport:
    """Summary of data-quality checks performed during loading."""

    n_farmers: int
    n_week_columns: int
    n_missing_cells: int
    missing_by_week: dict[str, int] = field(default_factory=dict)
    duplicate_ids: list = field(default_factory=list)

    @property
    def missing_fraction(self) -> float:
        """Fraction of payment cells that are missing."""
        total = self.n_farmers * self.n_week_columns
        return self.n_missing_cells / total if total else 0.0

    def summary(self) -> str:
        return (
            f"{self.n_farmers} farmers x {self.n_week_columns} weeks | "
            f"missing cells: {self.n_missing_cells} "
            f"({self.missing_fraction:.1%}) | "
            f"duplicate ids: {len(self.duplicate_ids)}"
        )


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to canonical snake_case names.

    Maps the id column (e.g. ``"Farmer No."``) to ``farmer_no`` and each
    ``Week<N>`` column to ``week<N>``.
    """
    rename: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        if key.startswith("farmer"):
            rename[col] = ID_COLUMN
        elif key.startswith("week"):
            # Normalize "Week 1" / "Week1" -> "week1".
            digits = "".join(ch for ch in key if ch.isdigit())
            if digits:
                rename[col] = f"week{int(digits)}"
    return df.rename(columns=rename)


def _coerce_payment(series: pd.Series) -> pd.Series:
    """Coerce a raw payment column to float.

    Handles values stored as strings with thousands separators (``"1,034"``)
    and the ``-`` placeholder used for skipped weeks (parsed to NaN).
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
    )
    # Map known missing tokens to NA before numeric coercion.
    cleaned = cleaned.where(~cleaned.str.lower().isin(_MISSING_TOKENS))
    return pd.to_numeric(cleaned, errors="coerce")


def _validate_week_columns(df: pd.DataFrame) -> None:
    present = [c for c in WEEK_COLUMNS if c in df.columns]
    missing = [c for c in WEEK_COLUMNS if c not in df.columns]
    extra = [
        c
        for c in df.columns
        if c.startswith("week") and c not in WEEK_COLUMNS
    ]
    if len(present) != N_WEEKS or missing or extra:
        raise DataValidationError(
            f"Expected exactly {N_WEEKS} week columns (week1..week{N_WEEKS}). "
            f"Missing: {missing or 'none'}; unexpected: {extra or 'none'}."
        )


def _validate_ids(df: pd.DataFrame) -> list:
    if ID_COLUMN not in df.columns:
        raise DataValidationError(f"Missing required id column '{ID_COLUMN}'.")
    duplicates = df[ID_COLUMN][df[ID_COLUMN].duplicated()].unique().tolist()
    if duplicates:
        raise DataValidationError(
            f"Found {len(duplicates)} duplicate farmer id(s): "
            f"{duplicates[:10]}{'...' if len(duplicates) > 10 else ''}."
        )
    return duplicates


def load_payments(
    path: str | Path = RAW_PAYMENTS_PATH,
    *,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, DataQualityReport]:
    """Load and validate the raw weekly-payments CSV.

    Parameters
    ----------
    path:
        Path to the CSV. Defaults to ``data/raw/payments.csv``.
    return_report:
        If True, also return a :class:`DataQualityReport`.

    Returns
    -------
    A clean DataFrame indexed by ``farmer_no`` with 52 numeric ``week*`` columns
    (missing weeks as NaN). Optionally the accompanying quality report.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    DataValidationError
        If the schema is invalid or farmer ids are duplicated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Payments file not found at '{path}'. "
            "Place the dataset at data/raw/payments.csv (see data/README.md)."
        )

    # Read everything as strings so we control numeric parsing ourselves.
    raw = pd.read_csv(path, dtype=str)
    df = _standardize_columns(raw)

    _validate_week_columns(df)
    duplicates = _validate_ids(df)

    df[ID_COLUMN] = pd.to_numeric(df[ID_COLUMN], errors="coerce").astype("Int64")
    for col in WEEK_COLUMNS:
        df[col] = _coerce_payment(df[col])

    df = df[[ID_COLUMN, *WEEK_COLUMNS]].set_index(ID_COLUMN).sort_index()

    missing_by_week = {c: int(df[c].isna().sum()) for c in WEEK_COLUMNS}
    report = DataQualityReport(
        n_farmers=len(df),
        n_week_columns=N_WEEKS,
        n_missing_cells=int(sum(missing_by_week.values())),
        missing_by_week=missing_by_week,
        duplicate_ids=duplicates,
    )

    if return_report:
        return df, report
    return df


if __name__ == "__main__":
    # Quick sanity check: load the real dataset and print the quality report.
    _df, _report = load_payments(return_report=True)
    print(_report.summary())
    print(_df.iloc[:3, :5])
