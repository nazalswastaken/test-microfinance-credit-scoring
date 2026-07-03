"""Shared helpers: project paths and small utilities used across modules."""

from __future__ import annotations

from pathlib import Path

# Anchor all paths to the repository root so modules and notebooks resolve the
# same locations regardless of the current working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Canonical location of the input dataset.
RAW_PAYMENTS_PATH: Path = RAW_DATA_DIR / "payments.csv"

# The dataset always covers a fixed 52-week history, and loans run for 13 weeks.
N_WEEKS: int = 52
LOAN_WEEKS: int = 13

# Standardized column names after loading.
ID_COLUMN: str = "farmer_no"
WEEK_COLUMNS: list[str] = [f"week{i}" for i in range(1, N_WEEKS + 1)]

# Lending policy knobs. The bank can only claw back part of each weekly payment,
# so a loan is repayable out of DEDUCTION_RATE * (sum of the next 13 payments).
# We size loans off a pessimistic percentile of the simulated payment stream.
DEDUCTION_RATE: float = 0.40
DOWNSIDE_PERCENTILE: int = 10
N_SIMULATIONS: int = 5000

# Risk score cutoffs (score runs 0..1, higher = safer).
BAND_LOW_MIN: float = 0.60      # >= this -> "Low" risk
BAND_MEDIUM_MIN: float = 0.35   # >= this -> "Medium"; below -> "High" (reject)


def week_columns(n: int) -> list[str]:
    """Column names for the first ``n`` weeks (``week1..week{n}``)."""
    return [f"week{i}" for i in range(1, n + 1)]


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
