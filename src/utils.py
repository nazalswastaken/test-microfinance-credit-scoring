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


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
