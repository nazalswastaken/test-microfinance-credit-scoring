"""Training entry point.

load data -> preprocess -> features -> train -> save model

    python scripts/train.py

Only the load step is wired up so far; the rest gets filled in as I go.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (``python scripts/train.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_payments  # noqa: E402


def main() -> None:
    df, report = load_payments(return_report=True)
    print("Loaded data:", report.summary())
    # TODO(phase 2+): preprocess -> features -> train -> save model.


if __name__ == "__main__":
    main()
