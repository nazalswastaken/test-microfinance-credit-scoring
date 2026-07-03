"""Scoring entry point.

load model -> score farmers -> recommend loan amounts -> export csv

    python scripts/score.py

Comes together once the modeling and recommendation code exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (``python scripts/score.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # TODO(phase 4+): load model -> score -> recommend loans -> export CSV.
    raise SystemExit("score.py is not implemented yet (see project roadmap).")


if __name__ == "__main__":
    main()
