"""Training entry point.

load data -> preprocess -> features -> fit risk model -> save model

    python scripts/train.py

The "model" is the heuristic scorer, which just needs the population feature
ranges it normalizes against. We save those to models/ so scoring is reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_payments  # noqa: E402
from src.features import build_features  # noqa: E402
from src.models import HeuristicRiskModel  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.utils import MODELS_DIR, ensure_dir  # noqa: E402


def main() -> None:
    df, report = load_payments(return_report=True)
    print("loaded:", report.summary())

    clean = preprocess(df)
    features = build_features(clean)
    print(f"built {features.shape[1]} features for {features.shape[0]} farmers")

    model = HeuristicRiskModel().fit(features)
    out = ensure_dir(MODELS_DIR) / "risk_model.json"
    model.save(out)
    print(f"saved model -> {out}")


if __name__ == "__main__":
    main()
