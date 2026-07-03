"""Scoring entry point.

load model -> features -> score -> simulate capacity -> recommend loans -> export csv

    python scripts/score.py

Writes reports/loan_recommendations.csv with a risk band and recommended loan
amount for every farmer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_payments  # noqa: E402
from src.features import build_features  # noqa: E402
from src.models import HeuristicRiskModel, stress_capacity_table  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.recommendation import recommend  # noqa: E402
from src.utils import MODELS_DIR, REPORTS_DIR, WEEK_COLUMNS, ensure_dir  # noqa: E402


def main() -> None:
    model_path = MODELS_DIR / "risk_model.json"
    if not model_path.exists():
        raise SystemExit("no trained model found. run scripts/train.py first.")
    model = HeuristicRiskModel.load(model_path)

    df = preprocess(load_payments())
    features = build_features(df)
    scores = model.score(features)

    capacity = stress_capacity_table(df, WEEK_COLUMNS)
    recs = recommend(scores, capacity)

    out = ensure_dir(REPORTS_DIR) / "loan_recommendations.csv"
    recs.to_csv(out)
    approved = (recs["decision"] == "Approve").sum()
    print(f"scored {len(recs)} farmers, approved {approved}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
