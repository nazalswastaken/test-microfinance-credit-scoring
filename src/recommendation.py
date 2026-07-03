"""Loan recommendation.

Combines the two model outputs into a decision:

  - the conservative (stress) loan capacity - how much payment will realistically
    flow through even in a low season, and
  - the heuristic risk score - how trustworthy that flow looks.

The bank can only deduct part of each weekly payment, so the most it can recover
over the loan is DEDUCTION_RATE * capacity. We lend that, and refuse anyone whose
score is in the "High" band. Kept separate from the models so the scoring can be
swapped out without touching the policy.
"""

from __future__ import annotations

import pandas as pd

from .utils import BAND_LOW_MIN, BAND_MEDIUM_MIN, DEDUCTION_RATE


def risk_band(score: float) -> str:
    if score >= BAND_LOW_MIN:
        return "Low"
    if score >= BAND_MEDIUM_MIN:
        return "Medium"
    return "High"


def max_repayable(capacity: float, deduction_rate: float = DEDUCTION_RATE) -> float:
    """Most the bank can claw back over the loan given the payment capacity."""
    return max(0.0, capacity) * deduction_rate


def recommend(
    scores: pd.Series,
    capacity: pd.Series,
    deduction_rate: float = DEDUCTION_RATE,
) -> pd.DataFrame:
    """Build the recommendation table.

    scores: risk score per farmer (0..1).
    capacity: conservative 13-week payment capacity per farmer.

    Returns a table with the risk band, the capacity used, and the recommended
    loan (0 = reject). "High" band farmers are rejected regardless of capacity,
    and so is anyone whose capacity rounds to nothing.
    """
    df = pd.DataFrame({"risk_score": scores, "loan_capacity": capacity})
    df["risk_band"] = df["risk_score"].map(risk_band)

    repayable = df["loan_capacity"].map(lambda c: max_repayable(c, deduction_rate))
    approved = df["risk_band"] != "High"
    df["recommended_loan"] = (repayable * approved).round(0)
    df["decision"] = approved.map({True: "Approve", False: "Reject"})
    tiny = df["recommended_loan"] <= 0
    df.loc[tiny, ["decision", "recommended_loan"]] = ["Reject", 0.0]

    cols = ["risk_score", "risk_band", "loan_capacity", "recommended_loan", "decision"]
    return df[cols].sort_values("risk_score", ascending=False)
