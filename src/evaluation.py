"""Evaluation via backtesting.

We have 52 weeks per farmer and loans last 13, which gives a natural test: pretend
it's the end of week 39, make the lending decision using only weeks 1-39, then use
the real weeks 40-52 as the "future" and check whether the loan would actually
have been repaid.

A loan is repaid if the amount lent is no more than what the bank could deduct
from the real next 13 weeks (DEDUCTION_RATE * their actual total). From that we get
repayment success, recovery, portfolio loss, utilization, and whether the risk
score actually ranked farmers sensibly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features
from .models import HeuristicRiskModel, stress_capacity_table
from .recommendation import recommend
from .utils import DEDUCTION_RATE, LOAN_WEEKS, N_WEEKS, week_columns


def train_test_split_weeks(history_weeks: int = N_WEEKS - LOAN_WEEKS) -> tuple[list[str], list[str]]:
    """Column names for the decision window and the held-out future window."""
    history = week_columns(history_weeks)
    future = [f"week{i}" for i in range(history_weeks + 1, N_WEEKS + 1)]
    return history, future


def run_recommendation(
    df: pd.DataFrame,
    week_cols: list[str],
    model: HeuristicRiskModel | None = None,
) -> tuple[pd.DataFrame, HeuristicRiskModel]:
    """Features -> fit/score -> stress capacity -> recommend, on ``week_cols``."""
    features = build_features(df, week_cols=week_cols)
    if model is None:
        model = HeuristicRiskModel().fit(features)
    scores = model.score(features)
    capacity = stress_capacity_table(df, week_cols)
    recs = recommend(scores, capacity)
    return recs, model


def backtest(
    df: pd.DataFrame,
    deduction_rate: float = DEDUCTION_RATE,
) -> tuple[pd.DataFrame, dict]:
    """Decide on weeks 1..39, score against real weeks 40..52.

    Returns the per-farmer table (recommendation joined with the realized
    outcome) and a dict of portfolio metrics.
    """
    history_cols, future_cols = train_test_split_weeks()
    recs, _ = run_recommendation(df, history_cols)

    # What actually happened in the held-out 13 weeks.
    actual_total = df[future_cols].to_numpy(dtype=float).sum(axis=1)
    actual = pd.Series(actual_total, index=df.index, name="actual_future_total")
    recs = recs.join(actual)
    recs["actual_repayable"] = (recs["actual_future_total"] * deduction_rate).round(0)
    recs["recovered"] = np.minimum(recs["recommended_loan"], recs["actual_repayable"])
    recs["shortfall"] = (recs["recommended_loan"] - recs["actual_repayable"]).clip(lower=0)

    approved = recs["decision"] == "Approve"
    n_approved = int(approved.sum())
    total_loan = float(recs.loc[approved, "recommended_loan"].sum())

    metrics = {
        "n_farmers": int(len(recs)),
        "n_approved": n_approved,
        "approval_rate": n_approved / len(recs) if len(recs) else 0.0,
        # fraction of approved loans fully covered by the real next 13 weeks
        "repayment_success_rate": float((recs.loc[approved, "shortfall"] <= 0).mean())
        if n_approved
        else 0.0,
        "total_loan_book": total_loan,
        "total_recovered": float(recs.loc[approved, "recovered"].sum()),
        "recovery_rate": float(recs.loc[approved, "recovered"].sum() / total_loan)
        if total_loan
        else 0.0,
        "expected_loss_rate": float(recs.loc[approved, "shortfall"].sum() / total_loan)
        if total_loan
        else 0.0,
        # does a higher score line up with a bigger real future stream?
        "score_capacity_spearman": float(
            recs["risk_score"].corr(recs["actual_future_total"], method="spearman")
        ),
    }
    return recs, metrics


def rolling_backtest(
    df: pd.DataFrame,
    starts: range | list[int] | None = None,
    deduction_rate: float = DEDUCTION_RATE,
) -> pd.DataFrame:
    """Backtest at several issue dates, not just the one before the trough.

    For each ``start`` (the 1-based week the loan begins), decide using every
    week before it and score against the 13 weeks from ``start``. Returns one row
    of metrics per issue week, which makes the seasonal timing effect obvious:
    loans issued into a rising season repay far better than loans issued into the
    trough.
    """
    if starts is None:
        starts = range(LOAN_WEEKS + 1, N_WEEKS - LOAN_WEEKS + 2)

    rows = []
    for start in starts:
        history_cols = week_columns(start - 1)
        future_cols = [f"week{i}" for i in range(start, start + LOAN_WEEKS)]
        if len(future_cols) < LOAN_WEEKS or f"week{start + LOAN_WEEKS - 1}" not in df.columns:
            continue

        recs, _ = run_recommendation(df, history_cols)
        actual = df[future_cols].to_numpy(dtype=float).sum(axis=1)
        approved = recs["decision"] == "Approve"
        loan = recs.loc[approved, "recommended_loan"]
        repayable = pd.Series(actual, index=df.index)[approved] * deduction_rate
        shortfall = (loan - repayable).clip(lower=0)

        rows.append(
            {
                "issue_week": start,
                "approval_rate": float(approved.mean()),
                "repayment_success_rate": float((shortfall <= 0).mean()) if len(loan) else 0.0,
                "expected_loss_rate": float(shortfall.sum() / loan.sum()) if loan.sum() else 0.0,
                "total_loan_book": float(loan.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("issue_week")


def success_by_band(recs: pd.DataFrame) -> pd.DataFrame:
    """Repayment success and average loan broken down by risk band."""
    approved = recs[recs["decision"] == "Approve"]
    if approved.empty:
        return pd.DataFrame()
    grouped = approved.groupby("risk_band")
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "repayment_success_rate": grouped.apply(
                lambda g: (g["shortfall"] <= 0).mean(), include_groups=False
            ),
            "avg_loan": grouped["recommended_loan"].mean(),
        }
    )
