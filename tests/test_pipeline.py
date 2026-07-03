"""Tests for features, model, simulation, recommendation and backtest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_FUNCTIONS, build_features  # noqa: E402
from src.evaluation import backtest, train_test_split_weeks  # noqa: E402
from src.models import (  # noqa: E402
    HeuristicRiskModel,
    capacity_percentile,
    simulate_totals,
    stress_capacity_table,
)
from src.preprocessing import preprocess  # noqa: E402
from src.recommendation import recommend, risk_band  # noqa: E402
from src.utils import DOWNSIDE_PERCENTILE, LOAN_WEEKS, N_WEEKS, WEEK_COLUMNS  # noqa: E402


def _synthetic(n: int = 60, seed: int = 1) -> pd.DataFrame:
    """A mix of steady, volatile and quitting farmers over 52 weeks."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        kind = i % 3
        if kind == 0:  # steady
            x = rng.normal(10_000, 800, N_WEEKS).clip(0)
        elif kind == 1:  # volatile with skips
            x = rng.normal(9_000, 6_000, N_WEEKS).clip(0)
            x[rng.random(N_WEEKS) < 0.35] = 0
        else:  # tails off to nothing
            x = np.linspace(12_000, 0, N_WEEKS) + rng.normal(0, 500, N_WEEKS)
            x = x.clip(0)
        rows.append(x)
    df = pd.DataFrame(rows, columns=WEEK_COLUMNS)
    df.index = pd.RangeIndex(1000, 1000 + n, name="farmer_no")
    return df


def test_build_features_shape_and_finiteness() -> None:
    df = preprocess(_synthetic())
    feats = build_features(df)
    assert feats.shape == (len(df), len(FEATURE_FUNCTIONS))
    assert np.isfinite(feats.to_numpy()).all()


def test_features_work_on_short_window() -> None:
    df = preprocess(_synthetic())
    feats = build_features(df, week_cols=WEEK_COLUMNS[:39])
    assert len(feats) == len(df)
    assert np.isfinite(feats.to_numpy()).all()


def test_model_score_range_and_roundtrip(tmp_path: Path) -> None:
    df = preprocess(_synthetic())
    feats = build_features(df)
    model = HeuristicRiskModel().fit(feats)
    scores = model.score(feats)
    assert scores.between(0, 1).all()

    path = tmp_path / "m.json"
    model.save(path)
    reloaded = HeuristicRiskModel.load(path)
    pd.testing.assert_series_equal(scores, reloaded.score(feats))


def test_simulation_downside_below_median() -> None:
    rng = np.random.default_rng(0)
    hist = np.array([10_000, 0, 9_000, 0, 8_000, 11_000] * 8, dtype=float)
    totals = simulate_totals(hist, n_sims=3000, method="block", rng=rng)
    p10 = capacity_percentile(totals, DOWNSIDE_PERCENTILE)
    assert 0 < p10 < np.median(totals)


def test_quitting_farmer_scores_below_steady() -> None:
    df = preprocess(_synthetic())
    feats = build_features(df)
    scores = HeuristicRiskModel().fit(feats).score(feats)
    steady = scores[df.index[0::3]].mean()   # kind 0
    quitting = scores[df.index[2::3]].mean()  # kind 2
    assert steady > quitting


def test_recommend_rejects_high_risk() -> None:
    scores = pd.Series([0.9, 0.5, 0.1], index=[1, 2, 3])
    capacity = pd.Series([100_000, 100_000, 100_000], index=[1, 2, 3])
    recs = recommend(scores, capacity)
    assert recs.loc[3, "decision"] == "Reject"
    assert recs.loc[1, "decision"] == "Approve"
    assert recs.loc[1, "recommended_loan"] > 0


def test_risk_band_thresholds() -> None:
    assert risk_band(0.9) == "Low"
    assert risk_band(0.45) == "Medium"
    assert risk_band(0.1) == "High"


def test_backtest_metrics_present() -> None:
    df = preprocess(_synthetic(n=90))
    recs, metrics = backtest(df)
    assert set(recs.index) == set(df.index)
    for key in ("approval_rate", "repayment_success_rate", "recovery_rate", "expected_loss_rate"):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["n_approved"] <= metrics["n_farmers"]


def test_stress_capacity_safer_than_mean() -> None:
    """The conservative (low-percentile) capacity should repay more often than
    sizing loans off the average payment."""
    df = preprocess(_synthetic(n=120))
    history_cols, future_cols = train_test_split_weeks()
    actual_future = df[future_cols].to_numpy(dtype=float).sum(axis=1)

    stress = stress_capacity_table(df, history_cols).to_numpy()
    mean_cap = df[history_cols].to_numpy(dtype=float).mean(axis=1) * LOAN_WEEKS

    stress_ok = (stress <= actual_future).mean()
    mean_ok = (mean_cap <= actual_future).mean()
    assert stress_ok > mean_ok
