"""Risk models for scoring farmers.

Two pieces live here:

1. HeuristicRiskModel - a transparent 0..1 safety score. It normalizes a handful
   of downside/consistency features against the population and takes a weighted
   average. No labels needed (we don't have any), and it's easy to read off why a
   farmer scored the way they did.

2. A simulation of the next 13 weeks of payments. Because farmers can skip or
   stop supplying, the honest question isn't "what's the expected payment" but
   "how bad could the next 13 weeks realistically get". We bootstrap/Monte-Carlo
   the farmer's own history to get a distribution of the 13-week payment total,
   then read a pessimistic percentile off it. That percentile is what the loan is
   sized against (see recommendation.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DOWNSIDE_PERCENTILE, LOAN_WEEKS, N_SIMULATIONS

# Features that feed the heuristic score, mapped to a signed weight. A positive
# weight means "more is safer"; a negative weight means "more is riskier".
DEFAULT_WEIGHTS: dict[str, float] = {
    "active_week_pct": 1.5,
    "min_roll13_mean": 1.5,
    "p10_payment": 1.0,
    "recent_vs_history": 0.5,
    "norm_entropy": 0.75,
    "coef_variation": -1.5,
    "longest_zero_streak": -1.0,
    "downside_deviation": -0.75,
}


@dataclass
class HeuristicRiskModel:
    """Population-normalized weighted score in [0, 1] (higher = safer)."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # per-feature (p5, p95) reference range learned at fit time
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, features: pd.DataFrame) -> "HeuristicRiskModel":
        self.ranges = {}
        for name in self.weights:
            if name not in features.columns:
                raise KeyError(f"feature '{name}' missing from feature table")
            lo = float(np.percentile(features[name], 5))
            hi = float(np.percentile(features[name], 95))
            self.ranges[name] = (lo, hi)
        return self

    def _normalize(self, name: str, values: pd.Series) -> np.ndarray:
        lo, hi = self.ranges[name]
        span = hi - lo
        if span == 0:
            return np.zeros(len(values))
        return np.clip((values.to_numpy(dtype=float) - lo) / span, 0.0, 1.0)

    def score(self, features: pd.DataFrame) -> pd.Series:
        if not self.ranges:
            raise RuntimeError("model is not fitted; call fit() first")
        total_weight = sum(abs(w) for w in self.weights.values())
        acc = np.zeros(len(features))
        for name, weight in self.weights.items():
            norm = self._normalize(name, features[name])
            oriented = norm if weight > 0 else (1.0 - norm)
            acc += abs(weight) * oriented
        return pd.Series(acc / total_weight, index=features.index, name="risk_score")

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "ranges": {k: list(v) for k, v in self.ranges.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "HeuristicRiskModel":
        model = cls(weights=dict(payload["weights"]))
        model.ranges = {k: (float(v[0]), float(v[1])) for k, v in payload["ranges"].items()}
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HeuristicRiskModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --- loan capacity -------------------------------------------------------

def stress_capacity(
    history: np.ndarray,
    percentile: int = DOWNSIDE_PERCENTILE,
    horizon: int = LOAN_WEEKS,
) -> float:
    """Conservative estimate of the payment total over the loan period.

    The payments are strongly seasonal, so a 13-week loan can fall entirely in a
    low season. Rather than trust the yearly average, we assume every week of the
    loan pays like the farmer's low-percentile week: p_low(weekly) * horizon. It's
    deliberately pessimistic, which is what keeps loans repayable through a trough
    (see the EDA notebook for the seasonality this guards against).
    """
    history = np.asarray(history, dtype=float)
    if len(history) == 0:
        return 0.0
    return float(np.percentile(history, percentile) * horizon)


def stress_capacity_table(
    df: pd.DataFrame,
    week_cols: list[str],
    percentile: int = DOWNSIDE_PERCENTILE,
    horizon: int = LOAN_WEEKS,
) -> pd.Series:
    """Stress capacity for every farmer, indexed like ``df``."""
    matrix = df[week_cols].to_numpy(dtype=float)
    values = np.percentile(matrix, percentile, axis=1) * horizon
    return pd.Series(values, index=df.index, name="loan_capacity")


# --- simulation (diagnostic) --------------------------------------------

def simulate_totals(
    history: np.ndarray,
    horizon: int = LOAN_WEEKS,
    n_sims: int = N_SIMULATIONS,
    method: str = "bootstrap",
    block: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate the total payment over the next ``horizon`` weeks.

    Returns an array of ``n_sims`` simulated horizon-week totals, drawn by
    resampling the farmer's own weekly history.

    method:
        "bootstrap" -> draw weeks independently with replacement
        "block"     -> moving-block bootstrap, keeps short runs together so
                       streaky skip/active behavior is preserved
    """
    history = np.asarray(history, dtype=float)
    n = len(history)
    if n == 0:
        return np.zeros(n_sims)
    rng = rng or np.random.default_rng()

    if method == "bootstrap":
        idx = rng.integers(0, n, size=(n_sims, horizon))
        return history[idx].sum(axis=1)

    if method == "block":
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, n, size=(n_sims, n_blocks))
        totals = np.zeros(n_sims)
        for b in range(n_blocks):
            take = min(block, horizon - b * block)
            offsets = np.arange(take)
            idx = (starts[:, b][:, None] + offsets) % n
            totals += history[idx].sum(axis=1)
        return totals

    raise ValueError(f"unknown method: {method!r}")


def capacity_percentile(totals: np.ndarray, percentile: int = DOWNSIDE_PERCENTILE) -> float:
    """Pessimistic estimate of the horizon-week payment total."""
    return float(np.percentile(totals, percentile))


def simulate_capacity_table(
    df: pd.DataFrame,
    week_cols: list[str],
    percentile: int = DOWNSIDE_PERCENTILE,
    n_sims: int = N_SIMULATIONS,
    method: str = "block",
    seed: int = 0,
) -> pd.DataFrame:
    """Run the simulation for every farmer and return capacity percentiles.

    Columns: sim_p{percentile}, sim_median, sim_mean of the 13-week total.
    """
    rng = np.random.default_rng(seed)
    matrix = df[week_cols].to_numpy(dtype=float)
    rows = []
    for hist in matrix:
        totals = simulate_totals(hist, n_sims=n_sims, method=method, rng=rng)
        rows.append(
            {
                f"sim_p{percentile}": capacity_percentile(totals, percentile),
                "sim_median": float(np.median(totals)),
                "sim_mean": float(totals.mean()),
            }
        )
    return pd.DataFrame(rows, index=df.index)
