"""Feature engineering.

Turns a farmer's weekly payment series into features describing payment level,
consistency, trend and downside risk. Every feature is a small function that
takes one farmer's payments (a 1-D numpy array) and returns a number. They're
registered in FEATURE_FUNCTIONS, so adding a feature is just writing a function
and decorating it. build_features runs them all across a DataFrame.

Nothing here assumes 52 weeks specifically, so the same code works on a shorter
history (handy for backtesting on the first N weeks).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .utils import LOAN_WEEKS, WEEK_COLUMNS

FeatureFn = Callable[[np.ndarray], float]
FEATURE_FUNCTIONS: dict[str, FeatureFn] = {}


def feature(name: str) -> Callable[[FeatureFn], FeatureFn]:
    """Register a feature function under ``name``."""

    def decorate(fn: FeatureFn) -> FeatureFn:
        FEATURE_FUNCTIONS[name] = fn
        return fn

    return decorate


# --- small helpers -------------------------------------------------------

def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values in a boolean array."""
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


# --- payment level -------------------------------------------------------

@feature("yearly_total")
def yearly_total(x: np.ndarray) -> float:
    return float(x.sum())


@feature("weekly_mean")
def weekly_mean(x: np.ndarray) -> float:
    return float(x.mean())


@feature("weekly_median")
def weekly_median(x: np.ndarray) -> float:
    return float(np.median(x))


@feature("max_payment")
def max_payment(x: np.ndarray) -> float:
    return float(x.max())


@feature("min_payment")
def min_payment(x: np.ndarray) -> float:
    return float(x.min())


@feature("first_13_mean")
def first_13_mean(x: np.ndarray) -> float:
    return float(x[:LOAN_WEEKS].mean())


@feature("last_13_mean")
def last_13_mean(x: np.ndarray) -> float:
    return float(x[-LOAN_WEEKS:].mean())


# --- consistency ---------------------------------------------------------

@feature("std")
def std(x: np.ndarray) -> float:
    return float(x.std())


@feature("coef_variation")
def coef_variation(x: np.ndarray) -> float:
    return _safe_div(float(x.std()), float(x.mean()))


@feature("zero_weeks")
def zero_weeks(x: np.ndarray) -> float:
    return float((x <= 0).sum())


@feature("active_week_pct")
def active_week_pct(x: np.ndarray) -> float:
    return _safe_div(float((x > 0).sum()), float(len(x)))


@feature("longest_zero_streak")
def longest_zero_streak(x: np.ndarray) -> float:
    return float(_longest_run(x <= 0))


@feature("longest_active_streak")
def longest_active_streak(x: np.ndarray) -> float:
    return float(_longest_run(x > 0))


# --- trend ---------------------------------------------------------------

@feature("trend_slope")
def trend_slope(x: np.ndarray) -> float:
    """Slope of a least-squares line through the payments (per week)."""
    weeks = np.arange(len(x), dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(weeks, x, 1)[0])


@feature("recent_vs_history")
def recent_vs_history(x: np.ndarray) -> float:
    """Mean of the last 13 weeks over the overall mean (>1 means improving)."""
    return _safe_div(float(x[-LOAN_WEEKS:].mean()), float(x.mean()))


@feature("momentum")
def momentum(x: np.ndarray) -> float:
    """Last-13 average minus first-13 average."""
    return float(x[-LOAN_WEEKS:].mean() - x[:LOAN_WEEKS].mean())


@feature("rolling_std_mean")
def rolling_std_mean(x: np.ndarray) -> float:
    """Average of the rolling 13-week standard deviation."""
    s = pd.Series(x).rolling(LOAN_WEEKS).std().dropna()
    return float(s.mean()) if len(s) else float(x.std())


# --- downside risk -------------------------------------------------------

@feature("min_roll13_mean")
def min_roll13_mean(x: np.ndarray) -> float:
    """Worst 13-week average anywhere in the history (a stress window)."""
    s = pd.Series(x).rolling(LOAN_WEEKS).mean().dropna()
    return float(s.min()) if len(s) else float(x.mean())


@feature("p10_payment")
def p10_payment(x: np.ndarray) -> float:
    return float(np.percentile(x, 10))


@feature("p25_payment")
def p25_payment(x: np.ndarray) -> float:
    return float(np.percentile(x, 25))


@feature("downside_deviation")
def downside_deviation(x: np.ndarray) -> float:
    """Std of the weeks that fall below the mean (semi-deviation)."""
    mean = x.mean()
    below = x[x < mean] - mean
    return float(np.sqrt((below**2).mean())) if len(below) else 0.0


# --- other ---------------------------------------------------------------

@feature("autocorr_lag1")
def autocorr_lag1(x: np.ndarray) -> float:
    """Week-to-week autocorrelation."""
    if len(x) < 3 or x.std() == 0:
        return 0.0
    a, b = x[:-1], x[1:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@feature("norm_entropy")
def norm_entropy(x: np.ndarray) -> float:
    """Shannon entropy of the payment distribution, scaled to 0..1.

    Near 1 means payments are spread evenly across weeks; near 0 means they're
    concentrated in a few weeks.
    """
    total = x.sum()
    if total <= 0:
        return 0.0
    p = x[x > 0] / total
    ent = -(p * np.log(p)).sum()
    return _safe_div(float(ent), float(np.log(len(x))))


@feature("max_weekly_change")
def max_weekly_change(x: np.ndarray) -> float:
    return float(np.abs(np.diff(x)).max()) if len(x) > 1 else 0.0


@feature("mean_abs_weekly_change")
def mean_abs_weekly_change(x: np.ndarray) -> float:
    return float(np.abs(np.diff(x)).mean()) if len(x) > 1 else 0.0


# --- builder -------------------------------------------------------------

def features_for_series(x: np.ndarray) -> dict[str, float]:
    """Run every registered feature on a single farmer's payments."""
    x = np.asarray(x, dtype=float)
    return {name: fn(x) for name, fn in FEATURE_FUNCTIONS.items()}


def build_features(df: pd.DataFrame, week_cols: list[str] | None = None) -> pd.DataFrame:
    """Build the feature table for every farmer.

    df is expected to be indexed by farmer id with weekly-payment columns. Pass
    week_cols to restrict to a sub-window (e.g. the first 39 weeks) for
    backtesting; otherwise all present week columns are used.
    """
    if week_cols is None:
        week_cols = [c for c in WEEK_COLUMNS if c in df.columns]
    if not week_cols:
        raise ValueError("no week columns found to build features from")

    matrix = df[week_cols].to_numpy(dtype=float)
    rows = [features_for_series(row) for row in matrix]
    return pd.DataFrame(rows, index=df.index)
