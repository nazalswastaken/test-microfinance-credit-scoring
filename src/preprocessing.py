"""Cleaning and imputation on top of the loaded data.

Takes the validated DataFrame from data.py and gets it ready for feature work.
The main decision here is what a missing (`-`) week means. For a milk supplier a
skipped week is a week with no delivery and therefore no payment, so by default
I treat it as a zero. That also keeps the "skip risk" in the data instead of
hiding it, which is exactly the thing we're trying to price.
"""

from __future__ import annotations

import pandas as pd

from .utils import WEEK_COLUMNS


def fill_missing(df: pd.DataFrame, strategy: str = "zero") -> pd.DataFrame:
    """Fill missing weekly payments.

    strategy:
        "zero"    -> a skipped week is no payment (default)
        "keep"    -> leave NaNs as-is
    """
    week_cols = [c for c in WEEK_COLUMNS if c in df.columns]
    out = df.copy()
    if strategy == "zero":
        out[week_cols] = out[week_cols].fillna(0.0)
    elif strategy == "keep":
        pass
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")
    return out


def clip_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Clip any negative payments to zero (payments shouldn't be negative)."""
    week_cols = [c for c in WEEK_COLUMNS if c in df.columns]
    out = df.copy()
    out[week_cols] = out[week_cols].clip(lower=0.0)
    return out


def preprocess(df: pd.DataFrame, strategy: str = "zero") -> pd.DataFrame:
    """Standard cleaning: fill missing weeks then clip negatives."""
    return clip_negatives(fill_missing(df, strategy=strategy))
