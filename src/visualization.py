"""Plotting helpers.

Reusable charts for the notebooks so the plot code doesn't get copy-pasted
around. Everything takes an optional ax and returns it, so they compose into
subplot grids. matplotlib only, seaborn if it's around for the heatmap.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import WEEK_COLUMNS


def _ax(ax: plt.Axes | None) -> plt.Axes:
    return ax if ax is not None else plt.gca()


def plot_payment_history(df: pd.DataFrame, farmer_id, ax: plt.Axes | None = None) -> plt.Axes:
    """Weekly payments for one farmer over the year."""
    ax = _ax(ax)
    week_cols = [c for c in WEEK_COLUMNS if c in df.columns]
    series = df.loc[farmer_id, week_cols].to_numpy(dtype=float)
    ax.plot(range(1, len(series) + 1), series, marker="o", ms=3)
    ax.set_title(f"Farmer {farmer_id}")
    ax.set_xlabel("week")
    ax.set_ylabel("payment")
    return ax


def plot_distribution(values, bins: int = 50, title: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    ax = _ax(ax)
    ax.hist(np.asarray(values, dtype=float), bins=bins)
    if title:
        ax.set_title(title)
    return ax


def plot_boxplot(values, title: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    ax = _ax(ax)
    ax.boxplot(np.asarray(values, dtype=float), vert=True)
    if title:
        ax.set_title(title)
    return ax


def plot_week_correlation(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Heatmap of correlations between weeks (are near weeks related?)."""
    ax = _ax(ax)
    week_cols = [c for c in WEEK_COLUMNS if c in df.columns]
    corr = df[week_cols].corr()
    im = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
    ax.set_title("week-to-week correlation")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_score_distribution(scores, ax: plt.Axes | None = None) -> plt.Axes:
    ax = _ax(ax)
    ax.hist(np.asarray(scores, dtype=float), bins=40)
    ax.set_title("risk score distribution")
    ax.set_xlabel("risk score (higher = safer)")
    return ax
