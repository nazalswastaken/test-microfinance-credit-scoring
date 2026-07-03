"""Risk models for scoring farmers.

Starting with a simple heuristic score and working up from there (regression,
random forest, gradient boosting). Also want to try a distributional approach --
bootstrap / Monte Carlo the next 13 weeks and estimate the odds the payments
cover a given loan. Same interface across models so they're swappable.

Not written yet.
"""

from __future__ import annotations
