"""Cleaning and imputation on top of the loaded data.

Takes the validated DataFrame from data.py and gets it ready for feature work.
This is where I decide what a missing (`-`) week actually means -- treat it as a
zero payment or leave it as NaN -- since that choice affects everything
downstream.

Not written yet.
"""

from __future__ import annotations
