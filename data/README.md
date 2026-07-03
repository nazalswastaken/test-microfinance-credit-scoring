# data

## raw/payments.csv

The dataset. One row per farmer, columns `Farmer No.` and `Week1 … Week52`, each
cell a weekly payment.

A couple of quirks the loader (`src/data.py`) deals with:

- Skipped weeks are written as `-`, which becomes `NaN`. Whether a skipped week
  really means "paid zero" or "unknown" is left for preprocessing to decide, not
  the loader.
- Amounts have thousands separators like `"1,034"`, so they come in as strings
  and need stripping before converting to numbers.

## processed/

Cleaned and feature tables produced by the pipeline.

Raw and processed files aren't committed, only this README is. Put the dataset
at `data/raw/payments.csv` before running anything.
