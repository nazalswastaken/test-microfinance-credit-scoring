# loan-risk

Modeling credit risk for a dairy-farmer lending product.

A bank lends to dairy farmers against the weekly milk payments they receive from
a food company. Loans run for 13 weeks and are repaid by deducting from those
future payments. The catch: farmers aren't obligated to keep supplying milk, so
a payment stream that looks healthy today can dry up. All I have to work with is
one year of weekly payments per farmer, no demographics, no loan history,
nothing else. So the whole thing comes down to reading the payment history.

Two questions I'm trying to answer:

1. Which farmers are likely to keep paying consistently over the next 13 weeks?
2. Given the history, how much is reasonable to lend?

Because farmers can walk away, I care more about the downside of the payment
stream than the average.

## Data

One CSV, one row per farmer: `Farmer No.` and `Week1 … Week52`. Each cell is the
total payment that week. Skipped weeks show up as `-` and amounts have thousands
separators. Raw data lives in `data/raw/` and is git-ignored (see
`data/README.md`).

## Layout

```
data/         raw + processed data (git-ignored)
notebooks/    exploration, not production code
  01_eda.ipynb
  02_feature_engineering.ipynb
  03_modeling.ipynb
  04_results.ipynb
src/          the reusable pieces
  data.py           load + validate
  preprocessing.py  cleaning / imputation
  features.py       feature engineering
  models.py         scoring / risk models
  evaluation.py     metrics
  visualization.py  plots
  recommendation.py loan sizing
  utils.py
scripts/
  train.py    load -> preprocess -> features -> train -> save
  score.py    load model -> score -> recommend -> export csv
tests/
```

Idea is to keep anything reusable in `src/` and keep the notebooks for poking at
the data. Each stage should be testable on its own, and the model/recommendation
side is kept swappable so I can try different approaches without rewriting
everything.

## Running

```
pip install -r requirements.txt
# drop the dataset at data/raw/payments.csv
python -c "from src.data import load_payments; print(load_payments().shape)"
```

## Where this is at

Data loading is done. Next up is EDA, then features, modeling, and the loan
recommendation logic.
