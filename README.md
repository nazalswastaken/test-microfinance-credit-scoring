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

Drop the dataset at `data/raw/payments.csv`, then from the `loan-risk` folder:

```
./setup.sh        # makes the venv, installs deps, trains and scores in one go
```

or step by step (see SETUP.md for the details, including `make` targets):

```
pip install -r requirements.txt
python scripts/train.py   # fit the scorer, save it to models/
python scripts/score.py   # score every farmer, write reports/loan_recommendations.csv
pytest                    # run the tests
```

The notebooks in `notebooks/` walk through the EDA, features, modeling and
results with charts.

## How it works

**Score, then size.** Two separate questions, two separate pieces.

The **risk score** (`models.py`) answers eligibility. No default labels exist, so
it's an unsupervised, readable thing: normalize a handful of downside and
consistency features against the population and take a weighted average, 0..1,
higher is safer. Farmers in the bottom band are rejected outright.

The **loan size** comes from the farmer's payment capacity over the 13-week loan.
The interesting part is that you can't just use their average. Payments are
strongly seasonal (see below), so a 13-week loan can land entirely in a low
season. Instead of the average I size off a *stress* capacity: assume every week
of the loan pays like the farmer's 10th-percentile week. The loan is then the
part of that the bank can actually deduct.

There's also a Monte-Carlo / bootstrap simulation of the next 13 weeks in
`models.py`, used in the modeling notebook to show *why* the stress number is the
one to size against (summing 13 resampled weeks averages out the bad ones, so its
downside isn't conservative enough on its own).

## The seasonality, and what the backtest shows

Average payments ramp to a peak around weeks 21-28 and then fall to about a third
of that at an annual trough around weeks 41-48. This dominates the risk.

Because loans last 13 weeks and we have 52, there's a natural backtest: decide
using the first 39 weeks, then check repayment against the real last 13. The
`rolling_backtest` runs this at every issue week:

- Loans issued into a rising season (weeks ~14-26): **95-98% repay, 1-4% loss.**
- Loans issued straight into the trough (weeks ~30-40): repayment drops toward
  65% and loss climbs past 20%.

So *when* a loan is issued matters as much as who it's issued to. A loan taken at
year end runs into weeks 53-65, the post-trough recovery, which is the safe
regime. Sizing off the stress capacity is what keeps even the hard windows
mostly repayable (it's the difference between ~25% and ~2% loss).

## Answers to the two questions

1. **Who to target:** farmers active most weeks with a solid worst-quarter floor,
   which is what the score rewards.
2. **How much to lend:** the deductible share of the stress capacity, i.e. a
   deliberately bad-season estimate of the next 13 weeks of payments.

## Future improvements

- Predict the actual next-13-week level with a seasonal model so strong farmers
  aren't under-lent in good seasons, instead of always using the stress proxy.
- Swap the heuristic score for a supervised model once real repayment outcomes
  are available, and calibrate the bands to a target loss rate.
- Optimise at the portfolio level (spread issue timing) rather than scoring
  farmers one at a time.
