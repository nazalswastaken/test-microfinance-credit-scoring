# Setup

Everything runs inside a local virtual environment (`.venv`), a private copy of
Python plus this project's libraries. It's git-ignored and machine-specific, so
it isn't committed. You create it once.

## Quickest way

From the `loan-risk` folder:

```bash
./setup.sh
```

That creates the venv, installs the requirements, trains the model, and writes
the recommendations to `reports/loan_recommendations.csv`. Run it any time; it's
safe to re-run.

If you get a permission error, make it executable first:

```bash
chmod +x setup.sh
```

## Or with make

```bash
make run      # setup (if needed) + train + score
make test     # run the tests
make notebooks# re-execute the notebooks with fresh outputs
make clean    # delete the venv and generated files
```

## Or by hand

```bash
python3 -m venv .venv
source .venv/bin/activate          # activate it (prompt shows (.venv))
pip install -r requirements.txt

python scripts/train.py            # fit + save model -> models/
python scripts/score.py            # score farmers  -> reports/loan_recommendations.csv
pytest                             # tests

deactivate                         # leave the environment when done
```

## Working in the environment later

Whenever you open a new terminal and want to run project code, activate the venv
first:

```bash
cd /Users/raem/Desktop/Projects/Famer_Credit_Scoring/loan-risk
source .venv/bin/activate
```

Then `jupyter notebook` to open the notebooks, or run any of the scripts above.

## The output

`reports/loan_recommendations.csv` has one row per farmer: risk score, risk band
(Low / Medium / High), recommended loan amount, and the approve/reject decision.
