VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STAMP := $(VENV)/.installed

.PHONY: setup train score run test notebooks clean

# create the venv and install requirements (only re-runs if requirements change)
setup: $(STAMP)

$(STAMP): requirements.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.txt
	touch $(STAMP)

train: setup
	$(PY) scripts/train.py

score: setup
	$(PY) scripts/score.py

# full run: fit the model, then score everyone into reports/
run: train score

test: setup
	$(PY) -m pytest -q

# re-run the notebooks in place with fresh outputs
notebooks: setup
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# remove the venv and generated artifacts (keeps your data and code)
clean:
	rm -rf $(VENV)
	rm -f models/risk_model.json reports/loan_recommendations.csv
