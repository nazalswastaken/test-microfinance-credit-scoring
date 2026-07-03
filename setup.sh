#!/usr/bin/env bash
# One-shot setup: make the virtualenv, install everything, train and score.
# Usage:  ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"

PY=python3
VENV=.venv

if [ ! -d "$VENV" ]; then
  echo ">> creating virtualenv in $VENV"
  "$PY" -m venv "$VENV"
fi

echo ">> installing requirements"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r requirements.txt

if [ ! -f data/raw/payments.csv ]; then
  echo "!! data/raw/payments.csv is missing. drop the dataset there, then re-run."
  exit 1
fi

echo ">> training"
"$VENV/bin/python" scripts/train.py

echo ">> scoring"
"$VENV/bin/python" scripts/score.py

echo
echo "done. recommendations are in reports/loan_recommendations.csv"
echo "to work in the environment yourself:  source $VENV/bin/activate"
