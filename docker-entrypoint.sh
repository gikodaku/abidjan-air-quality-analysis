#!/bin/sh
set -e
if [ ! -f models/forecaster.joblib ]; then
    echo "No model found: running the full pipeline (about 4 min)…"
    python -m aaqa fetch process report train
fi
exec "$@"
