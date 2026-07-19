#!/bin/sh
set -e

if [ -n "$DAGSHUB_USER" ] && [ -n "$DAGSHUB_TOKEN" ]; then
    echo "Configuring DVC remote credentials..."
    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user "$DAGSHUB_USER"
    dvc remote modify origin --local password "$DAGSHUB_TOKEN"
    echo "Pulling model artifacts from DagsHub..."
    dvc pull || echo "WARNING: dvc pull failed — models may be missing"
else
    echo "WARNING: DAGSHUB_USER/DAGSHUB_TOKEN not set — skipping dvc pull"
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"