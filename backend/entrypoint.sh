#!/bin/sh
set -e

if [ -n "$DAGSHUB_USER" ] && [ -n "$DAGSHUB_TOKEN" ]; then
    echo "Configuring DVC remote credentials..."
    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user "$DAGSHUB_USER"
    dvc remote modify origin --local password "$DAGSHUB_TOKEN"

    if [ ! -d ../.git ]; then
        echo "Initializing minimal git repo for DVC..."
        cd ..
        git init -q
        git config user.email "deploy@render.local"
        git config user.name "Render Deploy"
        git add .dvc models/*.dvc .dvcignore 2>/dev/null || true
        git commit -q -m "init for dvc" --allow-empty
        cd backend
    fi

    echo "Pulling model artifacts from DagsHub..."
    dvc pull || echo "WARNING: dvc pull failed — models may be missing"
else
    echo "WARNING: DAGSHUB_USER/DAGSHUB_TOKEN not set — skipping dvc pull"
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"