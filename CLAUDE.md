# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Backend (Python 3.11/3.12, venv at `.venv/`)**
```bash
pip install -r requirements.txt
cd backend && uvicorn main:app --reload        # dev server on :8000
pytest backend/test_main.py                    # all backend tests
pytest backend/test_main.py::test_health_reports_status   # single test
```
Backend tests import `main.py` directly via `importlib` from `backend/`, and `main.py` loads model artifacts at **module import time** — so `models/` must be populated (`dvc pull`) or the tests that hit `/segments` will fail.

**Frontend (CRA, React 19)**
```bash
cd frontend/my-app
npm start                                      # dev server on :3000
CI=true npm test -- --watchAll=false           # non-watch test run (what CI does)
npm test -- -t "renders the dashboard title"   # single test
npm run build
```

**Retraining / regenerating artifacts**
```bash
python src/build_decision_matrix.py            # needs data/raw/online_retail_II.csv
```
`requirements.txt` covers the *serving* path only. This script additionally needs `imbalanced-learn` (SMOTE) — install it separately; don't assume `pip install -r requirements.txt` is enough to retrain.

**Docker**
```bash
docker compose up --build                                # prod-ish: nginx-served static build
docker compose -f docker-compose.dev.yml up --build      # hot reload, source mounted
```

## Architecture

Four models feed one CSV, and the API serves that CSV plus live inference:

1. `src/build_decision_matrix.py` is the single source of truth for the ML pipeline (it reproduces `notebooks/Customer_Segmentation.ipynb`). It cleans raw transactions → RFM + K-Means segments (`Champions` / `Loyal Customers` / `At Risk` / `Hibernating`, assigned by ranking cluster means, not by cluster index) → BG/NBD + Gamma-Gamma LTV → XGBoost + LogisticRegression churn (SMOTE-balanced) → writes `models/final_decision_matrix.csv`.
2. `backend/main.py` loads `xgb_churn_model.pkl` (joblib), `bgf_model.pkl` / `ggf_model.pkl` (**dill**, not joblib — the `lifetimes` fitters hold lambdas that pickle can't handle), and the decision matrix at import time. Precomputed endpoints (`/segments`, `/actions`, `/retain`) read the CSV; `/customer/{id}` and `/predict` re-run live inference.
3. `frontend/my-app/src/App.js` is a single ~550-line component holding the whole dashboard (charts, theme toggle, customer lookup, predict form). There is no router and no component directory.

**The action matrix is duplicated in two places** — `recommend_action()` in `backend/main.py` and `build_action_column()` in `src/build_decision_matrix.py`. Both use `churn_probability > 0.5` × `predicted_ltv > median` → retain / let_go / nurture / monitor. Change one, change the other, or the live `/predict` result will disagree with the precomputed matrix.

**Model feature contract:** the XGBoost model is trained on exactly `["Frequency", "Monetary", "predicted_ltv"]`, in that order. Both `/customer/{id}` and `/predict` build DataFrames matching it; changing `FEATURE_COLUMNS` in the training script requires the same change in `backend/main.py`.

**Graceful degradation:** if `bgf`/`ggf` fail to load, `/predict` falls back to the heuristic `estimate_predicted_ltv()` and `/health` reports `using_ltv_fallback: true`. If the XGBoost model or CSV are missing, `MODELS_READY` is false and data endpoints return 503 rather than crashing at startup.

## Deployment constraints

- **`REACT_APP_API_URL` is baked in at build time** (CRA). Setting it at runtime on a built bundle does nothing — it must be a Docker build arg / Render build-time env var.
- **CORS**: backend reads `ALLOWED_ORIGINS` (comma-separated); the live frontend URL must be in it.
- **Docker layout matters**: `main.py` resolves paths as `BASE_DIR.parent / "models"`, so the backend image copies `backend/` and `models/` side by side under `/app`. Build from the repo root with `-f backend/Dockerfile .`.
- `backend/entrypoint.sh` runs on container start: configures the DVC remote from `DAGSHUB_USER`/`DAGSHUB_TOKEN`, `git init`s a throwaway repo (DVC needs one), `dvc pull`s the artifacts, then execs uvicorn.

## DVC

Model and data files are DVC-tracked with DagsHub as the remote; only `*.dvc` pointers are in git. After retraining, **all** of these are required:
```bash
dvc add models/<file>.pkl && dvc push
git add models/<file>.pkl.dvc && git commit && git push
```
Skipping `dvc push` commits a pointer to a hash that doesn't exist on the remote — everything works locally, and `dvc pull` fails on deploy with `missing-files`. Credentials live in `.dvc/config.local` (gitignored).

## CI

Jenkins runs locally (not GitHub Actions), triggered by a GitHub webhook through a fixed-domain ngrok tunnel. The backend stage mounts `models/` and `data/` from `LOCAL_REPO_WITH_DATA` (a pre-`dvc pull`ed checkout on the host, path hardcoded in the `Jenkinsfile` environment block) instead of pulling over the network — so CI depends on that local checkout staying populated. See the README's "Continuous Integration" section for the ngrok/webhook setup.
