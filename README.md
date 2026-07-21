# Customer Segmentation & Retention Analysis

> Identifies which customers are worth retaining by combining churn prediction with lifetime value modeling and RFM segmentation — deployed as a FastAPI + React app with MLflow experiment tracking.

## Problem Statement
Businesses lose revenue when valuable customers churn silently. This project answers: **which customers should we spend retention resources on, and which ones aren't worth saving?**

## Solution
A full end-to-end pipeline that:
1. Segments customers using RFM analysis + K-Means clustering
2. Predicts future revenue per customer using BG/NBD + Gamma-Gamma models
3. Predicts churn probability using XGBoost
4. Combines both into a priority action matrix — Retain, Nurture, Let Go, or Monitor

## Tech Stack
| Layer | Tools |
|---|---|
| Data & Cleaning | Python, Pandas, NumPy |
| Segmentation | K-Means (scikit-learn) |
| LTV Modeling | BG/NBD + Gamma-Gamma (lifetimes) |
| Churn Prediction | XGBoost, Logistic Regression, SMOTE |
| Interpretability | SHAP |
| Experiment Tracking | MLflow + DagsHub |
| Backend API | FastAPI |
| Frontend | React, Recharts |
| Version Control | Git, GitHub, DagsHub, DVC |

## Key Results
- 33.3% churn rate identified from 4,334 customers
- ROC-AUC: 0.788 (Logistic Regression), 0.7798 (XGBoost)
- 226 customers flagged as "Retain Immediately" — high LTV + high churn risk
- predicted_ltv (from BG/NBD) was the #1 most important churn feature (SHAP)

## Live Deployment

| Service | URL |
|---|---|
| Frontend (React, static site) | https://customer-segmentation-retention-1-x35i.onrender.com |
| Backend (FastAPI) | https://customer-segmentation-retention-lfyh.onrender.com |
| API docs (Swagger) | https://customer-segmentation-retention-lfyh.onrender.com/docs |

Both run on Render's free tier. An [UptimeRobot](https://uptimerobot.com) monitor pings the backend's `/health` endpoint every 5 minutes to prevent free-tier cold starts (spin-down after ~15 min idle).

## Project Structure
customer-segmentation-retention/
├── backend/          # FastAPI app
├── frontend/         # React dashboard
├── data/             # Raw and processed data (DVC tracked)
├── models/           # Trained model artifacts (DVC tracked, synced via DagsHub)
├── notebooks/        # Colab notebooks
├── src/              # Reusable Python scripts
└── Jenkinsfile       # CI pipeline definition

## How to Run Locally

**Backend:**
```bash
cd backend
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend/my-app
npm start
```

Note: the frontend reads its backend URL from `REACT_APP_API_URL` at **build time** (Create React App bakes env vars into the static bundle — changing it after building has no effect). For local dev this defaults to `http://127.0.0.1:8000` via `.env`.

## Run with Docker

Requires the model artifacts to already be present locally (`dvc pull` if you haven't):

```bash
docker compose up --build
```
- Frontend: http://localhost:3000
- Backend: http://localhost:8000

The backend container has a healthcheck against `/health`; the frontend container won't start until it passes.

**Local development with hot reload** (mounts source as volumes instead of baking a static build):
```bash
docker compose -f docker-compose.dev.yml up --build
```

## Model Artifacts & DVC

Trained model files (`.pkl`) aren't committed to git — they're tracked via DVC with DagsHub as the remote storage backend. After cloning:
```bash
dvc pull
```
requires `DAGSHUB_USER` / `DAGSHUB_TOKEN` configured in `.dvc/config.local` (gitignored, never commit this). On Render, these are set as environment variables and `entrypoint.sh` configures the DVC remote credentials at container startup before pulling.

**Important:** after retraining any model, both steps are required, not just one:
```bash
dvc add models/<file>.pkl   # updates the local .dvc pointer + cache
dvc push                    # uploads the actual blob to the DagsHub remote
git add models/<file>.pkl.dvc
git commit -m "..."
git push
```
Skipping `dvc push` leaves the git-committed pointer referencing a hash that doesn't exist on the remote — `dvc pull` will then fail with a `missing-files` error on deploy, even though everything looks fine locally.

## Continuous Integration (Jenkins)

CI runs on a locally-hosted Jenkins instance (not GitHub Actions), triggered automatically on every push to `main` via a GitHub webhook:

```
git push → GitHub webhook → ngrok tunnel → local Jenkins → pipeline runs
```

- **Backend stage** — runs inside a `python:3.12-slim` container with `models/` and `data/` mounted from a local pre-populated checkout (avoids depending on a network `dvc pull` succeeding on every CI run), then runs `pytest`
- **Frontend stage** — runs inside `node:20-alpine`: `npm ci`, test suite, production build

**Local setup requirements** (see `Jenkinsfile`):
- Jenkins running locally with the GitHub plugin installed, "GitHub hook trigger for GITScm polling" enabled on the job
- A persistent ngrok tunnel exposing Jenkins publicly, using a fixed dev domain so the GitHub webhook URL never needs updating:
  ```bash
  nohup ngrok http --url=https://porthole-unvocal-upstate.ngrok-free.dev 8080 > ngrok.log 2>&1 &
  ```
  (`nohup ... &` detaches it so it survives closing the terminal — it only stops on a full machine restart)
- GitHub webhook payload URL: `https://<ngrok-domain>/github-webhook/`, content type `application/json`
- `LOCAL_REPO_WITH_DATA` in the `Jenkinsfile` environment block must point at a local checkout with real model/data files already pulled via `dvc pull`

## API Endpoints
- `GET /health` — service + model load status
- `GET /segments` — customer segment breakdown
- `GET /actions` — retention priority matrix
- `GET /retain` — high-value, high-risk retain list
- `GET /customer/{customer_id}` — lookup by ID
- `POST /predict` — live churn/LTV inference for a new customer
- Rate limiting: `/predict` is limited to 10 requests per minute per client
- Validation errors return JSON errors for malformed dates and invalid input

## Deployment Notes / Known Quirks
- Both Render services are on the free tier — first request after ~15 min idle takes up to 50s (mitigated by the UptimeRobot monitor above)
- CORS: the backend's `ALLOWED_ORIGINS` env var must include the frontend's live URL, or browser requests from it will be blocked
- Renaming a Render service changes its display name but **not** its live `.onrender.com` subdomain — that's fixed at creation time

## Testing
- Backend regression tests: pytest backend/test_main.py
- Frontend smoke test: npm test -- --watch=false --runInBand

## Experiments
View all MLflow experiment runs on DagsHub:
[https://dagshub.com/pavansai2608/customer-segmentation-retention/experiments](https://dagshub.com/pavansai2608/customer-segmentation-retention/experiments)