import logging
import os
from datetime import datetime
from pathlib import Path

import dill
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_churn_model.pkl"
DATA_PATH = BASE_DIR.parent / "models" / "final_decision_matrix.csv"
BGF_PATH = BASE_DIR.parent / "models" / "bgf_model.pkl"
GGF_PATH = BASE_DIR.parent / "models" / "ggf_model.pkl"

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000"
).split(",")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("customer_segmentation")

app = FastAPI(title="Customer Segmentation API")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load trained model and decision matrix ──
def load_artifacts():
    model = None
    df = pd.DataFrame()
    bgf = None
    ggf = None
    errors = []

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        errors.append(f"model: {exc}")

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        errors.append(f"decision_matrix: {exc}")

    for artifact_name, artifact_path, target in [
        ("bgf", BGF_PATH, "bgf"),
        ("ggf", GGF_PATH, "ggf"),
    ]:
        try:
            with open(artifact_path, "rb") as f:
                loaded = dill.load(f)
            if artifact_name == "bgf":
                bgf = loaded
            else:
                ggf = loaded
        except Exception as exc:
            errors.append(f"{artifact_name}: {exc}")
            logger.warning("Unable to load %s artifact from %s: %s", artifact_name, artifact_path, exc)

    return model, df, bgf, ggf, errors


model, df, bgf, ggf, load_errors = load_artifacts()
ESSENTIAL_READY = model is not None and not df.empty
MODELS_READY = ESSENTIAL_READY
LTV_MEDIAN = float(df["predicted_ltv"].median()) if "predicted_ltv" in df.columns and not df.empty else 0.0
USING_LTV_FALLBACK = bgf is None or ggf is None
ACTION_CODE_MAP = {
    "🔴 Retain Immediately": "retain",
    "⚪ Let Go": "let_go",
    "🟢 Nurture": "nurture",
    "🔵 Monitor": "monitor",
}

def recommend_action(churn_probability: float, predicted_ltv: float) -> dict:
    high_risk = churn_probability > 0.5
    high_ltv = predicted_ltv > LTV_MEDIAN

    if high_risk and high_ltv:
        return {"code": "retain", "label": "🔴 Retain Immediately"}
    elif high_risk and not high_ltv:
        return {"code": "let_go", "label": "⚪ Let Go"}
    elif not high_risk and high_ltv:
        return {"code": "nurture", "label": "🟢 Nurture"}
    else:
        return {"code": "monitor", "label": "🔵 Monitor"}


def estimate_predicted_ltv(first_purchase_date: datetime, last_purchase_date: datetime, total_orders: int, total_spent: float, today: datetime) -> float:
    if total_orders <= 0 or total_spent < 0:
        return 0.0

    avg_order_value = total_spent / total_orders
    frequency = max(total_orders - 1, 0)
    recency_days = max((last_purchase_date - first_purchase_date).days, 0)
    age_days = max((today - first_purchase_date).days, 1)

    base_ltv = avg_order_value * (frequency + 1) * 1.2
    recency_factor = max(0.0, 1 - (recency_days / age_days))
    return round(max(base_ltv * (0.6 + recency_factor * 0.4), 0.0), 2)


class CustomerInput(BaseModel):
    first_purchase_date: str   # "YYYY-MM-DD"
    last_purchase_date: str    # "YYYY-MM-DD"
    total_orders: int
    total_spent: float


# ── Endpoints ──

# 1. Health check
@app.get("/")
def home():
    return {"status": "API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok" if MODELS_READY else "degraded",
        "models_loaded": MODELS_READY,
        "using_ltv_fallback": USING_LTV_FALLBACK,
        "load_errors": load_errors,
    }


# 2. Get all segments summary
@app.get("/segments")
def get_segments():
    if not MODELS_READY:
        return JSONResponse(status_code=503, content={"error": "Model artifacts are not available"})

    summary = df["Segment"].value_counts().reset_index()
    summary.columns = ["segment", "count"]
    return summary.to_dict(orient="records")


# 3. Get priority action summary
@app.get("/actions")
def get_actions():
    if not MODELS_READY:
        return JSONResponse(status_code=503, content={"error": "Model artifacts are not available"})

    summary = df["action"].value_counts().reset_index()
    summary.columns = ["action", "count"]
    return summary.to_dict(orient="records")


# 4. Get top customers to retain immediately
@app.get("/retain")
def get_retain_customers():
    if not MODELS_READY:
        return JSONResponse(status_code=503, content={"error": "Model artifacts are not available"})

    retain = df[df["action"].str.contains("Retain", na=False)]
    retain = retain.sort_values("predicted_ltv", ascending=False).head(20)

    result = retain[["Customer ID", "Frequency", "Monetary",
                     "predicted_ltv", "churn_probability",
                     "Segment", "action"]].copy()
    result["action_code"] = result["action"].map(ACTION_CODE_MAP)
    return result.to_dict(orient="records")


# 5. Search a specific customer (looks up existing customer, re-runs live inference)
@app.get("/customer/{customer_id}")
def get_customer(customer_id: int):
    if not MODELS_READY:
        return JSONResponse(status_code=503, content={"error": "Model artifacts are not available"})

    row = df[df["Customer ID"] == int(customer_id)]
    if row.empty:
        logger.warning("Customer lookup failed for customer_id=%s", customer_id)
        return JSONResponse(status_code=404, content={"error": "Customer not found"})

    #Features the XGBoost model was actually trained on (see notebook cell 40)
    FEATURE_COLUMNS = ["Frequency", "Monetary", "predicted_ltv"]
    features = row[FEATURE_COLUMNS]
    live_churn_prob = float(model.predict_proba(features)[0][1])

    result = row.iloc[0].to_dict()
    result["churn_probability"] = live_churn_prob
    result["action_code"] = ACTION_CODE_MAP.get(result.get("action"), None)
    logger.info("Customer lookup succeeded for customer_id=%s", customer_id)
    return result


# 6. Predict churn for a brand-new customer using simple, human-friendly inputs
@app.post("/predict")
@limiter.limit("10/minute")
def predict_customer(input: CustomerInput, request: Request):
    try:
        first = datetime.strptime(input.first_purchase_date, "%Y-%m-%d")
        last = datetime.strptime(input.last_purchase_date, "%Y-%m-%d")
    except ValueError:
        logger.warning("Prediction rejected due to invalid date format: %s", input)
        return JSONResponse(status_code=400, content={"error": "Dates must be in YYYY-MM-DD format"})
    if not MODELS_READY:
        return JSONResponse(status_code=503, content={"error": "Model artifacts are not available"})

    today = datetime.now()

    # ── NEW: validate date logic before doing any math ──
    if last < first:
        logger.warning("Prediction rejected due to invalid date order: %s", input)
        return JSONResponse(status_code=400, content={"error": "last_purchase_date cannot be before first_purchase_date"})
    if last > today:
        logger.warning("Prediction rejected due to future date: %s", input)
        return JSONResponse(status_code=400, content={"error": "last_purchase_date cannot be in the future"})
    if input.total_orders <= 0:
        logger.warning("Prediction rejected due to invalid order count: %s", input)
        return JSONResponse(status_code=400, content={"error": "total_orders must be greater than 0"})
    if input.total_spent < 0:
        logger.warning("Prediction rejected due to negative spend: %s", input)
        return JSONResponse(status_code=400, content={"error": "total_spent cannot be negative"})
    # ── end new validation ──

    frequency_rfm = input.total_orders
    monetary_rfm = input.total_spent

    lifetimes_frequency = max(input.total_orders - 1, 0)
    lifetimes_recency = (last - first).days
    T = (today - first).days
    avg_order_value = input.total_spent / input.total_orders

    if bgf is not None and ggf is not None and lifetimes_frequency > 0:
        try:
            predicted_ltv = float(
                ggf.customer_lifetime_value(
                    bgf,
                    pd.Series([lifetimes_frequency]),
                    pd.Series([lifetimes_recency]),
                    pd.Series([T]),
                    pd.Series([avg_order_value]),
                    time=3, freq="D", discount_rate=0.01
                ).iloc[0]
            )
        except Exception as exc:
            logger.warning("LTV calculation failed, using fallback: %s", exc)
            predicted_ltv = estimate_predicted_ltv(first, last, input.total_orders, input.total_spent, today)
    else:
        predicted_ltv = estimate_predicted_ltv(first, last, input.total_orders, input.total_spent, today)

    features = pd.DataFrame([{
        "Frequency": frequency_rfm,
        "Monetary": monetary_rfm,
        "predicted_ltv": predicted_ltv
    }])

    churn_prob = float(model.predict_proba(features)[0][1])
    action = recommend_action(churn_prob, predicted_ltv)

    logger.info(
        "Prediction succeeded for total_orders=%s total_spent=%s -> action=%s",
        input.total_orders,
        input.total_spent,
        action["code"],
    )

    return {
        "frequency": frequency_rfm,
        "monetary": monetary_rfm,
        "predicted_ltv": round(predicted_ltv, 2),
        "churn_probability": churn_prob,
        "action": action,
        "action_code": action["code"],
        "action_label": action["label"],
    }