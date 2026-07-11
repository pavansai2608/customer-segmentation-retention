import os
from datetime import datetime
from pathlib import Path

import dill
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_churn_model.pkl"
DATA_PATH = BASE_DIR.parent / "models" / "final_decision_matrix.csv"
BGF_PATH = BASE_DIR.parent / "models" / "bgf_model.pkl"
GGF_PATH = BASE_DIR.parent / "models" / "ggf_model.pkl"

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:3000").split(",")

app = FastAPI(title="Customer Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load trained model and decision matrix ──
try:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Required model/data file missing: {e}") from e

# ── Load BG/NBD and Gamma-Gamma LTV models ──
try:
    with open(BGF_PATH, "rb") as f:
        bgf = dill.load(f)
    with open(GGF_PATH, "rb") as f:
        ggf = dill.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Required BG/NBD or Gamma-Gamma model missing: {e}") from e

# Features the XGBoost model was actually trained on (see notebook cell 40)
FEATURE_COLUMNS = ["Frequency", "Monetary", "predicted_ltv"]


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


# 2. Get all segments summary
@app.get("/segments")
def get_segments():
    summary = df["Segment"].value_counts().reset_index()
    summary.columns = ["segment", "count"]
    return summary.to_dict(orient="records")


# 3. Get priority action summary
@app.get("/actions")
def get_actions():
    summary = df["action"].value_counts().reset_index()
    summary.columns = ["action", "count"]
    return summary.to_dict(orient="records")


# 4. Get top customers to retain immediately
@app.get("/retain")
def get_retain_customers():
    retain = df[df["action"].str.contains("Retain", na=False)]
    retain = retain.sort_values("predicted_ltv", ascending=False).head(20)
    return retain[["Customer ID", "Frequency", "Monetary",
                    "predicted_ltv", "churn_probability",
                    "Segment", "action"]].to_dict(orient="records")


# 5. Search a specific customer (looks up existing customer, re-runs live inference)
@app.get("/customer/{customer_id}")
def get_customer(customer_id: float):
    row = df[df["Customer ID"] == customer_id]
    if row.empty:
        return {"error": "Customer not found"}

    features = row[FEATURE_COLUMNS]
    live_churn_prob = float(model.predict_proba(features)[0][1])

    result = row.iloc[0].to_dict()
    result["churn_probability"] = live_churn_prob
    return result


# 6. Predict churn for a brand-new customer using simple, human-friendly inputs
@app.post("/predict")
def predict_customer(input: CustomerInput):
    first = datetime.strptime(input.first_purchase_date, "%Y-%m-%d")
    last = datetime.strptime(input.last_purchase_date, "%Y-%m-%d")
    today = datetime.now()

    frequency_rfm = input.total_orders
    monetary_rfm = input.total_spent

    lifetimes_frequency = max(input.total_orders - 1, 0)
    lifetimes_recency = (last - first).days
    T = (today - first).days
    avg_order_value = input.total_spent / input.total_orders

    if lifetimes_frequency > 0:
        predicted_ltv = ggf.customer_lifetime_value(
            bgf,
            pd.Series([lifetimes_frequency]),
            pd.Series([lifetimes_recency]),
            pd.Series([T]),
            pd.Series([avg_order_value]),
            time=3, freq="D", discount_rate=0.01
        ).iloc[0]
    else:
        predicted_ltv = 0.0

    features = pd.DataFrame([{
        "Frequency": frequency_rfm,
        "Monetary": monetary_rfm,
        "predicted_ltv": predicted_ltv
    }])

    churn_prob = float(model.predict_proba(features)[0][1])

    return {
        "frequency": frequency_rfm,
        "monetary": monetary_rfm,
        "predicted_ltv": round(predicted_ltv, 2),
        "churn_probability": churn_prob
    }