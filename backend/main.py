import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_churn_model.pkl"
DATA_PATH = BASE_DIR.parent / "models" / "final_decision_matrix.csv"

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:3000").split(",")

app = FastAPI(title="Customer Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Required model/data file missing: {e}") from e

# Load trained model and decision matrix
model = joblib.load("../models/xgb_churn_model.pkl")
df = pd.read_csv("../models/final_decision_matrix.csv")

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

# 5. Search a specific customer
@app.get("/customer/{customer_id}")
def get_customer(customer_id: float):
    customer = df[df["Customer ID"] == customer_id]
    if customer.empty:
        return {"error": "Customer not found"}
    return customer.to_dict(orient="records")[0]