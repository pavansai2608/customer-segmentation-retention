"""
Rebuilds final_decision_matrix.csv and trains all models from raw transaction data.

This script reproduces the pipeline originally developed in
notebooks/Customer Segmentation.ipynb, so the project doesn't depend on
manually re-running notebook cells to regenerate its data/model artifacts.

Usage:
    python src/build_decision_matrix.py

Expects:
    data/raw/online_retail_II.csv   (raw transaction export)

Produces:
    data/processed/cleaned_retail.csv
    models/xgb_churn_model.pkl
    models/lr_churn_model.pkl
    models/bgf_model.pkl
    models/ggf_model.pkl
    models/final_decision_matrix.csv
"""

import datetime as dt
from pathlib import Path

import dill
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "online_retail_II.csv"
CLEANED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"
MODELS_DIR = BASE_DIR / "models"

JUNK_STOCK_CODES = [
    "POST", "DOT", "M", "m", "D", "C2", "S", "BANK CHARGES",
    "AMAZONFEE", "CRUK", "PADS",
]

# Cluster index -> business label, from the K-Means fit in the original notebook.
# NOTE: cluster indices are not stable across re-runs with different data.
# If you retrain and the resulting cluster_summary averages look different from
# what these labels describe, update this mapping to match (see the printed
# cluster_summary output when you run this script).
CLUSTER_LABELS = {
    2: "Champions",        # recent, very frequent, highest spend
    3: "Loyal Customers",  # recent, frequent, high spend
    0: "At Risk",          # moderate recency, low frequency
    1: "Hibernating",      # bought long ago, rarely, low spend
}

FEATURE_COLUMNS = ["Frequency", "Monetary", "predicted_ltv"]


def load_and_clean_data() -> pd.DataFrame:
    print(f"Loading raw data from {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1")

    df = df.dropna(subset=["Customer ID"])

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    df = df[~df["StockCode"].astype(str).str.upper().isin(
        [c.upper() for c in JUNK_STOCK_CODES]
    )]
    df = df[~df["StockCode"].astype(str).str.startswith("gift_", na=False)]

    df = df.drop_duplicates()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    print(f"Cleaned data shape: {df.shape}")

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Saved cleaned data to {CLEANED_DATA_PATH}")

    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "TotalPrice": "sum",
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    )

    return rfm


def fit_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = km_final.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    print("\nCluster averages (check these against CLUSTER_LABELS mapping):")
    print(cluster_summary.round(2))

    rfm["Segment"] = rfm["Cluster"].map(CLUSTER_LABELS)
    print("\nSegment distribution:")
    print(rfm["Segment"].value_counts())

    return rfm


def fit_ltv_models(df: pd.DataFrame, rfm: pd.DataFrame):
    bgf_data = summary_data_from_transaction_data(
        df,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=df["InvoiceDate"].max(),
    )

    bgf = BetaGeoFitter(penalizer_coef=0.5)
    bgf.fit(bgf_data["frequency"], bgf_data["recency"], bgf_data["T"])
    print("\nBG/NBD model fitted.")

    returning_customers = bgf_data[bgf_data["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(returning_customers["frequency"], returning_customers["monetary_value"])
    print("Gamma-Gamma model fitted.")

    ltv = ggf.customer_lifetime_value(
        bgf,
        returning_customers["frequency"],
        returning_customers["recency"],
        returning_customers["T"],
        returning_customers["monetary_value"],
        time=3, freq="D", discount_rate=0.01,
    )
    ltv_df = ltv.reset_index()
    ltv_df.columns = ["Customer ID", "predicted_ltv"]

    rfm_ltv = rfm.reset_index().merge(ltv_df, on="Customer ID", how="left")
    rfm_ltv["predicted_ltv"] = rfm_ltv["predicted_ltv"].fillna(0.0)

    return rfm_ltv, bgf, ggf


def add_churn_label(df: pd.DataFrame, rfm_ltv: pd.DataFrame) -> pd.DataFrame:
    last_date = df["InvoiceDate"].max()
    last_purchase = df.groupby("Customer ID")["InvoiceDate"].max().reset_index()
    last_purchase.columns = ["Customer ID", "last_purchase_date"]

    rfm_ltv = rfm_ltv.merge(last_purchase, on="Customer ID", how="left")
    rfm_ltv["churned"] = (
        (last_date - rfm_ltv["last_purchase_date"]).dt.days > 90
    ).astype(int)

    print(f"\nChurn rate: {rfm_ltv['churned'].mean() * 100:.1f}%")
    return rfm_ltv


def train_models(rfm_ltv: pd.DataFrame):
    X = rfm_ltv[FEATURE_COLUMNS]
    y = rfm_ltv["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_balanced, y_train_balanced)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, lr_model.predict(X_test)))
    print("ROC-AUC:", round(roc_auc_score(y_test, lr_proba), 4))

    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, eval_metric="logloss",
    )
    xgb_model.fit(X_train_balanced, y_train_balanced)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    print("\n=== XGBoost ===")
    print(classification_report(y_test, xgb_model.predict(X_test)))
    print("ROC-AUC:", round(roc_auc_score(y_test, xgb_proba), 4))

    # Full-dataset churn probability, used in the final decision matrix
    rfm_ltv["churn_probability"] = xgb_model.predict_proba(X)[:, 1]

    return lr_model, xgb_model, rfm_ltv


def build_action_column(rfm_ltv: pd.DataFrame) -> pd.DataFrame:
    ltv_median = rfm_ltv["predicted_ltv"].median()

    def recommend(row):
        high_risk = row["churn_probability"] > 0.5
        high_ltv = row["predicted_ltv"] > ltv_median
        if high_risk and high_ltv:
            return "🔴 Retain Immediately"
        elif high_risk and not high_ltv:
            return "⚪ Let Go"
        elif not high_risk and high_ltv:
            return "🟢 Nurture"
        else:
            return "🔵 Monitor"

    rfm_ltv["action"] = rfm_ltv.apply(recommend, axis=1)
    return rfm_ltv


def save_models(lr_model, xgb_model, bgf, ggf):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_model, MODELS_DIR / "xgb_churn_model.pkl")
    joblib.dump(lr_model, MODELS_DIR / "lr_churn_model.pkl")
    print(f"\nSaved xgb_churn_model.pkl and lr_churn_model.pkl to {MODELS_DIR}")

    # bgf/ggf contain lambdas from the lifetimes library's internal optimizer,
    # which plain pickle/joblib cannot serialize — use dill instead.
    with open(MODELS_DIR / "bgf_model.pkl", "wb") as f:
        dill.dump(bgf, f)
    with open(MODELS_DIR / "ggf_model.pkl", "wb") as f:
        dill.dump(ggf, f)
    print(f"Saved bgf_model.pkl and ggf_model.pkl to {MODELS_DIR}")


def main():
    df = load_and_clean_data()
    rfm = compute_rfm(df)
    rfm = fit_segments(rfm)
    rfm_ltv, bgf, ggf = fit_ltv_models(df, rfm)
    rfm_ltv = add_churn_label(df, rfm_ltv)
    lr_model, xgb_model, rfm_ltv = train_models(rfm_ltv)
    rfm_ltv = build_action_column(rfm_ltv)

    save_models(lr_model, xgb_model, bgf, ggf)

    output_path = MODELS_DIR / "final_decision_matrix.csv"
    rfm_ltv.to_csv(output_path, index=False)
    print(f"\nSaved final decision matrix to {output_path}")
    print(f"Total customers processed: {len(rfm_ltv)}")


if __name__ == "__main__":
    main()