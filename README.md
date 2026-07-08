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

## Project Structure
customer-segmentation-retention/
├── backend/          # FastAPI app
├── frontend/         # React dashboard
├── data/             # Raw and processed data (DVC tracked)
├── models/           # Trained model artifacts
├── notebooks/        # Colab notebooks
└── src/              # Reusable Python scripts

## How to Run

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

## Experiments
View all MLflow experiment runs on DagsHub:
[https://dagshub.com/pavansai2608/customer-segmentation-retention/experiments](https://dagshub.com/pavansai2608/customer-segmentation-retention/experiments)