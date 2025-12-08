from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os


app = FastAPI()

# Load model + encoder
model = joblib.load("lead_win_model_xgb.pkl")
encoder = joblib.load("encoder.pkl")

# Cột categorical ban đầu dùng để encode
cat_cols = [
    "customer_region","source_id","campaign_id",
    "salesperson_id","team_id","stage_id","stage_sequence"
]

@app.get("/")
def root():
    return {"status": "API OK - model loaded"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Fill NA
    num_cols = ["expected_revenue","probability","lead_age_days","priority"]
    df[num_cols] = df[num_cols].fillna(0)

    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # Feature engineering GIỐNG HỆ TRAINING (18 features)
    df["rev_log"] = np.log1p(df["expected_revenue"])
    df["rev_per_day"] = df["expected_revenue"] / (df["lead_age_days"] + 1)

    # Model yêu cầu thêm 3 cột nữa:
    df["log_expected_revenue"] = df["rev_log"]
    df["rev_per_day_age"] = df["rev_per_day"]
    df["create_dayofweek"] = df.get("create_dow", 0)  # fallback

    # Nếu thiếu create_month or create_dow
    if "create_month" not in df:
        df["create_month"] = 1
    if "create_dow" not in df:
        df["create_dow"] = 0

    # Apply target encoder
    df[cat_cols] = encoder.transform(df[cat_cols])

    # Dùng đúng thứ tự cột mà model đòi hỏi
    ordered_cols = [
        'lead_age_days','expected_revenue','probability','stage_id',
        'stage_sequence','source_id','campaign_id','salesperson_id',
        'team_id','customer_region','priority','create_month',
        'create_dayofweek','log_expected_revenue','rev_per_day_age',
        'rev_log','rev_per_day','create_dow'
    ]

    df = df[ordered_cols]

    # Predict
    prob = model.predict_proba(df)[0][1]
    label = int(prob >= 0.5)

    return {
        "predicted_prob": float(prob),
        "predicted_label": label
    }