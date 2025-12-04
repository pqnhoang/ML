from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("lead_win_model_xgb.pkl")
encoder = joblib.load("encoder.pkl")

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

    # Feature engineering giống training
    df["rev_log"] = np.log1p(df["expected_revenue"])
    df["rev_per_day"] = df["expected_revenue"] / (df["lead_age_days"] + 1)

    # Target encoding
    df[cat_cols] = encoder.transform(df[cat_cols])

    # Predict
    prob = model.predict_proba(df)[0][1]
    label = int(prob >= 0.5)

    return {
        "predicted_prob": float(prob),
        "predicted_label": label
    }

