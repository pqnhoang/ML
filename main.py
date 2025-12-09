from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os
import sys
import logging

# Setup logging để hiển thị trong Railway logs
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Prediction API", version="1.0.0")

# Global variables để store model và encoder
model = None
encoder = None

@app.on_event("startup")
async def startup_event():
    """Load model khi app start"""
    global model, encoder
    try:
        logger.info("=" * 50)
        logger.info("Starting application...")
        logger.info(f"PORT environment variable: {os.environ.get('PORT', 'NOT SET')}")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        logger.info("Loading model files...")
        
        model = joblib.load("lead_win_model_xgb.pkl")
        encoder = joblib.load("encoder.pkl")
        logger.info("✅ Model and encoder loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Encoder type: {type(encoder)}")
        logger.info("=" * 50)
        logger.info("🚀 Application ready to accept requests!")
    except FileNotFoundError as e:
        logger.error(f"❌ Error: Model file not found - {e}")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Files in directory: {os.listdir('.')}")
        # Không raise để app vẫn có thể start, nhưng predict sẽ fail
        logger.warning("⚠️ App will start but predict endpoint will fail")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Không raise để app vẫn có thể start
        logger.warning("⚠️ App will start but predict endpoint will fail")

# Cột categorical ban đầu dùng để encode
cat_cols = [
    "customer_region","source_id","campaign_id",
    "salesperson_id","team_id","stage_id","stage_sequence"
]


@app.get("/health")
def health():
    """Health check endpoint cho Railway"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None
    }

@app.get("/")
def root():
    """Root endpoint với thông tin về API"""
    return {
        "status": "API OK - model loaded",
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }

@app.post("/predict")
def predict(data: dict):
    if model is None or encoder is None:
        logger.error("Model or encoder not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)