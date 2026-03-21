from fastapi import FastAPI, HTTPException
import joblib
import yaml
import pandas as pd
import numpy as np
import os
from pydantic import BaseModel
from typing import List, Optional
from src.features.engineer import FeatureEngineer
from src.models.predictor import ModelPredictor

app = FastAPI(title="Feed-to-Farm Prediction API", 
              description="Real-time purchasing recommendations for produce.")

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODELS = None
PREDICTOR = ModelPredictor(config)
FE_ENGINEER = FeatureEngineer()

@app.on_event("startup")
def load_models():
    global MODELS
    model_path = os.path.join(config['paths']['model_dir'], 'hybrid_ensemble.pkl')
    if os.path.exists(model_path):
        MODELS = joblib.load(model_path)
        print(f"Models loaded successfully from {model_path}.")
    else:
        print(f"Warning: Model not found at {model_path}. Predict endpoint will fail.")

class PurchaseRequest(BaseModel):
    customer_id: int
    product_unit_variant_id: int

class PredictionResponse(BaseModel):
    buy_1w_prob: float
    buy_2w_prob: float
    qty_1w: float
    qty_2w: float

def get_latest_features(customer_id: int, product_unit_variant_id: int) -> pd.DataFrame:
    # In production, this would query a feature store or database.
    # We generate a dummy row with expected feature columns to pass to the predictor.
    feature_cols = [
        "lag1", "lag2", "roll_mean_4",
        "cust_lag1", "cust_roll_4",
        "global_lag1", "global_roll_4",
        "pair_buy_rate", "pair_recency",
        "is_new_pair", "month", "week_of_year"
    ]
    # We add dummy categorical encoded columns assuming label encoding returned 0
    cat_cols = ["customer_category_x", "customer_status_x", "grade_name", "unit_name", "customer_category", "customer_status"]
    # Provide safe minimal defaults
    data = {c: 0.0 for c in feature_cols}
    for c in cat_cols:
         data[c] = 0
    data["customer_id"] = customer_id
    data["product_unit_variant_id"] = product_unit_variant_id
    data["ID"] = f"{customer_id}X{product_unit_variant_id}"
    
    return pd.DataFrame([data]), feature_cols + cat_cols

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": MODELS is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PurchaseRequest):
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train the model first.")
    
    try:
        # 1. Fetch live features (mocked for now)
        df, all_features = get_latest_features(request.customer_id, request.product_unit_variant_id)
        
        # 2. Call PREDICTOR
        submission = PREDICTOR.predict(MODELS, df, all_features)
        
        row = submission.iloc[0]
        return PredictionResponse(
            buy_1w_prob=float(row["Target_purchase_next_1w"]),
            buy_2w_prob=float(row["Target_purchase_next_2w"]),
            qty_1w=float(row["Target_qty_next_1w"]),
            qty_2w=float(row["Target_qty_next_2w"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
