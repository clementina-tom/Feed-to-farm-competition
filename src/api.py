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

# Initialize FastAPI app
app = FastAPI(title="Feed-to-Farm Prediction API", 
              description="Real-time purchasing recommendations for produce.")

# Load Config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global variables for models and predictors
MODELS = None
PREDICTOR = ModelPredictor(config)
FE_ENGINEER = FeatureEngineer()

@app.on_event("startup")
def load_models():
    global MODELS
    model_path = os.path.join(config['paths']['model_dir'], 'hybrid_ensemble.pkl')
    if os.path.exists(model_path):
        MODELS = joblib.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Predict endpoint will fail.")

class PurchaseRequest(BaseModel):
    # This expects the raw columns needed for feature engineering
    # For a real API, you might want to fetch some of this from a DB instead of passing it all
    customer_id: int
    product_unit_variant_id: int
    # ... any other immediate fields if needed by FE_ENGINEER ...
    # Note: In a production setting, the API would likely query the last X weeks of history 
    # from a DB rather than requiring the caller to provide all lag features.
    historical_data: List[dict] # List of weeks with 'week_start' and 'qty_this_week'

class PredictionResponse(BaseModel):
    buy_1w_prob: float
    buy_2w_prob: float
    qty_1w: float
    qty_2w: float

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": MODELS is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PurchaseRequest):
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # 1. Convert historical data to DataFrame
    df = pd.DataFrame(request.historical_data)
    df['customer_id'] = request.customer_id
    df['product_unit_variant_id'] = request.product_unit_variant_id
    df['week_start'] = pd.to_datetime(df['week_start'])
    
    # 2. Run feature engineering (minimal subset needed for inference)
    # This is a simplification; in production, you'd apply the same FeatureEngineer transforms
    # to the input window provided in the request.
    try:
        # Note: For real-time, we usually just need the features for the LAST week
        # Here we mock the process since full-scale FE needs the whole train context
        # In a real portfolio app, you'd show how you derive features for a single request.
        
        # We'll use the predictor logic directly for the demonstration
        # Usually, you'd pass the transformed row to PREDICTOR.predict()
        
        # This is a placeholder for the logic:
        # 1. Transform request to feature vector
        # 2. Call PREDICTOR.predict() on that vector
        
        return {
            "buy_1w_prob": 0.85, # placeholder values for logic demo
            "buy_2w_prob": 0.72,
            "qty_1w": 5.5,
            "qty_2w": 4.2
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
