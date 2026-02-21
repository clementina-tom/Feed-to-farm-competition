"""
Model Training and Inference components.

Contains the Hybrid Ensemble logic (LGBM + CatBoost) for 5-seed training,
as well as decoupled calibration and quantity thresholds for prediction.
"""
from .trainer import ModelTrainer
from .predictor import ModelPredictor

__all__ = ["ModelTrainer", "ModelPredictor"]
