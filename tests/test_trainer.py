import pytest
import pandas as pd
import numpy as np
import os
import shutil
from src.models.trainer import ModelTrainer

@pytest.fixture
def mock_config():
    return {
        'model': {
            'seeds': [42],
            'n_estimators': 2,
            'learning_rate': 0.1,
            'num_leaves': 4,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 1
        },
        'catboost': {
            'iterations': 2,
            'learning_rate': 0.1,
            'depth': 2
        },
        'paths': {
            'model_dir': 'tests/output/models'
        }
    }

@pytest.fixture
def mock_train_data():
    features = ["lag1", "lag2", "cat_col"]
    cat_cols = ["cat_col"]
    
    train = pd.DataFrame({
        "lag1": [1.0, 2.0, 0.0, 5.0],
        "lag2": [0.0, 1.0, 2.0, 1.0],
        "cat_col": [0, 1, 0, 1],
        "target_buy_1w": [1, 1, 0, 1],
        "target_buy_2w": [0, 1, 0, 1],
        "target_qty_1w": [5.0, 2.0, 0.0, 1.0],
        "target_qty_2w": [0.0, 3.0, 0.0, 4.0]
    })
    return train, features, cat_cols

def test_train_hybrid_ensemble(mock_config, mock_train_data):
    train, features, cat_cols = mock_train_data
    trainer = ModelTrainer(mock_config)
    
    models = trainer.train_hybrid_ensemble(train, features, cat_cols)
    
    assert 'lgb_clf1' in models
    assert 'cb_clf1' in models
    assert len(models['lgb_clf1']) == 1
    assert os.path.exists("tests/output/models/hybrid_ensemble.pkl")
    
    # Cleanup
    if os.path.exists("tests/output"):
        shutil.rmtree("tests/output")
