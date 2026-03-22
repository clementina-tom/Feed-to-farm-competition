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
    """Generate enough data for CatBoost to train without crashing."""
    np.random.seed(42)
    n = 50  # CatBoost needs more samples than 4 for stable training
    features = ["lag1", "lag2", "cat_col"]
    cat_cols = ["cat_col"]

    train = pd.DataFrame({
        "lag1": np.random.rand(n) * 10,
        "lag2": np.random.rand(n) * 5,
        "cat_col": np.random.choice([0, 1, 2], size=n),
        "target_buy_1w": np.random.choice([0, 1], size=n),
        "target_buy_2w": np.random.choice([0, 1], size=n),
        "target_qty_1w": np.random.rand(n) * 10,
        "target_qty_2w": np.random.rand(n) * 8,
    })
    # Ensure both classes are present for classification
    train.loc[0, "target_buy_1w"] = 0
    train.loc[1, "target_buy_1w"] = 1
    train.loc[0, "target_buy_2w"] = 0
    train.loc[1, "target_buy_2w"] = 1

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
