import pytest
import pandas as pd
import numpy as np
from src.features.engineer import FeatureEngineer

@pytest.fixture
def mock_data():
    train = pd.DataFrame({
        "customer_id": [1, 1, 2, 2],
        "product_unit_variant_id": [10, 10, 20, 20],
        "week_start": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-01", "2023-01-08"]),
        "qty_this_week": [5.0, 10.0, 0.0, 3.0]
    })
    test = pd.DataFrame({
        "customer_id": [1, 2],
        "product_unit_variant_id": [10, 20],
        "week_start": pd.to_datetime(["2023-01-15", "2023-01-15"])
    })
    return train, test

def test_engineer_features(mock_data):
    train, test = mock_data
    engineer = FeatureEngineer()
    train_out, test_out, features = engineer.engineer_features(train, test)
    
    assert "lag1" in features
    assert "roll_mean_4" in features
    assert "lag1" in train_out.columns
    assert "lag1" in test_out.columns
    assert len(train_out) == len(train)
    assert len(test_out) == len(test)

def test_generate_targets(mock_data):
    train, _ = mock_data
    engineer = FeatureEngineer()
    train_out = engineer.generate_targets(train)
    
    assert "target_qty_1w" in train_out.columns
    assert "target_buy_1w" in train_out.columns
