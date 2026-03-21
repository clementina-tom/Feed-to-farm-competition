import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_no_model(client, monkeypatch):
    import src.api
    monkeypatch.setattr(src.api, "MODELS", None)
    
    response = client.post("/predict", json={
        "customer_id": 123,
        "product_unit_variant_id": 456
    })
    assert response.status_code == 503
    assert "Models not loaded" in response.json()["detail"]
