import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid(client):
    payload = {
        "MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.9,
        "AveBedrms": 1.0, "Population": 322, "AveOccup": 2.5,
        "Latitude": 37.88, "Longitude": -122.23
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_predict_invalid(client):
    response = client.post("/predict", json={"MedInc": -1})
    assert response.status_code == 422  # Validation error