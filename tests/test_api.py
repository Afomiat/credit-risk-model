"""
test_api.py
Integration tests for the Bati Bank credit risk FastAPI application.

Tests:
  - GET  /         → root info response
  - GET  /health   → model load status
  - POST /predict  → valid prediction round-trip
  - POST /predict  → validation error on bad input
"""

import sys
import os
import pytest

# Ensure project root is on path so src.api.main imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient for the FastAPI app.
    The lifespan context manager loads the model on startup.
    Skips all tests if the model file is missing (CI without model artifact).
    """
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "best_model.pkl"
    )
    if not os.path.exists(model_path):
        pytest.skip(
            "models/best_model.pkl not found — "
            "run python src/train.py to generate it first."
        )

    from src.api.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample valid input matching the 25-feature schema
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "TotalTransactions": 12,
    "TotalAmount": 45000.0,
    "AvgAmount": 3750.0,
    "StdAmount": 1200.0,
    "MaxAmount": 8000.0,
    "MinAmount": 500.0,
    "MedianAmount": 3000.0,
    "TotalValue": 50000.0,
    "AvgValue": 4166.0,
    "MaxValue": 8000.0,
    "UniqueProducts": 3,
    "UniqueCategories": 2,
    "UniqueProviders": 2,
    "UniqueChannels": 2,
    "UniqueBatches": 12,
    "FraudCount": 0,
    "FraudRate": 0.0,
    "TenureDays": 30,
    "TransactionVelocity": 0.4,
    "PositiveCount": 10,
    "NegativeCount": 2,
    "PositiveAmountRatio": 0.83,
    "Recency": 5,
    "Frequency": 12,
    "Monetary": 45000.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_root(client):
    """Root endpoint returns correct API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Bati Bank Credit Risk API"
    assert data["version"] == "1.0.0"
    assert "/docs" in data["docs"]
    assert "/health" in data["health"]
    assert "/predict" in data["predict"]


def test_health_check(client):
    """Health endpoint reports model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["status"] == "healthy"
    assert data["model_version"] != ""


def test_predict_valid_input(client):
    """Valid input returns a well-formed prediction response."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text

    data = response.json()

    # Check all required fields are present
    assert "risk_probability" in data
    assert "is_high_risk" in data
    assert "risk_label" in data
    assert "model_version" in data

    # Probability is in [0, 1]
    assert 0.0 <= data["risk_probability"] <= 1.0

    # Boolean
    assert isinstance(data["is_high_risk"], bool)

    # Label matches boolean
    if data["is_high_risk"]:
        assert data["risk_label"] == "HIGH RISK"
    else:
        assert data["risk_label"] == "LOW RISK"


def test_predict_missing_required_field(client):
    """Request missing required fields returns 422 Unprocessable Entity."""
    incomplete = VALID_PAYLOAD.copy()
    del incomplete["Recency"]  # required field

    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_value_type(client):
    """Non-numeric value for a numeric field returns 422."""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["TotalTransactions"] = "not-a-number"

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_negative_fraud_rate(client):
    """FraudRate below 0 violates ge=0 constraint → 422."""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["FraudRate"] = -0.5

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_fraud_rate_above_one(client):
    """FraudRate above 1 violates le=1 constraint → 422."""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["FraudRate"] = 1.5

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
