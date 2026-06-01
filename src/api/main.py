"""
main.py
FastAPI application for Bati Bank credit risk scoring.

Endpoints:
  GET  /health   -> service health check
  POST /predict  -> credit risk prediction

The model is loaded from disk at startup and reused
for all requests — no reloading per request.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import logging
import os

from src.api.pydantic_models import (
    CustomerFeatures,
    PredictionResponse,
    HealthResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
# Loaded once at startup, shared across all requests
model = None
MODEL_VERSION = "LogisticRegression_v1"
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")

# Exact column order used during training — must match fit() order
FEATURE_COLUMNS = [
    'TotalTransactions', 'UniqueBatches', 'TotalAmount', 'AvgAmount',
    'StdAmount', 'MaxAmount', 'MinAmount', 'MedianAmount', 'TotalValue',
    'AvgValue', 'MaxValue', 'UniqueProducts', 'UniqueCategories',
    'UniqueProviders', 'UniqueChannels', 'FraudCount', 'PositiveCount',
    'NegativeCount', 'FraudRate', 'PositiveAmountRatio', 'TenureDays',
    'TransactionVelocity', 'Recency', 'Frequency', 'Monetary'
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    logger.info(f"Loading model from: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded: {type(model).__name__}")
    except FileNotFoundError:
        logger.error(
            f"Model file not found: {MODEL_PATH}. "
            f"Run python src/train.py first."
        )
    yield
    logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description=(
        "Predicts credit risk probability for buy-now-pay-later "
        "applicants using behavioral transaction features. "
        "Built for Bati Bank's eCommerce lending partnership."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Returns model load status and version.
    """
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=MODEL_VERSION
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: CustomerFeatures,
    customer_id: str = None
):
    """
    Predict credit risk for a new customer.

    Accepts customer behavioral features and returns:
    - risk_probability: float between 0 and 1
    - is_high_risk: boolean classification
    - risk_label: human-readable label
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first."
        )

    try:
        # Convert input to DataFrame, enforcing training column order
        input_data = pd.DataFrame([features.model_dump()])
        input_data = input_data[FEATURE_COLUMNS]

        # Get probability of high risk (class 1)
        prob = model.predict_proba(input_data)[0][1]
        is_high_risk = bool(prob >= 0.5)
        risk_label = "HIGH RISK" if is_high_risk else "LOW RISK"

        logger.info(
            f"Prediction: {risk_label} "
            f"(probability={prob:.4f})"
        )

        return PredictionResponse(
            customer_id=customer_id,
            risk_probability=round(float(prob), 4),
            is_high_risk=is_high_risk,
            risk_label=risk_label,
            model_version=MODEL_VERSION
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/")
def root():
    """Root endpoint — API info."""
    return {
        "name": "Bati Bank Credit Risk API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }
