"""
pydantic_models.py
Request and response schemas for the credit risk API.

Pydantic validates that incoming data has the correct
types and ranges before the model sees it — preventing
invalid predictions and providing clear error messages.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """
    Input schema for credit risk prediction.
    Matches the features produced by the pipeline.
    """
    # Transaction volume
    TotalTransactions:  float = Field(..., ge=0,
        description="Total number of transactions")
    TotalAmount:        float = Field(...,
        description="Sum of all transaction amounts (UGX)")
    AvgAmount:          float = Field(...,
        description="Average transaction amount (UGX)")
    StdAmount:          float = Field(0.0, ge=0,
        description="Std deviation of transaction amounts")
    MaxAmount:          float = Field(...,
        description="Largest single transaction (UGX)")
    MinAmount:          float = Field(...,
        description="Smallest single transaction (UGX)")
    MedianAmount:       float = Field(...,
        description="Median transaction amount (UGX)")

    # Value metrics
    TotalValue:         float = Field(..., ge=0,
        description="Sum of absolute transaction values")
    AvgValue:           float = Field(..., ge=0,
        description="Average absolute transaction value")
    MaxValue:           float = Field(..., ge=0,
        description="Maximum absolute transaction value")

    # Diversity
    UniqueProducts:     int   = Field(..., ge=1,
        description="Number of unique products purchased")
    UniqueCategories:   int   = Field(..., ge=1,
        description="Number of unique product categories")
    UniqueProviders:    int   = Field(..., ge=1,
        description="Number of unique providers used")
    UniqueChannels:     int   = Field(..., ge=1,
        description="Number of unique channels used")
    UniqueBatches:      int   = Field(..., ge=1,
        description="Number of unique batch IDs")

    # Fraud signals
    FraudCount:         int   = Field(0, ge=0,
        description="Number of fraudulent transactions")
    FraudRate:          float = Field(0.0, ge=0, le=1,
        description="Proportion of fraudulent transactions")

    # Temporal
    TenureDays:         int   = Field(..., ge=0,
        description="Days between first and last transaction")
    TransactionVelocity: float = Field(..., ge=0,
        description="Transactions per day")

    # Debit/credit split
    PositiveCount:      int   = Field(..., ge=0,
        description="Number of debit transactions")
    NegativeCount:      int   = Field(..., ge=0,
        description="Number of credit transactions")
    PositiveAmountRatio: float = Field(..., ge=0, le=1,
        description="Proportion of debit transactions")

    # RFM features
    Recency:            int   = Field(..., ge=0,
        description="Days since last transaction")
    Frequency:          int   = Field(..., ge=1,
        description="Total transaction count")
    Monetary:           float = Field(..., ge=0,
        description="Total positive spend (UGX)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "Monetary": 45000.0
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Output schema for credit risk prediction.
    """
    customer_id:        Optional[str] = Field(
        None, description="Customer ID if provided")
    risk_probability:   float = Field(...,
        description="Probability of being high-risk (0-1)")
    is_high_risk:       bool  = Field(...,
        description="Binary risk classification")
    risk_label:         str   = Field(...,
        description="Human-readable risk label")
    model_version:      str   = Field(...,
        description="Model version used for prediction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CustomerId_123",
                "risk_probability": 0.73,
                "is_high_risk": True,
                "risk_label": "HIGH RISK",
                "model_version": "LogisticRegression_v1"
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status:        str
    model_loaded:  bool
    model_version: str
