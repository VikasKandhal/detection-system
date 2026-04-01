"""
Pydantic schemas for the Fraud Detection API.
Defines request/response models for type validation and documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification based on fraud probability."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransactionInput(BaseModel):
    """
    Input schema for a single transaction to be scored.
    
    This schema captures the key fields from the IEEE dataset.
    In production, you'd adapt this to your actual transaction format.
    """
    TransactionAmt: float = Field(..., description="Transaction amount in USD")
    ProductCD: str = Field(..., description="Product code (W, H, C, S, R)")
    card1: int = Field(..., description="Card identifier (hashed)")
    card2: Optional[float] = Field(None, description="Card detail 2")
    card3: Optional[float] = Field(None, description="Card detail 3")
    card4: Optional[str] = Field(None, description="Card brand (visa, mastercard, etc)")
    card5: Optional[float] = Field(None, description="Card detail 5")
    card6: Optional[str] = Field(None, description="Card type (debit, credit)")
    addr1: Optional[float] = Field(None, description="Billing address part 1")
    addr2: Optional[float] = Field(None, description="Billing address part 2")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    DeviceType: Optional[str] = Field(None, description="Device type (mobile, desktop)")
    DeviceInfo: Optional[str] = Field(None, description="Device information")
    
    # Time-related
    TransactionDT: Optional[int] = Field(None, description="Transaction time delta (seconds)")
    
    # V-features (anonymous engineered features from Vesta)
    # In production, include all V1-V339 that your model uses
    
    # Additional features can be passed as a dict
    additional_features: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional features as key-value pairs"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 75.0,
                "ProductCD": "W",
                "card1": 13926,
                "card2": 361.0,
                "card3": 150.0,
                "card4": "visa",
                "card5": 226.0,
                "card6": "debit",
                "addr1": 299.0,
                "addr2": 87.0,
                "P_emaildomain": "gmail.com",
                "R_emaildomain": None,
                "DeviceType": "desktop",
                "DeviceInfo": "Windows",
                "TransactionDT": 86400
            }
        }


class RiskFactor(BaseModel):
    """A single risk factor contributing to the fraud decision."""
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value")
    shap_value: float = Field(..., description="SHAP contribution to prediction")
    direction: str = Field(..., description="'increases' or 'decreases' fraud risk")
    impact: str = Field(..., description="Human-readable impact description")


class PredictionResponse(BaseModel):
    """Response schema for a fraud prediction."""
    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    fraud_probability: float = Field(..., description="Probability of fraud (0 to 1)")
    is_fraud: bool = Field(..., description="Binary fraud decision based on threshold")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    threshold_used: float = Field(..., description="Decision threshold used")
    model_used: str = Field(..., description="Name of the model used")
    
    # Explainability
    top_risk_factors: List[RiskFactor] = Field(
        default=[], description="Top features contributing to the decision"
    )
    explanation: List[str] = Field(
        default=[], description="Human-readable explanation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_001",
                "fraud_probability": 0.87,
                "is_fraud": True,
                "risk_level": "critical",
                "threshold_used": 0.5,
                "model_used": "LightGBM",
                "top_risk_factors": [
                    {
                        "feature": "TransactionAmt",
                        "value": 999.99,
                        "shap_value": 0.34,
                        "direction": "increases",
                        "impact": "TransactionAmt=999.99 increases fraud risk by 0.34"
                    }
                ],
                "explanation": [
                    "⚠️ TransactionAmt = 999.99 (increases fraud risk)",
                    "⚠️ card1 = 13926 (increases fraud risk)"
                ]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    transactions: List[TransactionInput] = Field(
        ..., description="List of transactions to score"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_transactions: int
    flagged_count: int
    average_fraud_probability: float


class ModelInfo(BaseModel):
    """Information about the deployed model."""
    model_name: str
    model_type: str
    optimal_threshold: float
    training_metrics: Dict[str, float]
    feature_count: int
    training_date: Optional[str] = None


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    model_name: str
    version: str
