"""
FastAPI Application for the Fraud Detection System.
Provides REST API endpoints for real-time fraud scoring.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from typing import List
import numpy as np

from api.schemas import (
    TransactionInput, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, ModelInfo, HealthResponse, RiskFactor
)
from api.model_service import FraudModelService
from src.utils import logger

# ==============================================================================
# Application Lifecycle
# ==============================================================================

# Global model service
model_service = FraudModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Starting Fraud Detection API...")
    try:
        model_service.load()
        if model_service._demo_mode:
            logger.info("API started in DEMO mode — predictions use heuristic scoring.")
        else:
            logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API starting without model. /predict will not work.")
    yield
    logger.info("Shutting down Fraud Detection API...")


# ==============================================================================
# FastAPI App
# ==============================================================================

app = FastAPI(
    title="FraudShield AI - Fraud Detection API",
    description=(
        "Production-grade fraud detection system using ensemble ML models. "
        "Provides real-time transaction scoring with SHAP-based explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded,
        model_name=model_service.model_name or "not loaded",
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionInput):
    """
    Score a single transaction for fraud.
    
    Returns fraud probability, binary decision, risk level,
    and SHAP-based explanation of why the transaction was flagged.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization."
        )

    start_time = time.time()

    try:
        # Convert Pydantic model to dict
        transaction_data = transaction.model_dump(exclude_none=True)
        
        # Remove 'additional_features' and merge them into main dict
        additional = transaction_data.pop("additional_features", None)
        if additional:
            transaction_data.update(additional)

        # Get prediction
        result = model_service.predict(transaction_data)

        # Format risk factors
        risk_factors = []
        for rf in result.get("top_risk_factors", []):
            risk_factors.append(RiskFactor(
                feature=rf["feature"],
                value=float(rf["value"]),
                shap_value=float(rf["shap_value"]),
                direction=rf["direction"],
                impact=rf["impact"]
            ))

        latency = time.time() - start_time
        logger.info(
            f"Prediction: prob={result['fraud_probability']:.4f}, "
            f"fraud={result['is_fraud']}, risk={result['risk_level']}, "
            f"latency={latency*1000:.1f}ms"
        )

        return PredictionResponse(
            fraud_probability=result["fraud_probability"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            threshold_used=result["threshold_used"],
            model_used=result["model_used"],
            top_risk_factors=risk_factors,
            explanation=result.get("explanation", [])
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Score multiple transactions in a batch.
    More efficient than calling /predict multiple times.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    predictions = []
    for txn in request.transactions:
        try:
            transaction_data = txn.model_dump(exclude_none=True)
            additional = transaction_data.pop("additional_features", None)
            if additional:
                transaction_data.update(additional)

            result = model_service.predict(transaction_data)
            
            risk_factors = [
                RiskFactor(**rf) for rf in result.get("top_risk_factors", [])
            ]

            predictions.append(PredictionResponse(
                fraud_probability=result["fraud_probability"],
                is_fraud=result["is_fraud"],
                risk_level=result["risk_level"],
                threshold_used=result["threshold_used"],
                model_used=result["model_used"],
                top_risk_factors=risk_factors,
                explanation=result.get("explanation", [])
            ))
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            predictions.append(PredictionResponse(
                fraud_probability=0.0,
                is_fraud=False,
                risk_level="low",
                threshold_used=model_service.threshold,
                model_used=model_service.model_name or "error",
                explanation=[f"Error: {str(e)}"]
            ))

    flagged = sum(1 for p in predictions if p.is_fraud)
    avg_prob = np.mean([p.fraud_probability for p in predictions])

    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        flagged_count=flagged,
        average_fraud_probability=float(avg_prob)
    )


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
async def get_model_info():
    """Get information about the deployed model."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Find metrics for current model
    training_metrics = {}
    if model_service.model_info:
        for info in model_service.model_info:
            if model_service.model_name and info.get("Model", "").lower().replace(" ", "_") in model_service.model_name:
                training_metrics = {
                    k: v for k, v in info.items() if k != "Model"
                }
                break
        if not training_metrics and model_service.model_info:
            training_metrics = {
                k: v for k, v in model_service.model_info[0].items() if k != "Model"
            }

    return ModelInfo(
        model_name=model_service.model_name,
        model_type=type(model_service.model).__name__,
        optimal_threshold=model_service.threshold,
        training_metrics=training_metrics,
        feature_count=len(model_service.feature_cols) if model_service.feature_cols else 0,
    )


# ==============================================================================
# Run directly
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
