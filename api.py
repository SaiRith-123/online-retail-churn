"""
FastAPI real-time serving layer for churn predictions.

Start with: uvicorn api:app --reload
Access: POST http://127.0.0.1:8000/predict
"""
import logging
import os

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.predict import predict


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="Churn Prediction API",
    description="Real-time inference for online retail customer churn",
    version="1.0.0"
)


class Customer(BaseModel):
    """Customer features for churn prediction."""
    frequency: int = Field(gt=0)
    monetary: float = Field(gt=0)
    total_items: int = Field(gt=0)
    avg_basket_size: float = Field(gt=0)
    n_unique_products: int = Field(gt=0)
    country: str = Field(min_length=1)
    tenure_days: int = Field(ge=0)
    avg_days_between_purchases: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "frequency": 12,
                "monetary": 2500.50,
                "total_items": 287,
                "avg_basket_size": 23.92,
                "n_unique_products": 85,
                "country": "United Kingdom",
                "tenure_days": 365,
                "avg_days_between_purchases": 30
            }
        }
    }


@app.get("/")
def root():
    """API root endpoint."""
    return {
        "message": "Churn Prediction API",
        "usage": "POST /predict with customer features",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """Simple readiness endpoint for monitoring and smoke checks."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness endpoint with basic runtime metadata."""
    return {
        "status": "ok",
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        "default_model_stage": os.environ.get("DEFAULT_MODEL_STAGE", "Production"),
    }


@app.post("/predict")
def churn_predict(customer: Customer):
    """
    Predict churn probability for a customer.
    
    Args:
        customer: Customer features
    
    Returns:
        churn_probability: Float between 0 and 1
        churn_flag: Binary classification (0=retain, 1=churn)
    """
    logger.info("Received /predict request for country=%s", customer.country)
    df = pd.DataFrame([customer.model_dump()])
    try:
        result = predict(df)
        response = result.to_dict(orient="records")[0]
        logger.info("Completed /predict request with churn_probability=%s", response["churn_probability"])
        return response
    except Exception as exc:
        logger.exception("Prediction failed for /predict")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction service unavailable: {exc}",
        ) from exc


@app.post("/batch_predict")
def batch_churn_predict(customers: list[Customer]):
    """
    Predict churn for multiple customers.
    
    Args:
        customers: List of customer feature sets
    
    Returns:
        List of predictions with churn_probability and churn_flag
    """
    logger.info("Received /batch_predict request with %s customers", len(customers))
    df = pd.DataFrame([c.model_dump() for c in customers])
    try:
        result = predict(df)
        response = result.to_dict(orient="records")
        logger.info("Completed /batch_predict request for %s customers", len(response))
        return response
    except Exception as exc:
        logger.exception("Prediction failed for /batch_predict")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction service unavailable: {exc}",
        ) from exc


@app.post("/v1/predict")
def churn_predict_v1(customer: Customer):
    return churn_predict(customer)


@app.post("/v1/batch_predict")
def batch_churn_predict_v1(customers: list[Customer]):
    return batch_churn_predict(customers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
