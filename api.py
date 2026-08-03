"""
FastAPI real-time serving layer for churn predictions.

Start with: uvicorn api:app --reload
Access: POST http://127.0.0.1:8000/predict
"""
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict
import pandas as pd


app = FastAPI(
    title="Churn Prediction API",
    description="Real-time inference for online retail customer churn",
    version="1.0.0"
)


class Customer(BaseModel):
    """Customer features for churn prediction."""
    frequency: int
    monetary: float
    total_items: int
    avg_basket_size: float
    n_unique_products: int
    country: str
    tenure_days: int
    avg_days_between_purchases: float

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
    df = pd.DataFrame([customer.dict()])
    result = predict(df)
    return result.to_dict(orient="records")[0]


@app.post("/batch_predict")
def batch_churn_predict(customers: list[Customer]):
    """
    Predict churn for multiple customers.
    
    Args:
        customers: List of customer feature sets
    
    Returns:
        List of predictions with churn_probability and churn_flag
    """
    df = pd.DataFrame([c.dict() for c in customers])
    result = predict(df)
    return result.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
