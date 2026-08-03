"""
Real-time and batch inference module for churn prediction.

Loads trained model from MLflow model registry and makes predictions
on new customer data.
"""
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from src.data.ingest import load_config

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def load_model(stage="Production"):
    """Load trained model from MLflow model registry.
    
    Args:
        stage: Model stage to load (Production, Staging, or None for latest)
    
    Returns:
        Trained scikit-learn pipeline model
    """
    return mlflow.sklearn.load_model(
        model_uri=f"models:/churn_classifier_prod/{stage}"
    )


def predict(new_customer_df):
    """Generate churn predictions for customer data.
    
    Args:
        new_customer_df: DataFrame with customer features (no churn column needed)
    
    Returns:
        DataFrame with churn_probability and churn_flag (0/1) columns
    """
    model = load_model()
    probs = model.predict_proba(new_customer_df)[:, 1]
    return pd.DataFrame({
        "churn_probability": probs,
        "churn_flag": (probs >= 0.5).astype(int),
    })


if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["processed_path"]).head(10)
    X = df.drop(columns=["churn", "first_purchase", "last_purchase"])
    print(predict(X))
