"""
Real-time and batch inference module for churn prediction.

Loads trained model from MLflow model registry and makes predictions
on new customer data.
"""
import importlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
from src.data.ingest import load_config

mlflow_sklearn = importlib.import_module("mlflow.sklearn")
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
DEFAULT_STAGE = os.environ.get("DEFAULT_MODEL_STAGE", "Production")
LOCAL_MODEL_PATH = Path(os.environ.get("LOCAL_MODEL_PATH", "model.joblib"))


def load_model(stage: str | None = None):
    """Load trained model from MLflow model registry or a local joblib fallback.

    Args:
        stage: Model stage to load (Production, Staging, or None for latest).
            Defaults to the environment setting DEFAULT_MODEL_STAGE.

    Returns:
        Trained scikit-learn pipeline model
    """
    stage_name = stage or DEFAULT_STAGE
    try:
        model = mlflow_sklearn.load_model(
            model_uri=f"models:/churn_classifier_prod/{stage_name}"
        )
    except Exception:
        if LOCAL_MODEL_PATH.exists():
            model = joblib.load(LOCAL_MODEL_PATH)
        else:
            raise

    if model is None:
        raise RuntimeError("Failed to load a churn prediction model")

    return model


def predict(new_customer_df):
    """Generate churn predictions for customer data.

    Args:
        new_customer_df: DataFrame with customer features (no churn column needed)

    Returns:
        DataFrame with churn_probability and churn_flag (0/1) columns
    """
    model = load_model()
    prob_output = np.asarray(model.predict_proba(new_customer_df))
    probs = prob_output[:, 1]
    return pd.DataFrame({
        "churn_probability": probs,
        "churn_flag": (pd.Series(probs) >= 0.5).astype(int),
    })


if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["processed_path"]).head(10)
    X = df.drop(columns=["churn", "first_purchase", "last_purchase"])
    print(predict(X))
