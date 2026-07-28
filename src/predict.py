"""Model Prediction Module"""

import joblib
import pandas as pd
from pathlib import Path


def load_model(model_path: str):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    return model


def make_predictions(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Make predictions using a trained model.
    
    Args:
        X: Input features DataFrame
        model: Trained model
        
    Returns:
        DataFrame with predictions
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    results = pd.DataFrame({
        "prediction": predictions,
        "probability_class_0": probabilities[:, 0],
        "probability_class_1": probabilities[:, 1]
    })
    
    return results


def batch_predict(data_path: str, model_path: str) -> pd.DataFrame:
    """
    Make predictions on a batch of data.
    
    Args:
        data_path: Path to input data
        model_path: Path to trained model
        
    Returns:
        DataFrame with predictions
    """
    # Load data and model
    X = pd.read_csv(data_path)
    model = load_model(model_path)
    
    # Make predictions
    predictions = make_predictions(X, model)
    
    return predictions
