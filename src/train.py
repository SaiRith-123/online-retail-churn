"""Model Training Module"""

import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
from sklearn.preprocessing import StandardScaler
import mlflow
import pandas as pd


logger = logging.getLogger(__name__)


def train_model(df: pd.DataFrame, config: dict):
    """
    Train a machine learning model.
    
    Args:
        df: Processed DataFrame with features
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    # Separate features and target (adjust based on your target variable)
    # Assuming last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split data
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get model configuration
    model_config = config.get("model", {})
    model_type = model_config.get("type", "random_forest")
    hyperparams = model_config.get("hyperparameters", {})
    
    # Train model
    if model_type == "random_forest":
        model = RandomForestClassifier(**hyperparams)
    elif model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed in the environment")
        model = XGBClassifier(**hyperparams, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Start MLflow run
    mlflow.start_run()
    try:
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_params(hyperparams)
        
        logger.info(f"Train Accuracy: {train_score:.4f}")
        logger.info(f"Test Accuracy: {test_score:.4f}")
        
    finally:
        mlflow.end_run()
    
    return model
