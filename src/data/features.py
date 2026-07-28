"""Feature engineering module"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create and engineer features for modeling.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Add temporal features if date columns exist
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Month"] = df["InvoiceDate"].dt.month
        df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
        df["Quarter"] = df["InvoiceDate"].dt.quarter
    
    # Add aggregation features
    # These are examples; adjust based on your actual data
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["TotalValue"] = df["Quantity"] * df["UnitPrice"]
    
    return df


def select_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Select features based on correlation threshold.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with selected features
    """
    threshold = config.get("features", {}).get("correlation_threshold", 0.95)
    
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select features based on correlation threshold
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    return df.drop(columns=to_drop, errors="ignore")
