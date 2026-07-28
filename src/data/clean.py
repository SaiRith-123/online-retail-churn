"""Data cleaning module"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Clean and preprocess data.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove cancelled invoices (Invoice values starting with 'C') if present
    if "Invoice" in df.columns:
        df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Detect and handle outliers
    if config and config.get("preprocessing", {}).get("outlier_detection"):
        df = handle_outliers(df)
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # Drop rows with critical missing values
    df = df.dropna(subset=df.columns[:3])  # Adjust based on your data
    
    # Fill remaining missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    
    return df


def handle_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Remove outliers using z-score method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < threshold]
    
    return df
