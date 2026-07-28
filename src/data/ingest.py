"""Data ingestion module"""

import pandas as pd
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Load raw data using configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Raw DataFrame
    """
    raw_path = config.get("data", {}).get("raw_path")
    return load_data(raw_path)
