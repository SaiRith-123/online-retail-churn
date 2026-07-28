"""ML Pipeline Orchestration"""

import yaml
from pathlib import Path
import logging

from data.ingest import load_raw_data
from data.clean import clean_data
from data.features import engineer_features, select_features
from train import train_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline():
    """Execute the complete ML pipeline."""
    logger.info("Starting ML Pipeline")
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    # Load raw data
    logger.info("Loading raw data...")
    df_raw = load_raw_data(config)
    logger.info(f"Raw data shape: {df_raw.shape}")
    
    # Clean data
    logger.info("Cleaning data...")
    df_clean = clean_data(df_raw, config)
    logger.info(f"Cleaned data shape: {df_clean.shape}")
    
    # Save interim data
    interim_path = Path(config["data"]["interim_path"])
    interim_path.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(interim_path / "cleaned_data.csv", index=False)
    
    # Engineer features
    logger.info("Engineering features...")
    df_features = engineer_features(df_clean)
    logger.info(f"Features engineered, shape: {df_features.shape}")
    
    # Select features
    df_selected = select_features(df_features, config)
    logger.info(f"Features selected, shape: {df_selected.shape}")
    
    # Save processed data
    processed_path = Path(config["data"]["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(processed_path / "final_data.csv", index=False)
    
    # Train model
    logger.info("Training model...")
    model = train_model(df_selected, config)
    logger.info("Model training completed")
    
    logger.info("Pipeline completed successfully")
    
    return model


if __name__ == "__main__":
    run_pipeline()
