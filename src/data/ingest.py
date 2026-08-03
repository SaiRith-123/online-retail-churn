import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | None = None) -> dict:
    """Load YAML configuration from the configured path or the default config file."""
    if path is None:
        path = os.getenv("APP_CONFIG", "config/config.yaml")

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load the raw CSV specified in `cfg`.

    Reads the CSV using ISO-8859-1 encoding and strips whitespace
    from column names to make downstream code more robust.
    """
    path = cfg["data"]["raw_path"]
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df.columns = [c.strip() for c in df.columns]
    return df
