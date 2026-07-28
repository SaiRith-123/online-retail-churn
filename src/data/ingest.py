import pandas as pd
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML configuration from `path` and return as dict."""
    with open(path, "r") as f:
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
