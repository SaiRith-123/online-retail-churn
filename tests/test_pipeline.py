import pandas as pd

from src.data.features import build_features
from src.data.ingest import load_config


def test_config_validation_has_required_sections():
    cfg = load_config()

    assert set(["data", "cleaning", "features", "model"]).issubset(cfg)
    assert cfg["data"]["raw_path"].endswith(".csv")
    assert cfg["model"]["cv_folds"] >= 2


def test_feature_builder_returns_expected_columns_and_binary_target():
    sample_df = pd.DataFrame(
        {
            "Customer ID": [1, 1, 2, 2],
            "Invoice": ["A1", "A2", "B1", "B2"],
            "InvoiceDate": [
                "2021-01-01",
                "2021-01-03",
                "2021-03-01",
                "2021-03-02",
            ],
            "Quantity": [1, 2, 3, 1],
            "Price": [10.0, 10.0, 20.0, 20.0],
            "StockCode": ["X", "Y", "Z", "Q"],
            "Country": ["United Kingdom", "United Kingdom", "France", "France"],
        }
    )

    cfg = {
        "features": {
            "reference_date": "2021-12-10",
            "churn_threshold_days": 90,
        }
    }

    features = build_features(sample_df, cfg)

    assert features.shape[0] == 2
    assert {"Customer ID", "recency", "frequency", "monetary", "churn"}.issubset(features.columns)
    assert features["churn"].isin([0, 1]).all()
