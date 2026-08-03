import pandas as pd

from src.data.clean import clean
from src.data.ingest import load_config


def test_clean_removes_null_customers():
    fake_data = pd.DataFrame({
        "Invoice": ["A1", "A2", "A3"],
        "StockCode": ["X", "Y", "Z"],
        "Description": ["Item X", "Item Y", "Item Z"],
        "Quantity": [1, 2, 3],
        "InvoiceDate": ["2021-01-01", "2021-01-02", "2021-01-03"],
        "Price": [10.0, 20.0, 30.0],
        "Customer ID": [12345.0, None, 67890.0],
        "Country": ["UK", "UK", "UK"],
    })

    cfg = load_config()
    cleaned = clean(fake_data, cfg)

    assert len(cleaned) == 2
    assert cleaned["Customer ID"].isnull().sum() == 0
