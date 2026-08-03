import pandas as pd

from src import predict as predict_module


def test_load_model_falls_back_to_local_joblib(monkeypatch):
    class DummyModel:
        def predict_proba(self, df):
            return [[0.2, 0.8], [0.8, 0.2]]

    def fake_load_model(*args, **kwargs):
        raise RuntimeError("registry unavailable")

    def fake_joblib_load(*args, **kwargs):
        return DummyModel()

    monkeypatch.setattr(predict_module.mlflow.sklearn, "load_model", fake_load_model)
    monkeypatch.setattr(predict_module.joblib, "load", fake_joblib_load)

    model = predict_module.load_model()
    result = predict_module.predict(pd.DataFrame([{"frequency": 1, "monetary": 10.0, "total_items": 2, "avg_basket_size": 5.0, "n_unique_products": 2, "country": "United Kingdom", "tenure_days": 20, "avg_days_between_purchases": 10.0}]))

    assert model is not None
    assert result.iloc[0]["churn_probability"] == 0.8
    assert result.iloc[0]["churn_flag"] == 1
