import pandas as pd
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "frequency": 2,
    "monetary": 50.0,
    "total_items": 5,
    "avg_basket_size": 25.0,
    "n_unique_products": 3,
    "country": "United Kingdom",
    "tenure_days": 200,
    "avg_days_between_purchases": 100.0,
}


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_expected_response(monkeypatch):
    def fake_predict(df):
        return pd.DataFrame({
            "churn_probability": [0.12],
            "churn_flag": [0],
        })

    monkeypatch.setattr("api.predict", fake_predict)

    response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["churn_probability"] == 0.12
    assert body["churn_flag"] == 0


def test_batch_predict_endpoint_returns_multiple_results(monkeypatch):
    def fake_predict(df):
        return pd.DataFrame({
            "churn_probability": [0.15, 0.85],
            "churn_flag": [0, 1],
        })

    monkeypatch.setattr("api.predict", fake_predict)

    response = client.post(
        "/batch_predict",
        json=[SAMPLE_PAYLOAD, {**SAMPLE_PAYLOAD, "frequency": 12}],
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["churn_flag"] == 0
    assert body[1]["churn_flag"] == 1
