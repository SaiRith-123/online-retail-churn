import numpy as np

import app as app_module


def test_predict_churn_returns_high_risk_label_for_high_probability(monkeypatch):
    def fake_predict_proba(_df):
        return np.array([[0.1, 0.9]])

    monkeypatch.setattr(app_module.model, "predict_proba", fake_predict_proba)

    result = app_module.predict_churn(
        frequency=10,
        monetary=500,
        total_items=50,
        avg_basket_size=50,
        n_unique_products=10,
        country="United Kingdom",
        tenure_days=300,
        avg_days_between_purchases=30,
    )

    assert "HIGH RISK" in result
    assert "90.00%" in result


def test_predict_churn_returns_safe_label_for_low_probability(monkeypatch):
    def fake_predict_proba(_df):
        return np.array([[0.9, 0.1]])

    monkeypatch.setattr(app_module.model, "predict_proba", fake_predict_proba)

    result = app_module.predict_churn(
        frequency=2,
        monetary=50,
        total_items=5,
        avg_basket_size=25,
        n_unique_products=3,
        country="United Kingdom",
        tenure_days=200,
        avg_days_between_purchases=100,
    )

    assert "SAFE" in result
    assert "10.00%" in result
