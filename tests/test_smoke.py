import os

from src.predict import load_model


def test_registry_model_can_be_loaded():
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    model = load_model()

    assert model is not None
