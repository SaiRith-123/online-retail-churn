import pandas as pd

from src.monitoring import compute_feature_drift


def test_compute_feature_drift_reports_expected_structure():
    reference = pd.DataFrame({"monetary": [100.0, 120.0, 110.0], "frequency": [2, 3, 2]})
    current = pd.DataFrame({"monetary": [100.0, 150.0, 130.0], "frequency": [2, 3, 2]})

    report = compute_feature_drift(reference, current, threshold=0.1)

    assert report["drift_detected"] is True
    assert "monetary" in report["features"]
    assert report["features"]["monetary"]["drift_flag"] is True
    assert report["features"]["frequency"]["drift_flag"] is False
