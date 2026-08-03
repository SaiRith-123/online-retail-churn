from src.quality_report import build_quality_summary


def test_build_quality_summary_returns_expected_fields():
    summary = build_quality_summary({
        "roc_auc": 0.99,
        "f1": 0.96,
        "precision": 0.97,
        "recall": 0.95,
        "accuracy": 0.97,
    })

    assert summary["quality_band"] == "excellent"
    assert summary["roc_auc"] == 0.99
    assert set(["roc_auc", "f1", "precision", "recall", "accuracy", "quality_band"]).issubset(summary)
