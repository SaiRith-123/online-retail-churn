from __future__ import annotations

from typing import Dict, Any


def build_quality_summary(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Build a lightweight model-quality summary for experiment tracking and documentation."""
    return {
        "roc_auc": float(metrics.get("roc_auc", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "quality_band": "excellent" if metrics.get("roc_auc", 0.0) >= 0.95 else "good" if metrics.get("roc_auc", 0.0) >= 0.85 else "needs_review",
    }
