from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def compute_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
    """Compute a lightweight drift summary for numeric features.

    Args:
        reference_df: Baseline dataset for comparison.
        current_df: New production dataset to evaluate.
        threshold: Relative shift threshold used to flag drift.

    Returns:
        Dictionary containing per-feature drift metrics and an overall flag.
    """
    reference = reference_df.copy()
    current = current_df.copy()

    numeric_cols = [col for col in reference.columns if col in current.columns and pd.api.types.is_numeric_dtype(reference[col])]
    report: Dict[str, Any] = {"features": {}, "drift_detected": False}

    for col in numeric_cols:
        ref_mean = float(reference[col].mean())
        cur_mean = float(current[col].mean())
        ref_std = float(reference[col].std(ddof=0))
        cur_std = float(current[col].std(ddof=0))
        relative_shift = abs(cur_mean - ref_mean) / max(abs(ref_mean), 1e-9)

        drift_flag = relative_shift > threshold

        report["features"][col] = {
            "reference_mean": ref_mean,
            "current_mean": cur_mean,
            "reference_std": ref_std,
            "current_std": cur_std,
            "relative_shift": relative_shift,
            "drift_flag": drift_flag,
        }

        if drift_flag:
            report["drift_detected"] = True

    return report
