"""
drift.py
Data and Concept Drift Monitoring using Population Stability Index (PSI)
and Predicted-vs-Actual Churn Rate tracking.

Flags:
- PSI < 0.10: Stable (Green)
- 0.10 <= PSI < 0.25: Moderate Shift (Warning - Yellow)
- PSI >= 0.25: Significant Shift (Alert - Red)
"""

import numpy as np
import pandas as pd

def calculate_numeric_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI) for a continuous numeric feature.
    Uses reference quantiles to define bin edges.
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Quantile bin edges based on reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Handle boundary conditions
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = (ref_counts / len(ref)) + eps
    cur_pct = (cur_counts / len(cur)) + eps

    # Normalize back so sum equals 1
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, psi_val))

def calculate_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    eps: float = 1e-4
) -> float:
    """
    Computes PSI for a categorical feature across all distinct categories.
    """
    ref_counts = reference.value_counts(normalize=True)
    cur_counts = current.value_counts(normalize=True)

    all_categories = sorted(list(set(ref_counts.index).union(set(cur_counts.index))))
    psi_val = 0.0

    for cat in all_categories:
        p = ref_counts.get(cat, 0.0) + eps
        q = cur_counts.get(cat, 0.0) + eps
        psi_val += (q - p) * np.log(q / p)

    return float(max(0.0, psi_val))

def evaluate_psi_status(psi_score: float) -> tuple[str, str, str]:
    """
    Returns (status_label, alert_level, color_hex).
    Thresholds:
    - PSI < 0.10: Stable
    - 0.10 <= PSI < 0.25: Warning
    - PSI >= 0.25: Alert
    """
    if psi_score < 0.10:
        return "Stable", "normal", "#10B981"
    elif psi_score < 0.25:
        return "Moderate Shift", "warning", "#F59E0B"
    else:
        return "Significant Drift", "alert", "#EF4444"

def monitor_feature_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_list: list = None
) -> pd.DataFrame:
    """
    Evaluates PSI across all provided features between reference (train)
    and current (new quarter) datasets.
    """
    if feature_list is None:
        exclude_cols = ['customerID', 'Churn']
        feature_list = [c for c in df_reference.columns if c not in exclude_cols and c in df_current.columns]

    results = []
    for col in feature_list:
        ref_series = df_reference[col]
        cur_series = df_current[col]

        if pd.api.types.is_numeric_dtype(ref_series):
            psi_score = calculate_numeric_psi(ref_series.values, cur_series.values)
            f_type = "Numeric"
        else:
            psi_score = calculate_categorical_psi(ref_series, cur_series)
            f_type = "Categorical"

        status, alert_level, color = evaluate_psi_status(psi_score)
        results.append({
            "Feature": col,
            "Type": f_type,
            "PSI": round(psi_score, 4),
            "Status": status,
            "Alert_Level": alert_level,
            "Color": color
        })

    df_psi = pd.DataFrame(results).sort_values(by="PSI", ascending=False).reset_index(drop=True)
    return df_psi

def monitor_churn_rate_drift(
    y_true_reference: np.ndarray,
    y_true_current: np.ndarray,
    y_pred_probs_current: np.ndarray,
    decision_threshold: float = 0.35
) -> dict:
    """
    Monitors target drift and prediction calibration drift:
    - Actual churn rate reference vs current
    - Mean predicted probability vs actual churn rate in current batch
    - Predicted positive rate at decision threshold vs actual rate
    """
    actual_ref_rate = float(np.mean(y_true_reference))
    actual_cur_rate = float(np.mean(y_true_current))
    mean_pred_prob = float(np.mean(y_pred_probs_current))
    pred_pos_rate = float(np.mean(y_pred_probs_current >= decision_threshold))

    target_drift_delta = actual_cur_rate - actual_ref_rate
    calibration_gap = mean_pred_prob - actual_cur_rate

    is_alert = abs(target_drift_delta) >= 0.05 or abs(calibration_gap) >= 0.05

    return {
        "reference_churn_rate_pct": round(actual_ref_rate * 100, 2),
        "current_churn_rate_pct": round(actual_cur_rate * 100, 2),
        "target_drift_delta_pct": round(target_drift_delta * 100, 2),
        "mean_predicted_prob_pct": round(mean_pred_prob * 100, 2),
        "predicted_positive_rate_pct": round(pred_pos_rate * 100, 2),
        "calibration_gap_pct": round(calibration_gap * 100, 2),
        "is_alert": bool(is_alert),
        "status": "Alert: Significant Concept Drift" if is_alert else "Healthy Calibration"
    }
