"""
calibration.py
Probability calibration via CalibratedClassifierCV and Cost Matrix Decision Analysis.
Compares Isotonic vs Sigmoid (Platt) calibration using Brier score and Log Loss,
generates reliability curves, and evaluates expected profit curves under custom cost matrices.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

def compare_calibration_methods(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = 3
) -> dict:
    """
    Fits and compares Uncalibrated, Sigmoid (Platt), and Isotonic calibration.
    Returns comparison metrics and the optimal calibrated model bundle.
    """
    # 1. Uncalibrated
    y_prob_uncal = base_model.predict_proba(X_test)[:, 1]
    brier_uncal = brier_score_loss(y_test, y_prob_uncal)
    logloss_uncal = log_loss(y_test, y_prob_uncal)

    # 2. Sigmoid (Platt scaling)
    cal_sig = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=cv)
    cal_sig.fit(X_train, y_train)
    y_prob_sig = cal_sig.predict_proba(X_test)[:, 1]
    brier_sig = brier_score_loss(y_test, y_prob_sig)
    logloss_sig = log_loss(y_test, y_prob_sig)

    # 3. Isotonic Regression
    cal_iso = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=cv)
    cal_iso.fit(X_train, y_train)
    y_prob_iso = cal_iso.predict_proba(X_test)[:, 1]
    brier_iso = brier_score_loss(y_test, y_prob_iso)
    logloss_iso = log_loss(y_test, y_prob_iso)

    # Best method by Brier score
    candidates = [
        ("isotonic", brier_iso, logloss_iso, cal_iso, y_prob_iso),
        ("sigmoid", brier_sig, logloss_sig, cal_sig, y_prob_sig),
        ("uncalibrated", brier_uncal, logloss_uncal, base_model, y_prob_uncal)
    ]
    candidates.sort(key=lambda x: x[1])
    best_method, best_brier, best_logloss, best_cal_model, best_probs = candidates[0]

    brier_reduction_pct = ((brier_uncal - best_brier) / brier_uncal) * 100.0

    return {
        "best_method": best_method,
        "best_calibrated_model": best_cal_model,
        "best_probs": best_probs,
        "uncalibrated": {
            "brier_score": round(float(brier_uncal), 4),
            "log_loss": round(float(logloss_uncal), 4),
            "probs": y_prob_uncal
        },
        "sigmoid": {
            "brier_score": round(float(brier_sig), 4),
            "log_loss": round(float(logloss_sig), 4),
            "probs": y_prob_sig
        },
        "isotonic": {
            "brier_score": round(float(brier_iso), 4),
            "log_loss": round(float(logloss_iso), 4),
            "probs": y_prob_iso
        },
        "brier_reduction_pct": round(float(brier_reduction_pct), 2)
    }

def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Computes calibration curve coordinates (true fraction of positives vs mean predicted probability).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": round(float(brier), 4)
    }

def compute_cost_matrix_profit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_cost: float = 40.0,
    fn_cost: float = 780.0,
    save_rate: float = 0.25,
    thresholds: np.ndarray = None
) -> dict:
    """
    Computes expected net campaign profit across decision thresholds.
    Formulation:
    - Target: Contact customers with prob >= threshold (Predicted Churner).
    - Cost of contact: $FP_cost ($40) per contacted subscriber (both TP and FP).
    - Value of true positive: Customer would churn ($FN_cost = $780 CLV), but save_rate (25%) stay.
      Net Revenue saved = TP * save_rate * fn_cost.
    - Total Campaign Net Profit = (TP * save_rate * fn_cost) - ((TP + FP) * fp_cost).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.90, 0.01), 2)

    total_churners = int(np.sum(y_true == 1))
    total_samples = len(y_true)

    rows = []
    best_profit = -float("inf")
    optimal_th = 0.35

    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))

        contacted = tp + fp
        saved_churners = tp * save_rate
        campaign_cost = contacted * fp_cost
        revenue_saved = saved_churners * fn_cost
        net_profit = revenue_saved - campaign_cost

        recall = tp / total_churners if total_churners > 0 else 0
        precision = tp / contacted if contacted > 0 else 0

        if net_profit > best_profit:
            best_profit = net_profit
            optimal_th = float(th)

        rows.append({
            "threshold": round(float(th), 2),
            "contacted": contacted,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "campaign_cost": round(float(campaign_cost), 2),
            "revenue_saved": round(float(revenue_saved), 2),
            "net_profit": round(float(net_profit), 2)
        })

    profit_df = pd.DataFrame(rows)

    # Theoretical Bayes-optimal cutoff:
    # Net payoff for positive prediction = p * (save_rate * fn_cost) - fp_cost
    # Cutoff p* where payoff > 0: p* = fp_cost / (save_rate * fn_cost)
    theoretical_bayes_th = fp_cost / (save_rate * fn_cost) if (save_rate * fn_cost) > 0 else 0.50

    return {
        "optimal_threshold": optimal_th,
        "max_net_profit": round(float(best_profit), 2),
        "theoretical_bayes_threshold": round(float(theoretical_bayes_th), 3),
        "profit_curve_df": profit_df
    }
