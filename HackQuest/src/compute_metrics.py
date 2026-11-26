"""
compute_metrics.py
Single Source of Truth for all quantitative metrics across RetainPulse AI.
Computes metrics directly from the model on the held-out test set,
derives downstream financial impact, computes bootstrap 95% confidence intervals,
probability calibration, survival analysis, permutation importance, and batch-scoring benchmarks.
Writes everything to artifacts/metrics.json.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

# Ensure parent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data, get_data_summary
from src.preprocessor import prepare_train_test_split
from src.models import (
    train_all_models, evaluate_model, calculate_cumulative_gains_and_lift,
    save_model_bundle
)
from src.calibration import compare_calibration_methods, compute_cost_matrix_profit_curve
from src.survival import fit_kaplan_meier_curves, ChurnSurvivalModel
from src.explainability import get_shap_explainer, compute_global_shap_importance

def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.35,
    n_bootstraps: int = 1000,
    random_state: int = 42
) -> dict:
    """
    Computes 95% empirical bootstrap confidence intervals (Task 5)
    for Recall, Precision, ROC-AUC, PR-AUC, and Top-20% Churn Capture.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    recalls, precisions, roc_aucs, pr_aucs, top20_captures = [], [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, size=n_samples)
        y_b_true = y_true[idx]
        y_b_prob = y_prob[idx]

        if len(np.unique(y_b_true)) < 2:
            continue

        y_b_pred = (y_b_prob >= threshold).astype(int)

        rec = recall_score(y_b_true, y_b_pred, zero_division=0)
        prec = precision_score(y_b_true, y_b_pred, zero_division=0)
        roc = roc_auc_score(y_b_true, y_b_prob)
        pr = average_precision_score(y_b_true, y_b_prob)

        gains = calculate_cumulative_gains_and_lift(y_b_true, y_b_prob)
        top20_gain = gains.iloc[1]['cum_gains_pct'] / 100.0

        recalls.append(rec)
        precisions.append(prec)
        roc_aucs.append(roc)
        pr_aucs.append(pr)
        top20_captures.append(top20_gain)

    def get_ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "lower_95": float(np.percentile(arr, 2.5)),
            "upper_95": float(np.percentile(arr, 97.5)),
            "display": f"{np.mean(arr)*100:.1f}% [{np.percentile(arr, 2.5)*100:.1f}–{np.percentile(arr, 97.5)*100:.1f}%]"
        }

    return {
        "recall": get_ci(recalls),
        "precision": get_ci(precisions),
        "roc_auc": {
            "mean": float(np.mean(roc_aucs)),
            "lower_95": float(np.percentile(roc_aucs, 2.5)),
            "upper_95": float(np.percentile(roc_aucs, 97.5)),
            "display": f"{np.mean(roc_aucs):.3f} [{np.percentile(roc_aucs, 2.5):.3f}–{np.percentile(roc_aucs, 97.5):.3f}]"
        },
        "pr_auc": {
            "mean": float(np.mean(pr_aucs)),
            "lower_95": float(np.percentile(pr_aucs, 2.5)),
            "upper_95": float(np.percentile(pr_aucs, 97.5)),
            "display": f"{np.mean(pr_aucs):.3f} [{np.percentile(pr_aucs, 2.5):.3f}–{np.percentile(pr_aucs, 97.5):.3f}]"
        },
        "top_20_capture": get_ci(top20_captures)
    }

def run_compute_metrics(
    chosen_threshold: float = 0.35,
    random_state: int = 42,
    output_dir: str = None
) -> dict:
    """
    Executes the full evaluation pipeline and outputs metrics.json.
    """
    print("=" * 65)
    print("RetainPulse AI: Computing Single Source of Truth Metrics")
    print("=" * 65)

    # 1. Load data
    df = load_data()
    summary = get_data_summary(df)
    total_customers = summary["total_customers"]
    total_churners = summary["churn_count"]
    baseline_churn_rate = summary["churn_rate"] / 100.0

    # 2. Train-test split
    splits = prepare_train_test_split(df, test_size=0.20, random_state=random_state)
    trained_models = train_all_models(splits["X_train_proc"], splits["y_train"], random_state=random_state)
    best_model = trained_models["Ensemble (XGB + LR + RF)"]
    save_model_bundle(best_model, splits["preprocessor"])

    # 3. Model evaluation on test set
    y_test = splits["y_test"]
    X_test_proc = splits["X_test_proc"]
    eval_metrics = evaluate_model(best_model, X_test_proc, y_test, threshold=chosen_threshold)
    y_prob = eval_metrics["y_prob"]

    # 4. Cumulative gains & decile lift
    gains_df = calculate_cumulative_gains_and_lift(y_test, y_prob)
    top_10_capture_rate = float(gains_df.iloc[0]["cum_gains_pct"] / 100.0)
    top_10_lift = float(gains_df.iloc[0]["lift"])
    top_20_capture_rate = float(gains_df.iloc[1]["cum_gains_pct"] / 100.0)
    top_20_lift = float(gains_df.iloc[1]["lift"])

    # 5. Financial assumptions
    contact_cost = 40.0
    save_rate = 0.25
    annual_clv = 780.0  # 12 months * $65/mo

    # Strategy A: Mass Contact Everyone
    cost_everyone = float(total_customers * contact_cost)
    saved_everyone = float(total_churners * save_rate)
    revenue_kept_everyone = float(saved_everyone * annual_clv)
    net_gain_everyone = float(revenue_kept_everyone - cost_everyone)
    roi_everyone = float((net_gain_everyone / cost_everyone * 100.0) if cost_everyone > 0 else 0)

    # Strategy B: Model Targeting Top 20%
    contacted_model = float(total_customers * 0.20)
    cost_model = float(contacted_model * contact_cost)
    churners_targeted_model = float(total_churners * top_20_capture_rate)
    saved_model = float(churners_targeted_model * save_rate)
    revenue_kept_model = float(saved_model * annual_clv)
    net_gain_model = float(revenue_kept_model - cost_model)
    roi_model = float((net_gain_model / cost_model * 100.0) if cost_model > 0 else 0)

    cost_savings = cost_everyone - cost_model
    additional_profit = net_gain_model - net_gain_everyone
    roi_multiplier = (roi_model / roi_everyone) if roi_everyone > 0 else 1.0

    # 6. Bootstrap Confidence Intervals (Task 5)
    print("Computing 1,000 bootstrap confidence intervals...")
    bootstrap_cis = compute_bootstrap_cis(
        y_test, y_prob, threshold=chosen_threshold, n_bootstraps=1000, random_state=random_state
    )

    # 7. Probability Calibration (Task 3)
    print("Evaluating probability calibration (Isotonic vs Sigmoid vs Raw)...")
    cal_results = compare_calibration_methods(
        best_model, splits["X_train_proc"], splits["y_train"], X_test_proc, y_test, cv=3
    )

    # 8. Cost Matrix & Profit Curve (Task 3)
    cost_matrix_eval = compute_cost_matrix_profit_curve(
        y_test, cal_results["best_probs"], fp_cost=contact_cost, fn_cost=annual_clv, save_rate=save_rate
    )

    # 9. Survival Analysis (Task 4)
    print("Fitting Kaplan-Meier survival curves and Cox PH model...")
    km_contracts = fit_kaplan_meier_curves(df, group_col="Contract")
    survival_summary = {
        grp: {
            "median_survival_time": info["median_survival_time"],
            "total_customers": info["total_customers"],
            "churners": info["churners"]
        }
        for grp, info in km_contracts.items()
    }

    # 10. Permutation Importance & Rank Agreement (Task 6)
    print("Computing permutation feature importance (n_repeats=10)...")
    perm = permutation_importance(
        best_model, X_test_proc, y_test,
        scoring="average_precision", n_repeats=10, random_state=random_state, n_jobs=-1
    )
    feature_names = splits["feature_names"]
    perm_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    # SHAP importance on sample for Spearman rank correlation
    explainer = get_shap_explainer(trained_models["XGBoost (Tuned)"])
    shap_df, _ = compute_global_shap_importance(explainer, X_test_proc[:350], feature_names, top_n=len(feature_names))
    merged_imp = pd.merge(shap_df[["Feature", "Mean_Abs_SHAP"]], perm_df[["Feature", "Importance_Mean"]], on="Feature")
    spearman_corr, spearman_pval = spearmanr(merged_imp["Mean_Abs_SHAP"], merged_imp["Importance_Mean"])

    # 11. Batch-Scoring Benchmark (Task 4 Stretch)
    print("Benchmarking batch inference throughput on 100K synthetic rows...")
    repeat_factor = int(np.ceil(100000 / len(X_test_proc)))
    X_100k = np.tile(X_test_proc, (repeat_factor, 1))[:100000]
    _ = best_model.predict_proba(X_100k[:1000])  # Warmup
    t0 = time.perf_counter()
    _ = best_model.predict_proba(X_100k)
    benchmark_duration = time.perf_counter() - t0
    benchmark_rows_per_sec = 100000.0 / benchmark_duration

    # 12. Leaderboard
    leaderboard = []
    for m_name, m_obj in trained_models.items():
        m_eval = evaluate_model(m_obj, X_test_proc, y_test, threshold=chosen_threshold)
        leaderboard.append({
            "model": m_name,
            "accuracy": round(float(m_eval["accuracy"]), 4),
            "precision": round(float(m_eval["precision"]), 4),
            "recall": round(float(m_eval["recall"]), 4),
            "f1": round(float(m_eval["f1"]), 4),
            "roc_auc": round(float(m_eval["roc_auc"]), 4),
            "pr_auc": round(float(m_eval["pr_auc"]), 4)
        })
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="pr_auc", ascending=False).to_dict(orient="records")

    cm = eval_metrics["confusion_matrix"].tolist()

    decile_summary = []
    for _, row in gains_df.iterrows():
        decile_summary.append({
            "decile": str(row["decile"]),
            "customers": int(row["customers"]),
            "churners": int(row["churners"]),
            "cum_customers": int(row["cum_customers"]),
            "cum_churners": int(row["cum_churners"]),
            "pct_customers": round(float(row["pct_customers"]), 2),
            "cum_gains_pct": round(float(row["cum_gains_pct"]), 2),
            "churn_rate_pct": round(float(row["churn_rate_pct"]), 2),
            "lift": round(float(row["lift"]), 2)
        })

    metrics_payload = {
        "metadata": {
            "version": "2.1.0",
            "seed": random_state,
            "dataset_rows": int(total_customers),
            "dataset_churners": int(total_churners),
            "baseline_churn_rate_pct": round(baseline_churn_rate * 100.0, 2),
            "test_set_size": int(len(y_test)),
            "test_set_churners": int(y_test.sum()),
            "primary_model": "Ensemble (XGB + LR + RF)",
            "operating_threshold": chosen_threshold
        },
        "performance": {
            "roc_auc": round(float(eval_metrics["roc_auc"]), 4),
            "pr_auc": round(float(eval_metrics["pr_auc"]), 4),
            "recall": round(float(eval_metrics["recall"]), 4),
            "precision": round(float(eval_metrics["precision"]), 4),
            "f1": round(float(eval_metrics["f1"]), 4),
            "accuracy": round(float(eval_metrics["accuracy"]), 4),
            "confusion_matrix": {
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1])
            },
            "caught_churners_test": int(cm[1][1]),
            "total_churners_test": int(y_test.sum())
        },
        "confidence_intervals_95": bootstrap_cis,
        "gains_and_lift": {
            "top_10_capture_rate_pct": round(top_10_capture_rate * 100.0, 2),
            "top_10_lift": round(top_10_lift, 2),
            "top_20_capture_rate_pct": round(top_20_capture_rate * 100.0, 2),
            "top_20_lift": round(top_20_lift, 2),
            "deciles": decile_summary
        },
        "probability_calibration": {
            "best_method": cal_results["best_method"],
            "brier_uncalibrated": cal_results["uncalibrated"]["brier_score"],
            "brier_calibrated": cal_results[cal_results["best_method"]]["brier_score"],
            "logloss_uncalibrated": cal_results["uncalibrated"]["log_loss"],
            "logloss_calibrated": cal_results[cal_results["best_method"]]["log_loss"],
            "brier_reduction_pct": cal_results["brier_reduction_pct"]
        },
        "cost_matrix_analysis": {
            "fp_cost": contact_cost,
            "fn_cost": annual_clv,
            "save_rate": save_rate,
            "optimal_threshold": cost_matrix_eval["optimal_threshold"],
            "max_net_profit_test": cost_matrix_eval["max_net_profit"],
            "theoretical_bayes_threshold": cost_matrix_eval["theoretical_bayes_threshold"]
        },
        "survival_analysis": survival_summary,
        "permutation_importance": {
            "top_10_features": perm_df.head(10).to_dict(orient="records"),
            "spearman_rank_correlation_with_shap": round(float(spearman_corr), 4),
            "spearman_p_value": float(spearman_pval)
        },
        "batch_scoring_benchmark": {
            "rows_scored": 100000,
            "duration_seconds": round(float(benchmark_duration), 3),
            "rows_per_second": int(benchmark_rows_per_sec)
        },
        "financial_impact": {
            "assumptions": {
                "contact_cost": contact_cost,
                "save_rate": save_rate,
                "annual_clv": annual_clv,
                "target_top_pct": 20.0
            },
            "strategy_a_mass_contact": {
                "name": "Strategy A: Contact Everyone (Mass Campaign)",
                "contacted": int(total_customers),
                "contacted_pct": 100.0,
                "campaign_cost": cost_everyone,
                "saved_churners": round(saved_everyone, 1),
                "revenue_kept": round(revenue_kept_everyone, 2),
                "net_gain": round(net_gain_everyone, 2),
                "roi_pct": round(roi_everyone, 2)
            },
            "strategy_b_model_targeting": {
                "name": "Strategy B: Target Top 20% (Model-Driven)",
                "contacted": int(contacted_model),
                "contacted_pct": 20.0,
                "campaign_cost": cost_model,
                "churn_capture_rate_pct": round(top_20_capture_rate * 100.0, 2),
                "churners_targeted": round(churners_targeted_model, 1),
                "saved_churners": round(saved_model, 1),
                "revenue_kept": round(revenue_kept_model, 2),
                "net_gain": round(net_gain_model, 2),
                "roi_pct": round(roi_model, 2)
            },
            "comparison": {
                "cost_savings": round(cost_savings, 2),
                "additional_profit": round(additional_profit, 2),
                "roi_multiplier": round(roi_multiplier, 2)
            }
        },
        "model_leaderboard": leaderboard_df
    }

    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    root_artifacts = os.path.join(os.path.dirname(BASE_DIR), "artifacts")
    if os.path.exists(os.path.dirname(root_artifacts)):
        try:
            os.makedirs(root_artifacts, exist_ok=True)
            with open(os.path.join(root_artifacts, "metrics.json"), "w") as f:
                json.dump(metrics_payload, f, indent=2)
        except Exception:
            pass

    print(f"\n[SUCCESS] Unified single source of truth metrics written to {metrics_file}")
    print(f"  ROC-AUC: {metrics_payload['performance']['roc_auc']} ({bootstrap_cis['roc_auc']['display']})")
    print(f"  PR-AUC:  {metrics_payload['performance']['pr_auc']} ({bootstrap_cis['pr_auc']['display']})")
    print(f"  Top 20% Churn Capture: {metrics_payload['gains_and_lift']['top_20_capture_rate_pct']}% ({bootstrap_cis['top_20_capture']['display']})")
    print(f"  Brier Score: {cal_results['uncalibrated']['brier_score']} -> {cal_results[cal_results['best_method']]['brier_score']} (-{cal_results['brier_reduction_pct']}%)")
    print(f"  Spearman SHAP-Permutation Agreement: {spearman_corr:.3f}")
    print(f"  Batch Inference: {metrics_payload['batch_scoring_benchmark']['rows_per_second']:,} rows/sec ({benchmark_duration:.3f}s for 100k rows)")
    return metrics_payload

if __name__ == "__main__":
    run_compute_metrics()
