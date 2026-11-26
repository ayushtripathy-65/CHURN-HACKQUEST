"""
retrain.py
Quarterly Retraining & Production Drift Simulation Pipeline.
Simulates next-quarter cohort with realistic feature & target drift,
evaluates model degradation, and re-optimizes operating decision threshold for profit.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data
from src.preprocessor import prepare_train_test_split, ChurnPreprocessor
from src.drift import monitor_feature_drift, monitor_churn_rate_drift
from src.models import train_all_models, evaluate_model, calculate_cumulative_gains_and_lift

def generate_drifted_quarter_batch(
    df_base: pd.DataFrame,
    sample_size: int = 1500,
    random_state: int = 101
) -> pd.DataFrame:
    """
    Simulates a next-quarter cohort with injected macroeconomic drift:
    1. Inflation price hike: Monthly charges pushed up by 9-12%
    2. Contract instability: 18% of long-term contracts lapse into Month-to-Month
    3. Service cost-cutting: 15% fewer customers opt for TechSupport & Security
    4. Macroeconomic churn surge: Churn rate increases from ~26.5% to ~32.5%
    """
    rng = np.random.RandomState(random_state)
    df_sim = df_base.sample(n=sample_size, replace=True, random_state=random_state).copy()
    df_sim = df_sim.reset_index(drop=True)

    # 1. Inflation price hike
    if 'MonthlyCharges' in df_sim.columns:
        price_hike = rng.uniform(1.08, 1.14, size=len(df_sim))
        df_sim['MonthlyCharges'] = np.round(df_sim['MonthlyCharges'] * price_hike, 2)

    # 2. Contract downgrade (lapsed into month-to-month)
    if 'Contract' in df_sim.columns:
        contract_mask = (df_sim['Contract'] != 'Month-to-month') & (rng.rand(len(df_sim)) < 0.22)
        df_sim.loc[contract_mask, 'Contract'] = 'Month-to-month'

    # 3. Add-on cancellations
    if 'TechSupport' in df_sim.columns:
        ts_cancel = (df_sim['TechSupport'] == 'Yes') & (rng.rand(len(df_sim)) < 0.20)
        df_sim.loc[ts_cancel, 'TechSupport'] = 'No'

    if 'OnlineSecurity' in df_sim.columns:
        sec_cancel = (df_sim['OnlineSecurity'] == 'Yes') & (rng.rand(len(df_sim)) < 0.18)
        df_sim.loc[sec_cancel, 'OnlineSecurity'] = 'No'

    # 4. Injected churn surge based on risk compounding
    if 'Churn' in df_sim.columns:
        # Increase churn chance especially for high-bill month-to-month
        extra_churn_risk = (
            (df_sim['Contract'] == 'Month-to-month').astype(int) * 0.15 +
            (df_sim['MonthlyCharges'] > 85).astype(int) * 0.10 +
            (df_sim['TechSupport'] == 'No').astype(int) * 0.08
        )
        should_flip = (df_sim['Churn'] == 'No') & (rng.rand(len(df_sim)) < extra_churn_risk)
        df_sim.loc[should_flip, 'Churn'] = 'Yes'

    return df_sim

def calculate_expected_profit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    contact_cost: float = 40.0,
    annual_clv: float = 780.0,
    save_rate: float = 0.25,
    threshold_steps: np.ndarray = None
) -> pd.DataFrame:
    """
    Computes net campaign profit across candidate decision thresholds.
    Net Profit = (Saved Churners * CLV) - (Contacted * Contact Cost)
    where:
      Contacted = TP + FP
      Saved Churners = TP * Save Rate
    """
    if threshold_steps is None:
        threshold_steps = np.round(np.arange(0.10, 0.86, 0.01), 2)

    records = []
    total_samples = len(y_true)

    for th in threshold_steps:
        y_pred = (y_prob >= th).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        contacted = tp + fp
        saved_churners = tp * save_rate
        campaign_cost = contacted * contact_cost
        revenue_saved = saved_churners * annual_clv
        net_profit = revenue_saved - campaign_cost

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        records.append({
            "threshold": round(float(th), 2),
            "contacted": contacted,
            "tp": tp,
            "fp": fp,
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "campaign_cost": round(campaign_cost, 2),
            "revenue_saved": round(revenue_saved, 2),
            "net_profit": round(net_profit, 2)
        })

    df_profit = pd.DataFrame(records)
    return df_profit

def run_retraining_simulation(
    random_state: int = 42,
    contact_cost: float = 40.0,
    annual_clv: float = 780.0,
    save_rate: float = 0.25
) -> dict:
    """
    Simulates end-to-end quarterly retraining workflow:
    1. Loads baseline data and model
    2. Generates drifted next-quarter dataset
    3. Evaluates PSI drift per feature & target drift
    4. Evaluates deployed model on drifted batch
    5. Re-optimizes decision threshold for maximum profit
    6. Retrains fresh model on new data and compares results
    """
    df_raw = load_data()
    splits = prepare_train_test_split(df_raw, test_size=0.20, random_state=random_state)
    trained_models = train_all_models(splits["X_train_proc"], splits["y_train"], random_state=random_state)
    deployed_model = trained_models["Ensemble (XGB + LR + RF)"]
    preprocessor = splits["preprocessor"]

    # Baseline performance on test set
    base_eval = evaluate_model(deployed_model, splits["X_test_proc"], splits["y_test"], threshold=0.35)
    base_profit_df = calculate_expected_profit_curve(
        splits["y_test"], base_eval["y_prob"], contact_cost, annual_clv, save_rate
    )
    base_opt_idx = base_profit_df["net_profit"].idxmax()
    base_optimal_th = float(base_profit_df.loc[base_opt_idx, "threshold"])
    base_optimal_profit = float(base_profit_df.loc[base_opt_idx, "net_profit"])

    # 2. Simulate next quarter batch with drift
    df_drifted = generate_drifted_quarter_batch(df_raw, sample_size=1500, random_state=101)
    y_drifted = (df_drifted["Churn"] == "Yes").astype(int).values
    X_drifted_proc = preprocessor.transform(df_drifted.drop(columns=["Churn", "customerID"], errors="ignore"))

    # 3. Drift Analysis
    feature_psi_df = monitor_feature_drift(df_raw, df_drifted)
    drifted_probs = deployed_model.predict_proba(X_drifted_proc)[:, 1]
    churn_rate_diag = monitor_churn_rate_drift(splits["y_train"], y_drifted, drifted_probs, decision_threshold=0.35)

    # 4. Deployed model performance on drifted batch at default threshold 0.35
    drift_eval_default = evaluate_model(deployed_model, X_drifted_proc, y_drifted, threshold=0.35)

    # 5. Threshold Re-optimization on drifted data
    drift_profit_df = calculate_expected_profit_curve(
        y_drifted, drifted_probs, contact_cost, annual_clv, save_rate
    )
    new_opt_idx = drift_profit_df["net_profit"].idxmax()
    new_optimal_th = float(drift_profit_df.loc[new_opt_idx, "threshold"])
    new_optimal_profit = float(drift_profit_df.loc[new_opt_idx, "net_profit"])
    profit_at_default = float(drift_profit_df[drift_profit_df["threshold"] == 0.35]["net_profit"].values[0])
    profit_gain_from_reopt = new_optimal_profit - profit_at_default

    # 6. Retrain model on drifted quarterly data (simulating retrained deployment)
    drift_splits = prepare_train_test_split(df_drifted, test_size=0.20, random_state=random_state)
    retrained_models = train_all_models(drift_splits["X_train_proc"], drift_splits["y_train"], random_state=random_state)
    retrained_model = retrained_models["Ensemble (XGB + LR + RF)"]
    retrained_eval = evaluate_model(retrained_model, drift_splits["X_test_proc"], drift_splits["y_test"], threshold=new_optimal_th)

    return {
        "baseline_summary": {
            "churn_rate_pct": float(np.mean(splits["y_test"]) * 100),
            "roc_auc": round(float(base_eval["roc_auc"]), 4),
            "pr_auc": round(float(base_eval["pr_auc"]), 4),
            "recall": round(float(base_eval["recall"]), 4),
            "precision": round(float(base_eval["precision"]), 4),
            "threshold": 0.35,
            "optimal_threshold": base_optimal_th,
            "optimal_net_profit": base_optimal_profit
        },
        "drifted_batch_summary": {
            "sample_size": len(df_drifted),
            "actual_churn_rate_pct": float(np.mean(y_drifted) * 100),
            "deployed_model_recall_at_035": round(float(drift_eval_default["recall"]), 4),
            "deployed_model_precision_at_035": round(float(drift_eval_default["precision"]), 4),
            "deployed_model_pr_auc": round(float(drift_eval_default["pr_auc"]), 4),
            "profit_at_default_035": profit_at_default,
            "reoptimized_threshold": new_optimal_th,
            "profit_at_reoptimized_th": new_optimal_profit,
            "profit_lift_from_reopt": profit_gain_from_reopt
        },
        "feature_psi_df": feature_psi_df,
        "churn_rate_diag": churn_rate_diag,
        "drift_profit_curve": drift_profit_df,
        "retrained_model_eval": {
            "roc_auc": round(float(retrained_eval["roc_auc"]), 4),
            "pr_auc": round(float(retrained_eval["pr_auc"]), 4),
            "recall": round(float(retrained_eval["recall"]), 4),
            "precision": round(float(retrained_eval["precision"]), 4)
        }
    }

if __name__ == "__main__":
    results = run_retraining_simulation()
    print("Quarterly Retraining Simulation Completed Successfully.")
    print("Baseline Churn Rate:", results["baseline_summary"]["churn_rate_pct"], "%")
    print("Drifted Batch Churn Rate:", results["drifted_batch_summary"]["actual_churn_rate_pct"], "%")
    print("Optimal Threshold Shift:", results["baseline_summary"]["threshold"], "->", results["drifted_batch_summary"]["reoptimized_threshold"])
    print("Profit Lift from Re-optimizing Threshold: +$", f"{results['drifted_batch_summary']['profit_lift_from_reopt']:,.2f}")
