"""
models.py
Trains and evaluates Logistic Regression, Random Forest, XGBoost, and Voting Ensemble.
Calculates PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix, and Lift/Gains.
Provides live multi-threshold evaluation curves.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def get_imbalance_ratio(y_train: np.ndarray) -> float:
    """Calculates ratio of negative to positive samples for XGBoost scale_pos_weight."""
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    return float(neg_count / pos_count) if pos_count > 0 else 1.0

def train_all_models(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> dict:
    """
    Trains fine-tuned models:
    - Logistic Regression (regularized)
    - Random Forest (depth-regularized, balanced)
    - XGBoost (tuned learning rate, scale_pos_weight, regularization)
    - Soft Voting Ensemble (blending predictions for superior stability & PR-AUC)
    """
    scale_pos = get_imbalance_ratio(y_train)

    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=0.2,
        solver="lbfgs",
        random_state=random_state
    )

    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=9,
        class_weight="balanced",
        min_samples_split=8,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1
    )

    xgb_model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        scale_pos_weight=scale_pos * 0.92,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.2,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )

    # Train individual models
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Soft Voting Ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lr', lr_model),
            ('rf', rf_model)
        ],
        voting='soft',
        weights=[0.50, 0.30, 0.20]
    )
    ensemble_model.fit(X_train, y_train)

    return {
        "XGBoost (Tuned)": xgb_model,
        "Ensemble (XGB + LR + RF)": ensemble_model,
        "Logistic Regression": lr_model,
        "Random Forest": rf_model
    }

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluates model on test data using custom decision threshold.
    Returns standard metrics + PR-AUC, ROC-AUC, confusion matrix, curves, and threshold sweep table.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_prob)

    # Multi-threshold sweep table (0.10 to 0.85)
    sweep_records = []
    for th in np.arange(0.10, 0.90, 0.05):
        p_th = (y_prob >= th).astype(int)
        r_th = recall_score(y_test, p_th, zero_division=0)
        pr_th = precision_score(y_test, p_th, zero_division=0)
        f1_th = f1_score(y_test, p_th, zero_division=0)
        churners_caught = int(np.sum((y_test == 1) & (p_th == 1)))
        false_alarms = int(np.sum((y_test == 0) & (p_th == 1)))
        
        sweep_records.append({
            "Threshold": round(th, 2),
            "Recall": round(r_th, 4),
            "Precision": round(pr_th, 4),
            "F1-Score": round(f1_th, 4),
            "Caught_Churners": churners_caught,
            "False_Alarms": false_alarms
        })
    sweep_df = pd.DataFrame(sweep_records)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "roc_curve": (fpr, tpr),
        "pr_curve": (recall_vals, precision_vals),
        "threshold_sweep": sweep_df
    }

def compare_models(trained_models: dict, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """
    Evaluates all trained models and returns a comparative DataFrame and detailed metrics dictionary.
    """
    records = []
    eval_details = {}

    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, threshold)
        eval_details[name] = metrics
        records.append({
            "Model": name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1-Score": round(metrics["f1"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "PR-AUC": round(metrics["pr_auc"], 4)
        })

    comparison_df = pd.DataFrame(records).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    return comparison_df, eval_details

def calculate_cumulative_gains_and_lift(y_test: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Computes Cumulative Gains and Lift Table by Decile.
    """
    df = pd.DataFrame({'actual': y_test, 'prob': y_prob})
    df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)

    total_customers = len(df)
    total_churners = df['actual'].sum()

    df['decile'] = pd.qcut(df.index, q=10, labels=[f"Decile {i}" for i in range(1, 11)])

    gain_table = df.groupby('decile', observed=False).agg(
        customers=('actual', 'count'),
        churners=('actual', 'sum')
    ).reset_index()

    gain_table['cum_customers'] = gain_table['customers'].cumsum()
    gain_table['cum_churners'] = gain_table['churners'].cumsum()
    gain_table['pct_customers'] = (gain_table['cum_customers'] / total_customers) * 100
    gain_table['cum_gains_pct'] = (gain_table['cum_churners'] / total_churners) * 100
    gain_table['churn_rate_pct'] = (gain_table['churners'] / gain_table['customers']) * 100
    gain_table['lift'] = gain_table['cum_gains_pct'] / gain_table['pct_customers']

    return gain_table

def save_model_bundle(model, preprocessor, model_name: str = "best_churn_model.joblib"):
    """Saves model and preprocessor bundle using joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, model_name)
    bundle = {
        "model": model,
        "preprocessor": preprocessor
    }
    joblib.dump(bundle, save_path)
    return save_path

def load_model_bundle(model_name: str = "best_churn_model.joblib"):
    """Loads saved model bundle."""
    save_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(save_path):
        return joblib.load(save_path)
    return None
