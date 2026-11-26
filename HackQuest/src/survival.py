"""
survival.py
Time-to-Event Survival Analysis for Customer Retention.
- Kaplan-Meier survival curves by Contract and Internet Service tiers with median survival times.
- Cox Proportional Hazards model for predicting individual customer 90-day churn probability:
  P(T <= t + 3 months | T > t) = 1 - S(t + 3) / S(t).
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

def prepare_survival_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Prepares dataset for survival modeling:
    - tenure (duration in months)
    - event (1 for Churn == 'Yes', 0 for No)
    - Selected risk covariates (one-hot encoded)
    """
    clean_df = df.copy()
    if "Churn" in clean_df.columns:
        clean_df["event"] = (clean_df["Churn"] == "Yes").astype(int)
    else:
        clean_df["event"] = 0

    feature_cols = [
        "tenure", "event", "MonthlyCharges", "TotalCharges",
        "Contract", "InternetService", "TechSupport", "PaymentMethod", "PaperlessBilling"
    ]
    sub_df = clean_df[[c for c in feature_cols if c in clean_df.columns]].copy()
    
    # Fill any nulls
    sub_df["tenure"] = pd.to_numeric(sub_df["tenure"], errors="coerce").fillna(0).astype(float)
    if "TotalCharges" in sub_df.columns:
        sub_df["TotalCharges"] = pd.to_numeric(sub_df["TotalCharges"], errors="coerce").fillna(0.0)
    if "MonthlyCharges" in sub_df.columns:
        sub_df["MonthlyCharges"] = pd.to_numeric(sub_df["MonthlyCharges"], errors="coerce").fillna(0.0)

    cat_cols = ["Contract", "InternetService", "TechSupport", "PaymentMethod", "PaperlessBilling"]
    cat_cols = [c for c in cat_cols if c in sub_df.columns]

    encoded_df = pd.get_dummies(sub_df, columns=cat_cols, drop_first=True)
    covariates = [c for c in encoded_df.columns if c not in ["tenure", "event"]]

    return encoded_df, covariates

def fit_kaplan_meier_curves(df: pd.DataFrame, group_col: str = "Contract") -> dict:
    """
    Fits Kaplan-Meier survival curves grouped by contract or internet service.
    Returns curve coordinates and median survival times.
    """
    clean_df = df.copy()
    duration = clean_df["tenure"]
    event = (clean_df["Churn"] == "Yes").astype(int)

    km_results = {}
    unique_groups = clean_df[group_col].dropna().unique()

    for grp in unique_groups:
        mask = (clean_df[group_col] == grp)
        kmf = KaplanMeierFitter()
        kmf.fit(duration[mask], event[mask], label=str(grp))

        med_time = kmf.median_survival_time_
        med_str = f"{med_time:.0f} months" if np.isfinite(med_time) else "Not reached (>72m)"

        # Extract survival table
        surv_table = kmf.survival_function_.reset_index()
        surv_table.columns = ["timeline_months", "survival_probability"]

        km_results[str(grp)] = {
            "group": str(grp),
            "median_survival_time": med_str,
            "median_survival_raw": float(med_time) if np.isfinite(med_time) else None,
            "timeline": surv_table["timeline_months"].tolist(),
            "survival_prob": [round(float(p), 4) for p in surv_table["survival_probability"]],
            "total_customers": int(mask.sum()),
            "churners": int(event[mask].sum())
        }

    return km_results

class ChurnSurvivalModel:
    """
    Fits and manages a Cox Proportional Hazards model for forward-looking churn probability.
    """

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.cph = CoxPHFitter(penalizer=penalizer)
        self.covariates = []
        self.reference_encoded_df = None

    def fit(self, df: pd.DataFrame):
        surv_df, covariates = prepare_survival_dataframe(df)
        self.covariates = covariates
        self.reference_encoded_df = surv_df.iloc[:1].copy()
        self.cph.fit(surv_df, duration_col="tenure", event_col="event")
        return self

    def predict_90_day_churn_prob(self, customer_dict: dict, current_tenure: float = None) -> float:
        """
        Calculates customer's probability of churning within the next 90 days (3 months):
        P(T <= t + 3 | T > t) = 1 - S(t + 3) / S(t)
        """
        if current_tenure is None:
            current_tenure = float(customer_dict.get("tenure", 0))

        # Build feature vector matching one-hot training columns
        test_row = pd.DataFrame(0.0, index=[0], columns=self.covariates)
        for col in self.covariates:
            if col in customer_dict:
                test_row[col] = float(customer_dict[col])
            else:
                for cat in ["Contract", "InternetService", "TechSupport", "PaymentMethod", "PaperlessBilling"]:
                    if col.startswith(f"{cat}_") and col == f"{cat}_{customer_dict.get(cat)}":
                        test_row[col] = 1.0

        # Predict full survival curve
        surv_func = self.cph.predict_survival_function(test_row)
        timeline = surv_func.index.values
        probs = surv_func.iloc[:, 0].values

        t_now = float(current_tenure)
        t_target = t_now + 3.0  # 90 days ~ 3 months

        # Interpolate S(t_now) and S(t_target)
        s_now = float(np.interp(t_now, timeline, probs))
        s_target = float(np.interp(t_target, timeline, probs))

        if s_now <= 0:
            return 1.0

        p_churn_90d = 1.0 - (s_target / s_now)
        return float(np.clip(p_churn_90d, 0.0, 1.0))
