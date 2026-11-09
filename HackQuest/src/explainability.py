"""
explainability.py
Provides global and local model explainability using SHAP.
Generates feature importance rankings, interactive JavaScript force plots,
matplotlib force plots, and interactive waterfall charts.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def get_shap_explainer(model, X_background: np.ndarray = None):
    """Initializes TreeExplainer for XGBoost or standard Explainer."""
    model_type = type(model).__name__

    if "XGB" in model_type or "RandomForest" in model_type:
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "estimators_"):
        # For VotingClassifier, use the main XGBoost sub-estimator
        xgb_sub = model.named_estimators_.get('xgb', model.estimators_[0])
        explainer = shap.TreeExplainer(xgb_sub)
    else:
        if X_background is not None and len(X_background) > 100:
            bg = shap.sample(X_background, 100)
        else:
            bg = X_background
        explainer = shap.Explainer(model, bg)

    return explainer

def compute_global_shap_importance(explainer, X_sample: np.ndarray, feature_names: list, top_n: int = 15) -> tuple[pd.DataFrame, np.ndarray]:
    """Computes global mean |SHAP value| importance across sample."""
    raw_shap = explainer.shap_values(X_sample)

    if isinstance(raw_shap, list):
        vals = raw_shap[1]
    elif len(raw_shap.shape) == 3:
        vals = raw_shap[:, :, 1]
    else:
        vals = raw_shap

    mean_abs_shap = np.abs(vals).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).head(top_n).reset_index(drop=True)

    return importance_df, vals

def plot_shap_global_bar(importance_df: pd.DataFrame) -> go.Figure:
    """Creates a high-contrast interactive Plotly horizontal bar chart."""
    df_sorted = importance_df.sort_values(by="Mean_Abs_SHAP", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted["Mean_Abs_SHAP"],
        y=df_sorted["Feature"],
        orientation="h",
        marker=dict(
            color=df_sorted["Mean_Abs_SHAP"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f"{v:.4f}" for v in df_sorted["Mean_Abs_SHAP"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP Value| Across Subscribers)",
        xaxis_title="Average Impact on Model Output Magnitude",
        yaxis_title="Feature Name",
        height=520,
        margin=dict(l=20, r=40, t=50, b=40),
        template="plotly_white"
    )
    return fig

def explain_customer_prediction(explainer, x_row: np.ndarray, feature_names: list, top_n: int = 8) -> dict:
    """Explains an individual customer's prediction via SHAP values."""
    x_2d = x_row.reshape(1, -1)
    raw_shap = explainer.shap_values(x_2d)

    if isinstance(raw_shap, list):
        vals = raw_shap[1][0]
    elif len(raw_shap.shape) == 3:
        vals = raw_shap[0, :, 1]
    elif len(raw_shap.shape) == 2:
        vals = raw_shap[0]
    else:
        vals = raw_shap

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": vals,
        "Abs_SHAP": np.abs(vals),
        "Feature_Value": x_row
    }).sort_values(by="Abs_SHAP", ascending=False).reset_index(drop=True)

    top_churn_drivers = contrib_df[contrib_df["SHAP_Value"] > 0].head(top_n).to_dict(orient="records")
    top_retention_drivers = contrib_df[contrib_df["SHAP_Value"] < 0].head(top_n).to_dict(orient="records")

    return {
        "base_value": float(base_value),
        "shap_values": vals,
        "raw_features": x_row,
        "top_churn_drivers": top_churn_drivers,
        "top_retention_drivers": top_retention_drivers,
        "all_contributions": contrib_df
    }

def generate_shap_force_html(explanation: dict, feature_names: list) -> str:
    """
    Renders the official interactive JavaScript SHAP Force Plot into standalone HTML.
    Can be rendered cleanly in Streamlit via st.components.v1.html().
    """
    try:
        base_val = explanation["base_value"]
        shap_vals = explanation["shap_values"]
        raw_feat = explanation["raw_features"]

        force_obj = shap.force_plot(
            base_val,
            shap_vals,
            raw_feat,
            feature_names=feature_names,
            matplotlib=False,
            show=False
        )
        html_content = f"<head>{shap.getjs()}</head><body style='margin:0; padding:10px; font-family:sans-serif;'>{force_obj.html()}</body>"
        return html_content
    except Exception as e:
        return f"<div style='color:red;'>Could not render JS force plot: {e}</div>"

def plot_shap_force_matplotlib(explanation: dict, feature_names: list):
    """Generates a static Matplotlib SHAP Force Plot as fallback."""
    try:
        base_val = explanation["base_value"]
        shap_vals = explanation["shap_values"]
        raw_feat = explanation["raw_features"]

        fig = shap.force_plot(
            base_val,
            shap_vals,
            raw_feat,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        return plt.gcf()
    except Exception:
        return None

def plot_individual_waterfall(explanation: dict, max_features: int = 10) -> go.Figure:
    """Builds an interactive high-contrast waterfall chart showing features shifting churn risk."""
    df = explanation["all_contributions"].head(max_features).copy()
    df = df.iloc[::-1]

    colors = ['#EF4444' if val > 0 else '#10B981' for val in df["SHAP_Value"]]

    fig = go.Figure(go.Bar(
        x=df["SHAP_Value"],
        y=df["Feature"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in df["SHAP_Value"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="SHAP Feature Attribution (Red = Increases Churn Risk, Green = Protects Retention)",
        xaxis_title="SHAP Value (Contribution to Log-Odds / Probability)",
        yaxis_title="Feature",
        height=450,
        margin=dict(l=20, r=50, t=50, b=40),
        template="plotly_white"
    )
    return fig
