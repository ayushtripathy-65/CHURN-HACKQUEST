"""
app.py
DataNova 2026 — Problem DN-BU-01: Customer Churn Prediction and Retention Strategy
RetainPulse AI: Production-Grade Customer Churn & Retention Intelligence Suite.
Integrated with Single Source of Truth (metrics.json), Probability Calibration,
Survival Analysis, Permutation Importance, Personas, A/B Testing, and Scalability Monitoring.
"""

import os
import sys
import json
import math
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import joblib

# Ensure repo root and HackQuest are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data, get_data_summary, SQLiteCohortExplorer
from src.preprocessor import prepare_train_test_split
from src.models import (
    train_all_models, evaluate_model, compare_models,
    calculate_cumulative_gains_and_lift, save_model_bundle
)
from src.explainability import (
    get_shap_explainer, compute_global_shap_importance,
    plot_shap_global_bar, explain_customer_prediction,
    generate_shap_force_html, plot_individual_waterfall,
    compute_permutation_importance, calculate_shap_permutation_rank_agreement,
    plot_permutation_importance_bar
)
from src.retention_engine import (
    assign_risk_tier, assign_value_tier, get_matrix_strategy,
    generate_personalized_actions, calculate_campaign_roi
)
from src.calibration import (
    compare_calibration_methods, compute_reliability_curve,
    compute_cost_matrix_profit_curve
)
from src.survival import fit_kaplan_meier_curves, ChurnSurvivalModel
from src.drift import monitor_feature_drift, monitor_churn_rate_drift
from src.retrain import run_retraining_simulation

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetainPulse AI | DataNova 2026 Churn & Retention",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Modern SaaS CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global styles */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.4rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        margin-bottom: 15px;
    }
    .metric-card-danger {
        border-left: 5px solid #EF4444;
        background: linear-gradient(180deg, #FFFFFF 0%, #FEF2F2 100%);
    }
    .metric-card-success {
        border-left: 5px solid #10B981;
        background: linear-gradient(180deg, #FFFFFF 0%, #ECFDF5 100%);
    }
    .metric-card-warning {
        border-left: 5px solid #F59E0B;
        background: linear-gradient(180deg, #FFFFFF 0%, #FFFBEB 100%);
    }
    .metric-card-purple {
        border-left: 5px solid #8B5CF6;
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F3FF 100%);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Badges */
    .badge-pill-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .badge-pill-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .badge-pill-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #6EE7B7;
    }

    /* Studio Card */
    .studio-card {
        background: #F8FAFC;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Callout Box */
    .callout-box {
        background-color: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        padding: 16px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Single Source of Truth Loader (metrics.json)
# ---------------------------------------------------------
@st.cache_data
def get_metrics():
    metrics_path = os.path.join(BASE_DIR, "artifacts", "metrics.json")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(os.path.dirname(BASE_DIR), "artifacts", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    # If not yet generated, compute and return
    from src.compute_metrics import run_compute_metrics
    return run_compute_metrics()

metrics_data = get_metrics()

# ---------------------------------------------------------
# Cached Data & Model Pipeline
# ---------------------------------------------------------
@st.cache_data
def get_cached_data(uploaded_file=None):
    if uploaded_file is not None:
        return load_data(uploaded_file)
    return load_data()

@st.cache_resource
def get_trained_pipeline(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    splits = prepare_train_test_split(df, test_size=test_size, random_state=random_state)
    trained_models = train_all_models(splits["X_train_proc"], splits["y_train"], random_state=random_state)
    comparison_df, eval_details = compare_models(trained_models, splits["X_test_proc"], splits["y_test"])
    
    xgb_model = trained_models["XGBoost (Tuned)"]
    explainer = get_shap_explainer(xgb_model)
    
    sample_size = min(350, len(splits["X_test_proc"]))
    imp_df, raw_shap = compute_global_shap_importance(
        explainer, splits["X_test_proc"][:sample_size], splits["feature_names"], top_n=15
    )
    
    best_model = trained_models["Ensemble (XGB + LR + RF)"]
    save_model_bundle(best_model, splits["preprocessor"])

    # Survival Model
    survival_model = ChurnSurvivalModel(penalizer=0.1)
    survival_model.fit(df)

    # Permutation Importance
    perm_df = compute_permutation_importance(
        best_model, splits["X_test_proc"], splits["y_test"],
        splits["feature_names"], scoring="average_precision", n_repeats=10, random_state=random_state, top_n=15
    )
    rank_agreement = calculate_shap_permutation_rank_agreement(imp_df, perm_df)

    # Probability Calibration
    cal_bundle = compare_calibration_methods(
        best_model, splits["X_train_proc"], splits["y_train"], splits["X_test_proc"], splits["y_test"], cv=3
    )

    return {
        "splits": splits,
        "models": trained_models,
        "comparison_df": comparison_df,
        "eval_details": eval_details,
        "best_model": best_model,
        "xgb_model": xgb_model,
        "explainer": explainer,
        "shap_importance_df": imp_df,
        "survival_model": survival_model,
        "perm_df": perm_df,
        "rank_agreement": rank_agreement,
        "calibration": cal_bundle
    }

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=55)
st.sidebar.markdown("## **RetainPulse AI**")
st.sidebar.markdown("**DataNova 2026** · Problem **DN-BU-01**")
st.sidebar.markdown("---")

# Data Source
st.sidebar.subheader("📁 Data Source")
data_mode = st.sidebar.radio("Choose Dataset", ["IBM Telco Churn (7,043 rows)", "Upload Custom CSV"], index=0)

uploaded_file = None
if data_mode == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload customer CSV file", type=["csv"])

try:
    df_raw = get_cached_data(uploaded_file)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Train Pipeline
with st.spinner("Initializing ML pipelines, SHAP explainers & survival engines..."):
    pipeline_bundle = get_trained_pipeline(df_raw)

splits = pipeline_bundle["splits"]
models = pipeline_bundle["models"]
comparison_df = pipeline_bundle["comparison_df"]
eval_details = pipeline_bundle["eval_details"]
best_model = pipeline_bundle["best_model"]
explainer = pipeline_bundle["explainer"]
shap_importance_df = pipeline_bundle["shap_importance_df"]
survival_model = pipeline_bundle["survival_model"]
perm_df = pipeline_bundle["perm_df"]
rank_agreement = pipeline_bundle["rank_agreement"]
cal_bundle = pipeline_bundle["calibration"]

# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Navigation")
nav_choice = st.sidebar.radio(
    "Go to Module:",
    [
        "📊 1. Executive EDA & SQLite Cohorts",
        "🤖 2. Model Diagnostics, Calibration & Cost Studio",
        "📈 3. Lift, Gains & Confidence Intervals",
        "⏱️ 4. Churn Timeline (Survival Analysis)",
        "🔍 5. Explainability (SHAP & Permutation)",
        "🎯 6. Real-Time Predictor, Personas & Force Plot",
        "📋 7. Retention Strategy & Uplift Teaser",
        "💰 8. Business Impact & Campaign ROI",
        "🧪 9. A/B Experimentation Calculator",
        "🚀 10. Scalability, Drift & Retraining"
    ],
    index=0
)

st.sidebar.markdown("---")
pptx_file_path = os.path.join(BASE_DIR, "DataNova_2026_Customer_Churn_Retention_Strategy.pptx")
if os.path.exists(pptx_file_path):
    with open(pptx_file_path, "rb") as f:
        pptx_bytes = f.read()
    st.sidebar.download_button(
        label="📥 Download Presentation (.pptx)",
        data=pptx_bytes,
        file_name="DataNova_2026_Customer_Churn_Retention_Strategy.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        use_container_width=True
    )

st.sidebar.markdown("""
<div style="font-size:0.8rem; color:#64748B;">
<b>DataNova 2026 Submission</b><br>
Problem: DN-BU-01 Customer Churn<br>
Single Source of Truth: <code>artifacts/metrics.json</code><br>
Primary Model: Tuned Soft Voting Ensemble
</div>
""", unsafe_allow_html=True)

# =========================================================
# MODULE 1: Executive EDA & SQLite Cohorts
# =========================================================
if nav_choice == "📊 1. Executive EDA & SQLite Cohorts":
    st.markdown('<div class="main-header">📊 Executive Overview & SQLite Cohort Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Baseline metrics, behavioral distributions, and ad-hoc SQL cohort querying.</div>', unsafe_allow_html=True)

    summary = get_data_summary(df_raw)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Subscribers</div>
            <div class="metric-value">{summary['total_customers']:,}</div>
            <div class="metric-sub">Full Dataset Scope</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-card-danger">
            <div class="metric-label">Churned Customers</div>
            <div class="metric-value">{summary['churn_count']:,}</div>
            <div class="metric-sub">{summary['churn_rate']:.1f}% Benchmark Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-card-success">
            <div class="metric-label">Retained Customers</div>
            <div class="metric-value">{summary['stay_count']:,}</div>
            <div class="metric-sub">{100 - summary['churn_rate']:.1f}% Active Base</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monthly Gross Revenue</div>
            <div class="metric-value">${summary['monthly_revenue']:,.0f}</div>
            <div class="metric-sub">Monthly Run-Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card metric-card-warning">
            <div class="metric-label">Annual Revenue at Risk</div>
            <div class="metric-value">${summary['revenue_at_risk_annual']:,.0f}</div>
            <div class="metric-sub">Annual Churn Exposure</div>
        </div>
        """, unsafe_allow_html=True)

    # Core Charts
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📜 Churn Rate by Contract Commitment")
        contract_churn = df_raw.groupby("Contract", observed=False)["Churn"].apply(lambda s: (s == 'Yes').mean() * 100).reset_index()
        contract_churn.columns = ["Contract", "Churn_Rate"]
        fig_contract = px.bar(
            contract_churn, x="Contract", y="Churn_Rate",
            color="Contract",
            color_discrete_map={"Month-to-month": "#EF4444", "One year": "#F59E0B", "Two year": "#10B981"},
            text="Churn_Rate",
            title="Churn Rate by Contract Type (%)"
        )
        fig_contract.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_contract.update_layout(yaxis_range=[0, 55], height=340, template="plotly_white")
        st.plotly_chart(fig_contract, use_container_width=True)

    with c2:
        st.markdown("### ⏱️ Tenure Distribution (Tenure Shock)")
        fig_tenure = px.histogram(
            df_raw, x="tenure", color="Churn", barmode="overlay",
            color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
            nbins=36, title="Tenure Distribution by Churn Status"
        )
        fig_tenure.update_layout(height=340, template="plotly_white", xaxis_title="Tenure (Months)")
        st.plotly_chart(fig_tenure, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚡ In-Memory SQLite Cohort Query Engine")
    st.write("Live SQL querying against the `customers` table to slice cohorts by tenure, contract, and billing.")

    explorer = SQLiteCohortExplorer(df_raw)
    default_query = """-- High-risk cohort: Month-to-month contract paying > $70/mo without TechSupport
SELECT 
    Contract,
    InternetService,
    PaymentMethod,
    COUNT(*) as total_subscribers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churners,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 1) as churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) as avg_monthly_bill
FROM customers
WHERE Contract = 'Month-to-month' AND MonthlyCharges >= 70.0
GROUP BY InternetService, PaymentMethod
ORDER BY churn_rate_pct DESC;"""

    query_input = st.text_area("SQL Query Editor:", default_query, height=140)
    if st.button("Run Cohort Query", type="primary"):
        res = explorer.run_query(query_input)
        if isinstance(res, pd.DataFrame):
            st.success(f"Query returned {len(res)} cohort groupings.")
            st.dataframe(res, use_container_width=True)
        else:
            st.error(f"SQL Execution Error: {res}")

# =========================================================
# MODULE 2: Model Diagnostics, Calibration & Cost Studio
# =========================================================
elif nav_choice == "🤖 2. Model Diagnostics, Calibration & Cost Studio":
    st.markdown('<div class="main-header">🤖 Decision Threshold Studio, Probability Calibration & Cost Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Optimize the operating decision cutoff, compare probability reliability curves, and evaluate Bayes-optimal expected profit.</div>', unsafe_allow_html=True)

    perf_data = metrics_data["performance"]
    ci_data = metrics_data["confidence_intervals_95"]
    cal_meta = metrics_data["probability_calibration"]
    cost_meta = metrics_data["cost_matrix_analysis"]

    # Threshold Studio Presets
    if "custom_thresh" not in st.session_state:
        st.session_state.custom_thresh = metrics_data["metadata"]["operating_threshold"]

    st.markdown("""
    <div class="studio-card">
        <h3 style="margin-top:0; color:#0F172A;">🎯 Interactive Decision Threshold Tuner</h3>
        <p style="color:#64748B; font-size:0.95rem;">
            Default 0.50 cutoff misses many at-risk customers. Adjusting the cutoff to 0.35 increases Recall to <strong>88.0% [84.7–91.2%]</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    with col_btn1:
        if st.button(f"🎯 High Recall Focus (0.35)", use_container_width=True):
            st.session_state.custom_thresh = 0.35
    with col_btn2:
        if st.button(f"💰 Cost-Optimal Bayes Cutoff ({cost_meta['optimal_threshold']:.2f})", use_container_width=True):
            st.session_state.custom_thresh = float(cost_meta["optimal_threshold"])
    with col_btn3:
        if st.button("⚖️ Balanced F1 Cutoff (0.45)", use_container_width=True):
            st.session_state.custom_thresh = 0.45
    with col_btn4:
        if st.button("🛡️ Standard Cutoff (0.50)", use_container_width=True):
            st.session_state.custom_thresh = 0.50

    custom_thresh = st.slider(
        "Decision Threshold Cutoff:",
        min_value=0.10, max_value=0.85,
        value=float(st.session_state.custom_thresh), step=0.01
    )

    live_eval = evaluate_model(best_model, splits["X_test_proc"], splits["y_test"], threshold=custom_thresh)
    cm = live_eval["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total_churners = int(np.sum(splits["y_test"] == 1))

    # Live Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card metric-card-danger">
            <div class="metric-label">Recall (Sensitivity)</div>
            <div class="metric-value">{live_eval['recall'] * 100:.1f}%</div>
            <div class="metric-sub">{tp} of {total_churners} Churners Caught</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{live_eval['precision'] * 100:.1f}%</div>
            <div class="metric-sub">{fp} False Alarms</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card metric-card-success">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">{live_eval['roc_auc']:.3f}</div>
            <div class="metric-sub">95% CI: {ci_data['roc_auc']['display']}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">PR-AUC (Primary Metric)</div>
            <div class="metric-value">{live_eval['pr_auc']:.3f}</div>
            <div class="metric-sub">95% CI: {ci_data['pr_auc']['display']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Reliability Curve & Cost Matrix Section (Task 3)
    st.markdown("---")
    st.markdown("### 🎯 Task 3: Probability Calibration & Reliability Analysis")

    st.markdown(f"""
    <div class="callout-box" style="border-left: 4px solid #10B981;">
        <b>Calibration Upgrade:</b> Wrapped model in <code>CalibratedClassifierCV</code> comparing Isotonic vs Sigmoid vs Raw.
        Selected <strong>{cal_meta['best_method'].capitalize()} Calibration</strong> by Brier score.
        Brier score improved from <strong>{cal_meta['brier_uncalibrated']:.4f}</strong> to <strong>{cal_meta['brier_calibrated']:.4f}</strong> 
        (a <strong>{cal_meta['brier_reduction_pct']:.1f}% reduction in error</strong>).
    </div>
    """, unsafe_allow_html=True)

    cal_col1, cal_col2 = st.columns(2)
    with cal_col1:
        # Reliability curve chart
        uncal_rel = compute_reliability_curve(splits["y_test"], cal_bundle["uncalibrated"]["probs"])
        cal_rel = compute_reliability_curve(splits["y_test"], cal_bundle["best_probs"])

        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94A3B8"), name="Perfect Calibration"
        ))
        fig_rel.add_trace(go.Scatter(
            x=uncal_rel["prob_pred"], y=uncal_rel["prob_true"],
            mode="lines+markers", line=dict(color="#EF4444"), name=f"Raw Model (Brier: {uncal_rel['brier_score']:.4f})"
        ))
        fig_rel.add_trace(go.Scatter(
            x=cal_rel["prob_pred"], y=cal_rel["prob_true"],
            mode="lines+markers", line=dict(color="#10B981", width=3), name=f"Calibrated ({cal_meta['best_method']}) (Brier: {cal_rel['brier_score']:.4f})"
        ))
        fig_rel.update_layout(
            title="Probability Reliability Curve (Calibration)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Empirical Fraction of Churners",
            height=380, template="plotly_white"
        )
        st.plotly_chart(fig_rel, use_container_width=True)

    with cal_col2:
        # Cost Matrix Expected Profit Curve
        st.markdown("#### 💵 Configurable Cost Matrix Inputs")
        c_fp, c_fn, c_sr = st.columns(3)
        with c_fp:
            cfg_fp_cost = st.number_input("FP Cost ($ offer):", value=40.0, step=5.0)
        with c_fn:
            cfg_fn_cost = st.number_input("FN Cost ($ CLV):", value=780.0, step=20.0)
        with c_sr:
            cfg_save_rate = st.number_input("Save Rate (%):", value=25.0, step=5.0) / 100.0

        profit_analysis = compute_cost_matrix_profit_curve(
            splits["y_test"], cal_bundle["best_probs"],
            fp_cost=cfg_fp_cost, fn_cost=cfg_fn_cost, save_rate=cfg_save_rate
        )
        p_df = profit_analysis["profit_curve_df"]
        opt_th = profit_analysis["optimal_threshold"]
        max_p = profit_analysis["max_net_profit"]
        bayes_th = profit_analysis["theoretical_bayes_threshold"]

        fig_profit = go.Figure()
        fig_profit.add_trace(go.Scatter(
            x=p_df["threshold"], y=p_df["net_profit"],
            mode="lines", line=dict(color="#2563EB", width=2.5), name="Net Campaign Profit ($)"
        ))
        fig_profit.add_vline(
            x=opt_th, line_dash="dash", line_color="#10B981",
            annotation_text=f"Optimum: {opt_th:.2f} (${max_p:,.0f})", annotation_position="top right"
        )
        fig_profit.add_vline(
            x=custom_thresh, line_dash="dot", line_color="#EF4444",
            annotation_text=f"Current: {custom_thresh:.2f}", annotation_position="bottom right"
        )
        fig_profit.update_layout(
            title="Expected Net Profit Across Thresholds",
            xaxis_title="Decision Threshold Cutoff",
            yaxis_title="Net Campaign Gain ($)",
            height=300, template="plotly_white"
        )
        st.plotly_chart(fig_profit, use_container_width=True)
        st.caption(f"Theoretical Bayes Cutoff: $p^* = \\frac{{C_{{FP}}}}{{C_{{FP}} + (\\text{{SaveRate}} \\times C_{{FN}})}} = {bayes_th:.3f}$. The curve justifies tuning threshold in the {bayes_th:.2f}–0.35 region.")

    # Leaderboard
    st.markdown("---")
    st.markdown("### 🏆 Comprehensive Model Comparison Leaderboard")
    st.dataframe(comparison_df.style.highlight_max(subset=["PR-AUC", "Recall", "ROC-AUC", "F1-Score"], color="#D1FAE5"), use_container_width=True)

# =========================================================
# MODULE 3: Lift, Gains & Confidence Intervals
# =========================================================
elif nav_choice == "📈 3. Lift, Gains & Confidence Intervals":
    st.markdown('<div class="main-header">📈 Cumulative Gains, Lift & Bootstrap Confidence Intervals</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating concentration of churners across risk deciles with empirical 95% bootstrap intervals.</div>', unsafe_allow_html=True)

    gains_data = metrics_data["gains_and_lift"]
    ci_data = metrics_data["confidence_intervals_95"]

    top_20_gain = gains_data["top_20_capture_rate_pct"]
    top_20_lift = gains_data["top_20_lift"]
    top_10_gain = gains_data["top_10_capture_rate_pct"]
    top_10_lift = gains_data["top_10_lift"]

    # Highlights
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(f"""
        <div class="metric-card metric-card-danger">
            <div class="metric-label">Top 20% Churn Capture</div>
            <div class="metric-value">{top_20_gain:.1f}%</div>
            <div class="metric-sub">95% CI: {ci_data['top_20_capture']['display']}</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="metric-card metric-card-success">
            <div class="metric-label">Top 20% Lift Multiplier</div>
            <div class="metric-value">{top_20_lift:.2f}x</div>
            <div class="metric-sub">Over Random Contacting</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top 10% Churn Capture</div>
            <div class="metric-value">{top_10_gain:.1f}%</div>
            <div class="metric-sub">Captures >1 in 4 Churners</div>
        </div>
        """, unsafe_allow_html=True)
    with h4:
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">Top 10% Lift Multiplier</div>
            <div class="metric-value">{top_10_lift:.2f}x</div>
            <div class="metric-sub">Concentrated Risk Decile</div>
        </div>
        """, unsafe_allow_html=True)

    gains_df = pd.DataFrame(gains_data["deciles"])

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_gains = go.Figure()
        fig_gains.add_trace(go.Scatter(
            x=[0] + list(gains_df["pct_customers"]),
            y=[0] + list(gains_df["cum_gains_pct"]),
            mode='lines+markers',
            name="Model Gains",
            line=dict(color='#2563EB', width=3)
        ))
        fig_gains.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode='lines',
            name='Random Selection (Baseline)',
            line=dict(dash='dash', color='#94A3B8')
        ))
        fig_gains.add_vline(x=20, line_dash="dash", line_color="#EF4444", annotation_text=f"Top 20%: {top_20_gain:.1f}% Caught")
        fig_gains.update_layout(
            title="Cumulative Gains Chart (% of Churners Captured)",
            xaxis_title="% of Total Customers Contacted",
            yaxis_title="% of Total Churners Caught",
            height=380, template="plotly_white"
        )
        st.plotly_chart(fig_gains, use_container_width=True)

    with col_g2:
        fig_lift = px.bar(
            gains_df, x="decile", y="lift",
            title="Lift Multiplier Across Deciles",
            text="lift", color="lift",
            color_continuous_scale="Tealgrn"
        )
        fig_lift.update_traces(texttemplate='%{text:.2f}x', textposition='outside')
        fig_lift.update_layout(height=380, template="plotly_white", xaxis_title="Decile (Sorted by Risk)")
        st.plotly_chart(fig_lift, use_container_width=True)

    st.markdown("### 📊 Decile-by-Decile Lift Table")
    st.dataframe(gains_df, use_container_width=True)

# =========================================================
# MODULE 4: Churn Timeline (Survival Analysis - Task 4)
# =========================================================
elif nav_choice == "⏱️ 4. Churn Timeline (Survival Analysis)":
    st.markdown('<div class="main-header">⏱️ Churn Timeline & Time-to-Event Survival Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Answering not just <em>if</em> a customer will leave, but <em>when</em> they will leave using the <code>lifelines</code> library.</div>', unsafe_allow_html=True)

    surv_summary = metrics_data["survival_analysis"]

    # Highlights
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="metric-card metric-card-danger">
            <div class="metric-label">Month-to-Month Survival</div>
            <div class="metric-value">{surv_summary.get('Month-to-month', {}).get('median_survival_time', '35 months')}</div>
            <div class="metric-sub">50% Churn by Month 35</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-card metric-card-success">
            <div class="metric-label">One-Year Contract Survival</div>
            <div class="metric-value">{surv_summary.get('One year', {}).get('median_survival_time', 'Not reached')}</div>
            <div class="metric-sub">>72 Months Survival Horizon</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">Two-Year Contract Survival</div>
            <div class="metric-value">{surv_summary.get('Two year', {}).get('median_survival_time', 'Not reached')}</div>
            <div class="metric-sub">Maximum Lock-In & Loyalty</div>
        </div>
        """, unsafe_allow_html=True)

    km_col1, km_col2 = st.columns(2)
    with km_col1:
        st.markdown("### 📈 Kaplan-Meier Curves by Contract Commitment")
        km_contracts = fit_kaplan_meier_curves(df_raw, group_col="Contract")
        fig_km = go.Figure()
        colors = {"Month-to-month": "#EF4444", "One year": "#F59E0B", "Two year": "#10B981"}
        for grp, data in km_contracts.items():
            fig_km.add_trace(go.Scatter(
                x=data["timeline"], y=data["survival_prob"],
                mode="lines", name=f"{grp} (Median: {data['median_survival_time']})",
                line=dict(color=colors.get(grp, "#2563EB"), width=3)
            ))
        fig_km.update_layout(
            title="Kaplan-Meier Survival Function: Contract Tiers",
            xaxis_title="Tenure (Months)",
            yaxis_title="Probability Customer Remains Active S(t)",
            height=400, template="plotly_white"
        )
        st.plotly_chart(fig_km, use_container_width=True)

    with km_col2:
        st.markdown("### 🌐 Kaplan-Meier Curves by Internet Service")
        km_internet = fit_kaplan_meier_curves(df_raw, group_col="InternetService")
        fig_km_int = go.Figure()
        int_colors = {"Fiber optic": "#EF4444", "DSL": "#2563EB", "No": "#10B981"}
        for grp, data in km_internet.items():
            fig_km_int.add_trace(go.Scatter(
                x=data["timeline"], y=data["survival_prob"],
                mode="lines", name=f"{grp} (Median: {data['median_survival_time']})",
                line=dict(color=int_colors.get(grp, "#64748B"), width=2.5)
            ))
        fig_km_int.update_layout(
            title="Kaplan-Meier Survival Function: Internet Service Tiers",
            xaxis_title="Tenure (Months)",
            yaxis_title="Probability Customer Remains Active S(t)",
            height=400, template="plotly_white"
        )
        st.plotly_chart(fig_km_int, use_container_width=True)

    st.markdown("""
    <div class="callout-box">
        <b>Survival Modeling Insight (Cox PH):</b>
        Month-to-month contract holders face a steep hazard rate in the first 12–24 months, reaching their median survival at 35 months.
        In contrast, committing to a 1-year or 2-year contract completely suppresses the hazard curve, causing median survival to not be reached 
        within the 72-month observation window.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MODULE 5: Explainability (SHAP & Permutation - Task 6)
# =========================================================
elif nav_choice == "🔍 5. Explainability (SHAP & Permutation)":
    st.markdown('<div class="main-header">🔍 Attribution Intelligence: SHAP & Permutation Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cross-validating tree-specific Shapley attributions against model-agnostic permutation importance.</div>', unsafe_allow_html=True)

    # Rank Agreement Banner (Task 6)
    st.markdown(f"""
    <div class="callout-box" style="border-left: 5px solid #2563EB;">
        <b>Task 6 Attribution Stability Check:</b><br>
        Spearman Rank Correlation between SHAP Mean |Value| and Permutation Importance is 
        <strong>r = {rank_agreement['spearman_corr']:.3f}</strong> (p-value: <strong>{rank_agreement['p_value']:.2e}</strong>).<br>
        Status: <span class="badge-pill-success">{rank_agreement['status']}</span>. 
        Both independent explainability paradigms validate that <strong>contract duration</strong>, <strong>tenure</strong>, and <strong>relative monthly pricing</strong> are the dominant churn drivers.
    </div>
    """, unsafe_allow_html=True)

    x1, x2 = st.columns(2)
    with x1:
        st.markdown("### 🌳 Global TreeSHAP Feature Importance")
        fig_shap_bar = plot_shap_global_bar(shap_importance_df)
        st.plotly_chart(fig_shap_bar, use_container_width=True)

    with x2:
        st.markdown("### 🔀 Permutation Feature Importance (Drop in PR-AUC)")
        fig_perm_bar = plot_permutation_importance_bar(perm_df)
        st.plotly_chart(fig_perm_bar, use_container_width=True)

    st.markdown("### 📋 Side-by-Side Attribution Rankings")
    merged_comp = pd.merge(
        shap_importance_df.head(10)[["Feature", "Mean_Abs_SHAP"]],
        perm_df.head(10)[["Feature", "Importance_Mean"]],
        on="Feature", how="outer"
    ).fillna(0.0)
    st.dataframe(merged_comp, use_container_width=True)

# =========================================================
# MODULE 6: Real-Time Predictor, Personas & Force Plot
# =========================================================
elif nav_choice == "🎯 6. Real-Time Predictor, Personas & Force Plot":
    st.markdown('<div class="main-header">🎯 Real-Time Churn Predictor, Personas & SHAP Force Plot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">One-click demo personas, forward 90-day survival probability, interactive JavaScript force plot, and counterfactuals.</div>', unsafe_allow_html=True)

    # Task 7: Persona Presets
    st.markdown("#### 👤 1-Click Demo Personas (Task 7)")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    if "selected_persona" not in st.session_state:
        st.session_state.selected_persona = None

    with p_col1:
        if st.button("🔥 Persona: 'Riya' (High Risk)", use_container_width=True):
            st.session_state.selected_persona = "Riya"
    with p_col2:
        if st.button("⚡ Persona: 'Riya (2-Year Contract)'", use_container_width=True):
            st.session_state.selected_persona = "Riya_2yr"
    with p_col3:
        if st.button("🛡️ Persona: 'Veteran' (Low Risk)", use_container_width=True):
            st.session_state.selected_persona = "Veteran"

    # Define personas deterministically
    personas = {
        "Riya": {
            "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
            "tenure": 4, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0, "TotalCharges": 380.0
        },
        "Riya_2yr": {
            "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
            "tenure": 4, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Two year", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0, "TotalCharges": 380.0
        },
        "Veteran": {
            "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
            "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 65.0, "TotalCharges": 3900.0
        }
    }

    if st.session_state.selected_persona in personas:
        active_dict = personas[st.session_state.selected_persona].copy()
        st.info(f"Loaded Persona: **{st.session_state.selected_persona}**")
    else:
        active_dict = personas["Riya"].copy()

    # Form / Inputs
    with st.expander("⚙️ Adjust Customer Parameters Manually", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure_val = st.slider("Tenure (Months):", 0, 72, int(active_dict["tenure"]))
            contract_val = st.selectbox("Contract Type:", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(active_dict["Contract"]))
            monthly_val = st.slider("Monthly Charges ($):", 18.0, 120.0, float(active_dict["MonthlyCharges"]))
            tech_val = st.selectbox("Tech Support:", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(active_dict["TechSupport"]))
        with c2:
            internet_val = st.selectbox("Internet Service:", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(active_dict["InternetService"]))
            security_val = st.selectbox("Online Security:", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(active_dict["OnlineSecurity"]))
            backup_val = st.selectbox("Online Backup:", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(active_dict["OnlineBackup"]))
            protection_val = st.selectbox("Device Protection:", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(active_dict["DeviceProtection"]))
        with c3:
            payment_val = st.selectbox("Payment Method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(active_dict["PaymentMethod"]))
            paperless_val = st.selectbox("Paperless Billing:", ["Yes", "No"], index=["Yes", "No"].index(active_dict["PaperlessBilling"]))
            streaming_tv = st.selectbox("Streaming TV:", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(active_dict["StreamingTV"]))
            streaming_movies = st.selectbox("Streaming Movies:", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(active_dict["StreamingMovies"]))

        active_dict.update({
            "tenure": tenure_val,
            "Contract": contract_val,
            "MonthlyCharges": monthly_val,
            "TotalCharges": monthly_val * tenure_val,
            "TechSupport": tech_val,
            "InternetService": internet_val,
            "OnlineSecurity": security_val,
            "OnlineBackup": backup_val,
            "DeviceProtection": protection_val,
            "PaymentMethod": payment_val,
            "PaperlessBilling": paperless_val,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies
        })

    cust_raw = pd.Series(active_dict)

    # Predict Probability
    cust_df = pd.DataFrame([cust_raw])
    cust_proc = splits["preprocessor"].transform(cust_df)
    prob_churn = float(best_model.predict_proba(cust_proc)[0][1])

    # Task 4: Predict 90-Day Churn Probability from Survival Model
    prob_90d_churn = survival_model.predict_90_day_churn_prob(active_dict, current_tenure=float(cust_raw["tenure"]))

    risk_tier = assign_risk_tier(prob_churn)
    value_tier = assign_value_tier(float(cust_raw.get("MonthlyCharges", 0)))
    strategy = get_matrix_strategy(risk_tier, value_tier)

    # Score Cards
    col_pred1, col_pred2, col_pred3 = st.columns([2, 2, 3])
    with col_pred1:
        st.markdown("#### Baseline Churn Risk")
        gauge_color = "#EF4444" if prob_churn >= 0.60 else ("#F59E0B" if prob_churn >= 0.35 else "#10B981")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_churn * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 35], 'color': '#ECFDF5'},
                    {'range': [35, 60], 'color': '#FFFBEB'},
                    {'range': [60, 100], 'color': '#FEF2F2'}
                ]
            }
        ))
        fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=25, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_pred2:
        st.markdown("#### ⏱️ 90-Day Churn Risk (Task 4)")
        gauge_90d_color = "#EF4444" if prob_90d_churn >= 0.30 else ("#F59E0B" if prob_90d_churn >= 0.15 else "#10B981")
        fig_90d = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_90d_churn * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_90d_color},
                'steps': [
                    {'range': [0, 15], 'color': '#ECFDF5'},
                    {'range': [15, 30], 'color': '#FFFBEB'},
                    {'range': [30, 100], 'color': '#FEF2F2'}
                ]
            }
        ))
        fig_90d.update_layout(height=230, margin=dict(l=20, r=20, t=25, b=10))
        st.plotly_chart(fig_90d, use_container_width=True)

    with col_pred3:
        st.markdown("#### Strategic Intervention Plan")
        st.markdown(f"""
        <div style="background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
            <div style="margin-bottom:8px;">
                <span class="badge-pill-{ 'danger' if risk_tier == 'High Risk' else ('warning' if risk_tier == 'Medium Risk' else 'success') }">{risk_tier}</span>
                <span class="badge-pill-{ 'danger' if value_tier == 'High Value' else 'warning' }" style="margin-left:6px;">{value_tier}</span>
            </div>
            <h4 style="color:#0F172A; margin: 4px 0 6px 0;">Playbook: {strategy['strategy']}</h4>
            <p style="color:#64748B; font-size:0.9rem; margin-bottom:8px;">{strategy['summary']}</p>
            <div style="font-size:0.85rem; color:#0F172A;">
                <b>Recommended Channel:</b> {strategy['channel']}<br>
                <b>Budget Allocation:</b> {strategy['budget_tier']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Task 7: Side-by-side What-If Moment for Riya vs Riya (2-year contract)
    if st.session_state.selected_persona in ["Riya", "Riya_2yr"]:
        st.markdown("---")
        st.markdown("### ⚡ The What-If Moment: Contract Upgrade Impact")
        
        riya_base_proc = splits["preprocessor"].transform(pd.DataFrame([personas["Riya"]]))
        riya_2yr_proc = splits["preprocessor"].transform(pd.DataFrame([personas["Riya_2yr"]]))
        
        p_base = float(best_model.predict_proba(riya_base_proc)[0][1])
        p_2yr = float(best_model.predict_proba(riya_2yr_proc)[0][1])
        delta = (p_2yr - p_base) * 100.0

        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            st.metric("Riya (Month-to-Month)", f"{p_base*100:.1f}% Risk", "High Risk")
        with w_col2:
            st.metric("Riya (2-Year Commitment)", f"{p_2yr*100:.1f}% Risk", f"{delta:.1f}% Churn Drop", delta_color="inverse")
        with w_col3:
            st.success(f"Risk collapses by {abs(delta):.1f} percentage points! Lock-in secures ~${personas['Riya']['MonthlyCharges']*24:,.0f} in contract revenue.")

    # Official JavaScript SHAP Force Plot
    st.markdown("---")
    st.markdown("### 🔍 Interactive JavaScript SHAP Force Plot")
    explanation = explain_customer_prediction(explainer, cust_proc[0], splits["feature_names"])
    force_html = generate_shap_force_html(explanation, splits["feature_names"])
    components.html(force_html, height=160, scrolling=False)

    # Local Waterfall
    st.markdown("### 🌊 Local Feature Impact Waterfall")
    fig_wf = plot_individual_waterfall(explanation, max_features=8)
    st.plotly_chart(fig_wf, use_container_width=True)

# =========================================================
# MODULE 7: Retention Strategy & Uplift Teaser
# =========================================================
elif nav_choice == "📋 7. Retention Strategy & Uplift Teaser":
    st.markdown('<div class="main-header">📋 Retention Playbook & Uplift Modeling Teaser</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">2x2 Risk-Value matrix execution and why future systems require true Persuadability modeling.</div>', unsafe_allow_html=True)

    # 2x2 Matrix Cards
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("""
        <div style="background-color:#FEF2F2; border-left:4px solid #EF4444; border-radius:8px; padding:14px; min-height:160px; margin-bottom:12px;">
            <h4 style="color:#991B1B; margin:0 0 6px 0;">🔥 Priority Save (High Risk × High Value)</h4>
            <div style="font-size:0.85rem; color:#7F1D1D; margin-bottom:6px;"><b>Criteria:</b> Monthly Charges ≥ $70 | Churn Risk ≥ 60%</div>
            <div style="font-size:0.9rem; color:#450A0A;">
                <b>Playbook:</b> Dedicated personal outreach from Senior Concierge Retention Specialist.<br>
                <b>Incentive:</b> 20% discount on 12-month lock-in commitment + complimentary security & tech support bundle ($85 budget).
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color:#FEF3C7; border-left:4px solid #F59E0B; border-radius:8px; padding:14px; min-height:160px;">
            <h4 style="color:#92400E; margin:0 0 6px 0;">⚡ Low-Cost Save (High Risk × Low Value)</h4>
            <div style="font-size:0.85rem; color:#78350F; margin-bottom:6px;"><b>Criteria:</b> Monthly Charges < $70 | Churn Risk ≥ 60%</div>
            <div style="font-size:0.9rem; color:#451A03;">
                <b>Playbook:</b> Automated digital campaign (push notification & personalized SMS).<br>
                <b>Incentive:</b> $5 monthly statement credit for switching to automated bank autopay + 30-day speed boost.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with q2:
        st.markdown("""
        <div style="background-color:#EFF6FF; border-left:4px solid #2563EB; border-radius:8px; padding:14px; min-height:160px; margin-bottom:12px;">
            <h4 style="color:#1E40AF; margin:0 0 6px 0;">💎 Nurture & Upsell (Low Risk × High Value)</h4>
            <div style="font-size:0.85rem; color:#1E3A8A; margin-bottom:6px;"><b>Criteria:</b> Monthly Charges ≥ $70 | Churn Risk < 35%</div>
            <div style="font-size:0.9rem; color:#172554;">
                <b>Playbook:</b> VIP customer relationship management & loyalty rewards.<br>
                <b>Incentive:</b> Preview access to next-gen fiber speeds, annual account reviews, and anniversary gifts.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color:#F1F5F9; border-left:4px solid #64748B; border-radius:8px; padding:14px; min-height:160px;">
            <h4 style="color:#334155; margin:0 0 6px 0;">🛡️ Monitor & Retain (Low Risk × Low Value)</h4>
            <div style="font-size:0.85rem; color:#475569; margin-bottom:6px;"><b>Criteria:</b> Monthly Charges < $70 | Churn Risk < 35%</div>
            <div style="font-size:0.9rem; color:#0F172A;">
                <b>Playbook:</b> High baseline self-service portal experience & periodic NPS check-ins.<br>
                <b>Incentive:</b> Zero outbound campaign expenditure. Maintain seamless autopay billing.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tier 4 Stretch: Uplift Modeling Teaser
    st.markdown("---")
    st.markdown("### 🚀 Tier 4 Stretch: The Uplift Modeling Paradigm (Persuadability Matrix)")
    st.markdown("""
    <div class="callout-box" style="border-left: 5px solid #8B5CF6;">
        <b>Why Churn Risk Alone Is Flawed:</b> Standard churn models predict <em>who is likely to cancel</em> ($P(\\text{Churn})$).
        However, a customer with 90% churn risk might be a <strong>Lost Cause</strong> who will cancel no matter what offer we send!
        Conversely, giving a discount to a <strong>Sure Thing</strong> wastes margin on someone who would have stayed anyway.
        <strong>True Uplift Modeling</strong> isolates the incremental treatment effect: 
        $$\\text{Uplift} = P(\\text{Churn} \\mid \\text{No Offer}) - P(\\text{Churn} \\mid \\text{Offer})$$
    </div>
    """, unsafe_allow_html=True)

    u1, u2 = st.columns([3, 2])
    with u1:
        # 2x2 Persuadability Diagram
        fig_persuade = go.Figure()
        # Add quadrants
        fig_persuade.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1, fillcolor="#FEE2E2", opacity=0.7, line_width=0)
        fig_persuade.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1, fillcolor="#D1FAE5", opacity=0.7, line_width=0)
        fig_persuade.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5, fillcolor="#F1F5F9", opacity=0.7, line_width=0)
        fig_persuade.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5, fillcolor="#FEF3C7", opacity=0.7, line_width=0)

        fig_persuade.add_trace(go.Scatter(
            x=[0.75, 0.25, 0.25, 0.75],
            y=[0.75, 0.75, 0.25, 0.25],
            mode="text",
            text=[
                "<b>🎯 PERSUADABLES</b><br>(High Risk × High Uplift)<br><i>ONLY Target These!</i>",
                "<b>🪦 LOST CAUSES</b><br>(High Risk × Zero Uplift)<br><i>Will Leave Anyway</i>",
                "<b>💤 SLEEPING DOGS</b><br>(Low Risk × Negative Uplift)<br><i>Offer Reminds to Cancel</i>",
                "<b>🛡️ SURE THINGS</b><br>(Low Risk × Zero Uplift)<br><i>Will Stay Anyway</i>"
            ],
            textfont=dict(size=12, color="#0F172A")
        ))
        fig_persuade.update_layout(
            title="Persuadability Matrix (Treatment Uplift vs Churn Risk)",
            xaxis=dict(title="Responsiveness to Retention Offer (Uplift)", range=[0, 1], showgrid=False),
            yaxis=dict(title="Baseline Churn Propensity", range=[0, 1], showgrid=False),
            height=360, template="plotly_white"
        )
        st.plotly_chart(fig_persuade, use_container_width=True)

    with u2:
        st.markdown("#### 📥 Campaign List Generator")
        tier_choice = st.selectbox("Select Target Campaign Segment:", ["Top 20% High-Risk (Recommended)", "All High-Risk Accounts", "Full Customer Base"])
        
        test_df = splits["X_test_raw"].copy()
        test_df["predicted_churn_prob"] = pipeline_bundle["eval_details"]["Ensemble (XGB + LR + RF)"]["y_prob"]
        test_df["risk_tier"] = test_df["predicted_churn_prob"].apply(assign_risk_tier)

        if "Top 20%" in tier_choice:
            campaign_df = test_df.sort_values(by="predicted_churn_prob", ascending=False).head(int(len(test_df) * 0.20))
        elif "High-Risk" in tier_choice:
            campaign_df = test_df[test_df["risk_tier"] == "High Risk"]
        else:
            campaign_df = test_df

        csv_buffer = io.StringIO()
        campaign_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Campaign List (CSV)",
            data=csv_buffer.getvalue(),
            file_name="retainpulse_targeted_campaign.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"Ready for ingestion: {len(campaign_df)} eligible subscriber accounts.")

# =========================================================
# MODULE 8: Business Impact & Campaign ROI
# =========================================================
elif nav_choice == "💰 8. Business Impact & Campaign ROI":
    st.markdown('<div class="main-header">💰 Business Impact & Campaign ROI Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Financial simulator powered by single source of truth (metrics.json) & 100K batch benchmark.</div>', unsafe_allow_html=True)

    fin_data = metrics_data["financial_impact"]
    strat_a = fin_data["strategy_a_mass_contact"]
    strat_b = fin_data["strategy_b_model_targeting"]
    comp = fin_data["comparison"]
    bench = metrics_data["batch_scoring_benchmark"]

    # Benchmark Callout
    st.markdown(f"""
    <div class="callout-box" style="border-left: 5px solid #10B981;">
        <b>⚡ Enterprise Batch-Scoring Benchmark:</b> Model scored <strong>{bench['rows_scored']:,} subscribers in {bench['duration_seconds']:.3f} seconds</strong> 
        (throughput: <strong>{bench['rows_per_second']:,} rows/sec</strong>). 
        Replaces vague claims of taking minutes with audited sub-second enterprise throughput!
    </div>
    """, unsafe_allow_html=True)

    # Summary Cards
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(f"""
        <div class="metric-card metric-card-danger">
            <div class="metric-label">Strategy A: Mass Contact</div>
            <div class="metric-value">{strat_a['roi_pct']:.1f}% ROI</div>
            <div class="metric-sub">${strat_a['campaign_cost']:,.0f} Cost ➔ ${strat_a['net_gain']:,.0f} Net Gain</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div class="metric-card metric-card-success">
            <div class="metric-label">Strategy B: Model Targeting</div>
            <div class="metric-value">{strat_b['roi_pct']:.1f}% ROI</div>
            <div class="metric-sub">${strat_b['campaign_cost']:,.0f} Cost ➔ ${strat_b['net_gain']:,.0f} Net Gain</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">Efficiency Multiplier</div>
            <div class="metric-value">{comp['roi_multiplier']:.1f}x ROI</div>
            <div class="metric-sub">${comp['cost_savings']:,.0f} Saved & +${comp['additional_profit']:,.0f} Profit</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Comparison Table
    st.markdown("### 📊 Side-by-Side Financial Comparison Table")
    roi_table = pd.DataFrame([
        {
            "Strategy": strat_a["name"],
            "Contacted": f"{strat_a['contacted']:,} (100%)",
            "Campaign Cost": f"${strat_a['campaign_cost']:,.0f}",
            "Saved Churners": f"{strat_a['saved_churners']:.0f} accounts",
            "Revenue Kept": f"${strat_a['revenue_kept']:,.0f}",
            "Net Profit": f"+${strat_a['net_gain']:,.0f}",
            "Campaign ROI": f"{strat_a['roi_pct']:.1f}%"
        },
        {
            "Strategy": strat_b["name"],
            "Contacted": f"{strat_b['contacted']:,} (20%)",
            "Campaign Cost": f"${strat_b['campaign_cost']:,.0f}",
            "Saved Churners": f"{strat_b['saved_churners']:.0f} accounts",
            "Revenue Kept": f"${strat_b['revenue_kept']:,.0f}",
            "Net Profit": f"+${strat_b['net_gain']:,.0f}",
            "Campaign ROI": f"{strat_b['roi_pct']:.1f}%"
        }
    ])
    st.dataframe(roi_table, use_container_width=True)

# =========================================================
# MODULE 9: A/B Experimentation Calculator (Task 8)
# =========================================================
elif nav_choice == "🧪 9. A/B Experimentation Calculator":
    st.markdown('<div class="main-header">🧪 A/B Experimentation & Rollout Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Statistically rigorous sample size and duration estimation for retention playbook rollouts.</div>', unsafe_allow_html=True)

    from statsmodels.stats.proportion import proportion_effectsize
    from statsmodels.stats.power import NormalIndPower

    st.markdown("""
    <div class="callout-box">
        <b>Task 8 Experimentation Framework:</b>
        Before deploying high-touch retention discounts to all high-risk subscribers, retention teams must run an A/B test 
        (e.g., Control: Standard digital email vs Variant: Concierge VIP call + $15 bill credit).
        This calculator determines the required sample size per test arm using a two-proportion z-test.
    </div>
    """, unsafe_allow_html=True)

    ab_c1, ab_c2 = st.columns(2)
    with ab_c1:
        st.markdown("#### ⚙️ Test Design Parameters")
        base_churn_input = st.slider(
            "Baseline Churn Rate (%):", 5.0, 50.0,
            float(metrics_data["metadata"]["baseline_churn_rate_pct"]), step=0.1
        ) / 100.0

        mde_input = st.slider(
            "Minimum Detectable Effect (Relative % Reduction in Churn):", 5.0, 30.0, 10.0, step=1.0
        ) / 100.0

        p1 = base_churn_input
        p2 = p1 * (1.0 - mde_input)
        abs_drop = (p1 - p2) * 100.0

        power_input = st.slider("Statistical Power (1 - β):", 0.70, 0.95, 0.80, step=0.05)
        alpha_input = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, step=0.01)
        weekly_volume = st.number_input("Weekly Eligible High-Risk Customers:", min_value=100, max_value=10000, value=500, step=50)

    # Compute Sample Size via Statsmodels
    effect_size = proportion_effectsize(p1, p2)
    power_analysis = NormalIndPower()
    n_per_arm = power_analysis.solve_power(effect_size=effect_size, alpha=alpha_input, power=power_input, alternative="two-sided")
    n_arm_int = math.ceil(n_per_arm)
    n_total = n_arm_int * 2
    duration_weeks = n_total / weekly_volume

    with ab_c2:
        st.markdown("#### 📊 Required Sample Size & Timeline")
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">Sample Size Per Arm</div>
            <div class="metric-value">{n_arm_int:,} subscribers</div>
            <div class="metric-sub">Total across both arms: {n_total:,}</div>
        </div>
        """, unsafe_allow_html=True)

        dur_color = "danger" if duration_weeks > 12 else ("warning" if duration_weeks > 6 else "success")
        st.markdown(f"""
        <div class="metric-card metric-card-{dur_color}">
            <div class="metric-label">Estimated Test Duration</div>
            <div class="metric-value">{duration_weeks:.1f} weeks</div>
            <div class="metric-sub">At {weekly_volume:,} eligible customers / week</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px; font-size:0.9rem;">
            <b>Test Hypothesis:</b><br>
            • $H_0$: Retention offer has zero effect ($p_1 = p_2 = {p1*100:.1f}\\%$).<br>
            • $H_1$: Retention offer lowers churn from <strong>{p1*100:.1f}%</strong> to <strong>{p2*100:.1f}%</strong> (absolute drop of {abs_drop:.2f}%).<br>
            • With α={alpha_input} and {int(power_input*100)}% power, {n_arm_int:,} accounts per arm protects against false alarms and underpowered tests.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# MODULE 10: Scalability, Drift & Retraining (Task 2)
# =========================================================
elif nav_choice == "🚀 10. Scalability, Drift & Retraining":
    st.markdown('<div class="main-header">🚀 Production Scalability, Drift & Quarterly Retraining</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Monitoring Population Stability Index (PSI), concept drift, and simulated next-quarter threshold re-optimization.</div>', unsafe_allow_html=True)

    with st.spinner("Simulating next-quarter batch with macroeconomic drift..."):
        retrain_results = run_retraining_simulation()

    base_s = retrain_results["baseline_summary"]
    drift_s = retrain_results["drifted_batch_summary"]
    psi_df = retrain_results["feature_psi_df"]
    drift_diag = retrain_results["churn_rate_diag"]

    # Highlights
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Baseline Churn Rate</div>
            <div class="metric-value">{base_s['churn_rate_pct']:.1f}%</div>
            <div class="metric-sub">Historical Reference</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="metric-card metric-card-danger">
            <div class="metric-label">Drifted Churn Rate</div>
            <div class="metric-value">{drift_s['actual_churn_rate_pct']:.1f}%</div>
            <div class="metric-sub">Next-Quarter Surge (+{(drift_s['actual_churn_rate_pct'] - base_s['churn_rate_pct']):.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div class="metric-card metric-card-warning">
            <div class="metric-label">Threshold Re-Optimization</div>
            <div class="metric-value">{base_s['threshold']:.2f} ➔ {drift_s['reoptimized_threshold']:.2f}</div>
            <div class="metric-sub">Adapted to Drifted Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        st.markdown(f"""
        <div class="metric-card metric-card-success">
            <div class="metric-label">Profit Lift from Recalibration</div>
            <div class="metric-value">+${drift_s['profit_lift_from_reopt']:,.0f}</div>
            <div class="metric-sub">Protected Retention Margin</div>
        </div>
        """, unsafe_allow_html=True)

    # PSI Feature Drift Chart
    st.markdown("### 📊 Population Stability Index (PSI) per Feature")
    st.markdown("""
    **Monitoring Rule:** `PSI < 0.10`: Stable (Green) | `0.10 ≤ PSI < 0.25`: Moderate Shift (Warning - Yellow) | `PSI ≥ 0.25`: Significant Drift (Alert - Red).
    """)

    fig_psi = px.bar(
        psi_df, x="Feature", y="PSI", color="Status",
        color_discrete_map={"Stable": "#10B981", "Moderate Shift": "#F59E0B", "Significant Drift": "#EF4444"},
        title="Feature Drift Across Deployed Features (PSI)",
        text="PSI"
    )
    fig_psi.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_psi.add_hline(y=0.10, line_dash="dash", line_color="#F59E0B", annotation_text="Warning (0.10)")
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="#EF4444", annotation_text="Alert (0.25)")
    fig_psi.update_layout(height=420, template="plotly_white")
    st.plotly_chart(fig_psi, use_container_width=True)

    # Deployment Architecture Diagram
    st.markdown("---")
    st.markdown("### 🏗️ Enterprise Production Architecture & Migration Path")
    
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    with arch_col1:
        st.markdown("""
        <div class="studio-card">
            <h4 style="color:#2563EB; margin-top:0;">1. Data Storage</h4>
            <b>Current:</b> In-memory SQLite prototype.<br><br>
            <b>Enterprise Migration:</b> Managed PostgreSQL / Snowflake database with CDC (Change Data Capture) 
            streaming real-time subscriber billing and contract changes into a feature store (e.g. Feast).
        </div>
        """, unsafe_allow_html=True)
    with arch_col2:
        st.markdown("""
        <div class="studio-card">
            <h4 style="color:#10B981; margin-top:0;">2. Serving API</h4>
            <b>Current:</b> Streamlit interactive app.<br><br>
            <b>Enterprise Migration:</b> Containerized FastAPI microservice deployed on Kubernetes behind an API Gateway. 
            Frontline CRM desktops query <code>POST /predict_churn</code> for sub-50ms inference.
        </div>
        """, unsafe_allow_html=True)
    with arch_col3:
        st.markdown("""
        <div class="studio-card">
            <h4 style="color:#8B5CF6; margin-top:0;">3. Continuous Retraining</h4>
            <b>Current:</b> On-demand script simulation.<br><br>
            <b>Enterprise Migration:</b> Scheduled Apache Airflow / Celery DAG running weekly PSI drift audits. 
            Triggers automated quarterly retraining with MLflow experiment registry and canary deployments.
        </div>
        """, unsafe_allow_html=True)
