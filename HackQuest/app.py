"""
app.py
DataNova 2026 — Problem DN-BU-01: Customer Churn Prediction and Retention Strategy
Enhanced production-grade Streamlit application.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
import os

from src.data_loader import load_data, get_data_summary, SQLiteCohortExplorer
from src.preprocessor import prepare_train_test_split
from src.models import (
    train_all_models, evaluate_model, compare_models,
    calculate_cumulative_gains_and_lift, save_model_bundle
)
from src.explainability import (
    get_shap_explainer, compute_global_shap_importance,
    plot_shap_global_bar, explain_customer_prediction,
    generate_shap_force_html, plot_individual_waterfall
)
from src.retention_engine import (
    assign_risk_tier, assign_value_tier, get_matrix_strategy,
    generate_personalized_actions, calculate_campaign_roi
)

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
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 3px;
    }

    /* Badges */
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .badge-low {
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
    
    /* Callout Card */
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
    
    # Explainer on XGBoost
    xgb_model = trained_models["XGBoost (Tuned)"]
    explainer = get_shap_explainer(xgb_model)
    
    # Global importance
    sample_size = min(350, len(splits["X_test_proc"]))
    imp_df, raw_shap = compute_global_shap_importance(
        explainer, splits["X_test_proc"][:sample_size], splits["feature_names"], top_n=15
    )
    
    # Save best bundle
    save_model_bundle(trained_models["Ensemble (XGB + LR + RF)"], splits["preprocessor"])

    return {
        "splits": splits,
        "models": trained_models,
        "comparison_df": comparison_df,
        "eval_details": eval_details,
        "best_model": trained_models["Ensemble (XGB + LR + RF)"],
        "xgb_model": xgb_model,
        "explainer": explainer,
        "shap_importance_df": imp_df
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
with st.spinner("Optimizing feature pipelines & training tuned models..."):
    pipeline_bundle = get_trained_pipeline(df_raw)

splits = pipeline_bundle["splits"]
models = pipeline_bundle["models"]
comparison_df = pipeline_bundle["comparison_df"]
eval_details = pipeline_bundle["eval_details"]
best_model = pipeline_bundle["best_model"]
explainer = pipeline_bundle["explainer"]
shap_importance_df = pipeline_bundle["shap_importance_df"]

# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Navigation")
nav_choice = st.sidebar.radio(
    "Go to Module:",
    [
        "📊 1. Executive EDA & SQLite Cohorts",
        "🤖 2. Model Benchmarking & Threshold Studio",
        "📈 3. Lift, Gains & Campaign Ranking",
        "🔍 4. Global SHAP Feature Explainability",
        "🎯 5. Real-Time Predictor & SHAP Force Plot",
        "📋 6. Retention Strategy & Campaign Export",
        "💰 7. Business Impact & ROI Planner"
    ],
    index=0
)

st.sidebar.markdown("---")
pptx_file_path = "DataNova_2026_Customer_Churn_Retention_Strategy.pptx"
if os.path.exists(pptx_file_path):
    with open(pptx_file_path, "rb") as f:
        pptx_bytes = f.read()
    st.sidebar.download_button(
        label="📥 Download Deck (.pptx)",
        data=pptx_bytes,
        file_name="DataNova_2026_Customer_Churn_Retention_Strategy.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        use_container_width=True
    )

st.sidebar.markdown("""
<div style="font-size:0.8rem; color:#64748B;">
<b>DataNova 2026 Submission</b><br>
Problem: DN-BU-01 Customer Churn<br>
Stack: Streamlit, XGBoost, SHAP, SQLite
</div>
""", unsafe_allow_html=True)

# =========================================================
# MODULE 1: Executive EDA & SQLite Cohorts
# =========================================================
if nav_choice == "📊 1. Executive EDA & SQLite Cohorts":
    st.markdown('<div class="main-header">📊 Executive Overview & Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding subscriber behaviors, churn baseline, revenue at risk, and cohort patterns.</div>', unsafe_allow_html=True)

    summary = get_data_summary(df_raw)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Subscribers</div>
            <div class="metric-value">{summary['total_customers']:,}</div>
            <div class="metric-sub">Accounts in Scope</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-card-danger">
            <div class="metric-label">Churned Accounts</div>
            <div class="metric-value">{summary['churn_count']:,}</div>
            <div class="metric-sub">Lost Customers</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-card-warning">
            <div class="metric-label">Baseline Churn Rate</div>
            <div class="metric-value">{summary['churn_rate']:.1f}%</div>
            <div class="metric-sub">~1 in 4 Subscribers</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card metric-card-success">
            <div class="metric-label">Monthly Billing (MRR)</div>
            <div class="metric-value">${summary['monthly_revenue']:,.0f}</div>
            <div class="metric-sub">Gross Recurring Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card metric-card-danger">
            <div class="metric-label">Annual Revenue at Risk</div>
            <div class="metric-value">${summary['revenue_at_risk_annual']:,.0f}</div>
            <div class="metric-sub">Lost Run-Rate Per Year</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔍 Core Churn Behavioral Insights")
    row1_c1, row1_c2, row1_c3 = st.columns([1, 1, 1])

    with row1_c1:
        # Contract churn bar
        contract_churn = df_raw.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().fillna(0) * 100
        fig_contract = px.bar(
            contract_churn,
            barmode='group',
            title="Churn Rate by Contract Commitment",
            labels={'value': 'Percentage (%)', 'Contract': 'Contract'},
            color_discrete_map={'Yes': '#EF4444', 'No': '#3B82F6'}
        )
        fig_contract.update_layout(template="plotly_white", height=320, legend_title="Churn")
        st.plotly_chart(fig_contract, use_container_width=True)

    with row1_c2:
        # Churn Donut
        fig_pie = px.pie(
            df_raw,
            names='Churn',
            title="Customer Base Retention Split",
            color='Churn',
            color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'},
            hole=0.5
        )
        fig_pie.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1_c3:
        # Internet Service Churn
        net_churn = df_raw.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().fillna(0) * 100
        fig_net = px.bar(
            net_churn,
            barmode='group',
            title="Churn by Internet Service Type",
            labels={'value': 'Percentage (%)', 'InternetService': 'Internet'},
            color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'}
        )
        fig_net.update_layout(template="plotly_white", height=320, legend_title="Churn")
        st.plotly_chart(fig_net, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        fig_tenure = px.histogram(
            df_raw,
            x='tenure',
            color='Churn',
            nbins=36,
            barmode='overlay',
            opacity=0.75,
            title="Tenure Distribution: New vs Veteran Customers",
            color_discrete_map={'Yes': '#EF4444', 'No': '#3B82F6'}
        )
        fig_tenure.update_layout(template="plotly_white", height=320, xaxis_title="Tenure (Months)", yaxis_title="Subscribers")
        st.plotly_chart(fig_tenure, use_container_width=True)

    with row2_c2:
        fig_charges = px.box(
            df_raw,
            x='Contract',
            y='MonthlyCharges',
            color='Churn',
            title="Monthly Charges by Contract & Churn Status",
            color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'}
        )
        fig_charges.update_layout(template="plotly_white", height=320, yaxis_title="Monthly Charges ($)")
        st.plotly_chart(fig_charges, use_container_width=True)

    # SQLite Cohort Explorer
    st.markdown("---")
    st.markdown("### 🗄️ In-Memory SQLite Cohort Explorer (Presentation Step 1)")
    st.write("Live SQL querying against the `customers` table to slice cohorts by tenure, contract, and billing.")

    cohort_explorer = SQLiteCohortExplorer(df_raw)
    presets = cohort_explorer.get_preset_queries()

    col_sql_select, col_sql_btn = st.columns([3, 1])
    with col_sql_select:
        selected_query_name = st.selectbox("Select Preset Cohort Query:", list(presets.keys()))
    with col_sql_btn:
        st.write("")
        st.write("")
        run_sql_preset = st.button("🔄 Reload Preset SQL", use_container_width=True)

    default_sql = presets[selected_query_name]
    sql_input = st.text_area("SQL Query Editor", value=default_sql.strip(), height=150)

    if st.button("▶ Execute SQL Query"):
        query_res = cohort_explorer.query(sql_input)
        st.dataframe(query_res, use_container_width=True)

# =========================================================
# MODULE 2: Model Benchmarking & Threshold Studio
# =========================================================
elif nav_choice == "🤖 2. Model Benchmarking & Threshold Studio":
    st.markdown('<div class="main-header">🤖 Model Benchmarking & Decision Threshold Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Tuned XGBoost, Ensemble, Random Forest & Logistic Regression with Live Operating Threshold Tuning.</div>', unsafe_allow_html=True)

    # Threshold Studio Prominent Card
    st.markdown("""
    <div class="studio-card">
        <h3 style="margin-top:0; color:#1E293B;">🎛️ Live Decision Threshold Studio</h3>
        <p style="color:#64748B; font-size:0.9rem; margin-bottom:12px;">
            In subscription churn, default 0.50 cutoff misses many at-risk customers. Adjusting the cutoff increases Recall to catch <strong>85%+ of churners</strong> while controlling retention offer costs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Presets Buttons
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
    if "custom_thresh" not in st.session_state:
        st.session_state.custom_thresh = 0.38

    with c_btn1:
        if st.button("🎯 High Recall Focus (0.35)", use_container_width=True):
            st.session_state.custom_thresh = 0.35
    with c_btn2:
        if st.button("💰 Max Profit Strategy (0.40)", use_container_width=True):
            st.session_state.custom_thresh = 0.40
    with c_btn3:
        if st.button("⚖️ Balanced F1 Cutoff (0.48)", use_container_width=True):
            st.session_state.custom_thresh = 0.48
    with c_btn4:
        if st.button("🛡️ High Precision Focus (0.65)", use_container_width=True):
            st.session_state.custom_thresh = 0.65

    current_threshold = st.slider(
        "Current Operating Decision Threshold:",
        min_value=0.10, max_value=0.85, value=st.session_state.custom_thresh, step=0.01,
        help="Probability threshold above which an account is flagged as at-risk."
    )

    # Re-evaluate chosen model live
    selected_model_name = st.selectbox("Select Model to Inspect:", list(models.keys()), index=0)
    live_eval = evaluate_model(models[selected_model_name], splits["X_test_proc"], splits["y_test"], threshold=current_threshold)

    cm = live_eval["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total_churners = tp + fn
    recall_pct = (tp / total_churners) * 100 if total_churners > 0 else 0

    # Live Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card metric-card-success">
            <div class="metric-label">Recall (Sensitivity)</div>
            <div class="metric-value">{live_eval['recall'] * 100:.1f}%</div>
            <div class="metric-sub">{tp} of {total_churners} Churners Caught</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{live_eval['precision'] * 100:.1f}%</div>
            <div class="metric-sub">Accuracy of Offers Sent</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card metric-card-warning">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">{live_eval['f1']:.3f}</div>
            <div class="metric-sub">Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">{live_eval['roc_auc']:.3f}</div>
            <div class="metric-sub">Ranking Quality</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">PR-AUC</div>
            <div class="metric-value">{live_eval['pr_auc']:.3f}</div>
            <div class="metric-sub">Avg Precision Score</div>
        </div>
        """, unsafe_allow_html=True)

    # Side-by-side Visuals: Confusion Matrix + PR Tradeoff Curve
    col_cm, col_curve = st.columns(2)

    with col_cm:
        cm_labels = [["True Negatives (Kept)", "False Positives (Wasted Offer)"],
                     ["False Negatives (Missed Churn)", "True Positives (Caught Churn)"]]
        cm_text = [[f"{tn}<br><span style='font-size:11px'>{cm_labels[0][0]}</span>",
                    f"{fp}<br><span style='font-size:11px'>{cm_labels[0][1]}</span>"],
                   [f"{fn}<br><span style='font-size:11px'>{cm_labels[1][0]}</span>",
                    f"{tp}<br><span style='font-size:11px'>{cm_labels[1][1]}</span>"]]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Stay", "Predicted Churn"],
            y=["Actual Stay", "Actual Churn"],
            text=cm_text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False
        ))
        fig_cm.update_layout(
            title=f"Live Confusion Matrix (Threshold = {current_threshold:.2f})",
            height=370,
            template="plotly_white"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_curve:
        # Precision-Recall Tradeoff Curve with Operating Point Dot
        sweep_df = live_eval["threshold_sweep"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=sweep_df["Recall"],
            y=sweep_df["Precision"],
            mode="lines+markers",
            name="PR Curve Across Thresholds",
            line=dict(color="#2563EB", width=2.5)
        ))
        # Highlight current point
        fig_pr.add_trace(go.Scatter(
            x=[live_eval["recall"]],
            y=[live_eval["precision"]],
            mode="markers",
            name=f"Current ({current_threshold:.2f})",
            marker=dict(color="#EF4444", size=14, symbol="circle", line=dict(color="black", width=2))
        ))
        fig_pr.update_layout(
            title="Precision vs Recall Curve (Live Operating Marker)",
            xaxis_title="Recall (Proportion of Churners Caught)",
            yaxis_title="Precision",
            height=370,
            template="plotly_white"
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("### 🏆 Model Comparison Leaderboard")
    st.dataframe(
        comparison_df.style.highlight_max(subset=["PR-AUC", "Recall", "ROC-AUC", "F1-Score"], color="#D1FAE5"),
        use_container_width=True
    )

    # Threshold sweep table
    with st.expander("🔍 View Complete Multi-Threshold Sweep Table"):
        st.dataframe(sweep_df, use_container_width=True)

# =========================================================
# MODULE 3: Lift, Gains & Campaign Ranking
# =========================================================
elif nav_choice == "📈 3. Lift, Gains & Campaign Ranking":
    st.markdown('<div class="main-header">📈 Cumulative Gains, Lift & Campaign Deciles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demonstrating how ranking customers by risk concentrates retention effectiveness.</div>', unsafe_allow_html=True)

    best_probs = eval_details["Ensemble (XGB + LR + RF)"]["y_prob"]
    gains_df = calculate_cumulative_gains_and_lift(splits["y_test"], best_probs)

    top_20_gain = gains_df.iloc[1]["cum_gains_pct"]
    top_20_lift = gains_df.iloc[1]["lift"]

    st.success(f"""
    🎯 **Key Presentation Takeaway (Slide 6 & Slide 9):**  
    Targeting just the **Top 20% of customers by risk** captures **{top_20_gain:.1f}% of all actual churners**, delivering a **{top_20_lift:.2f}x Lift** over random contacting!
    """)

    c1, c2 = st.columns(2)

    with c1:
        fig_gains = go.Figure()
        fig_gains.add_trace(go.Scatter(
            x=[0] + list(gains_df["pct_customers"]),
            y=[0] + list(gains_df["cum_gains_pct"]),
            mode="lines+markers",
            name="Model Gains",
            line=dict(color="#10B981", width=3)
        ))
        fig_gains.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Random Guessing (Baseline)",
            line=dict(dash="dash", color="gray")
        ))
        fig_gains.update_layout(
            title="Cumulative Gains Chart (% of Churners Captured)",
            xaxis_title="% of Total Customers Contacted",
            yaxis_title="% of Total Churners Caught",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_gains, use_container_width=True)

    with c2:
        fig_lift = px.bar(
            gains_df,
            x="decile",
            y="lift",
            title="Lift Multiplier Across Deciles",
            text="lift",
            color="lift",
            color_continuous_scale="Viridis"
        )
        fig_lift.update_traces(texttemplate='%{text:.2f}x', textposition='outside')
        fig_lift.update_layout(height=400, template="plotly_white", xaxis_title="Decile (Sorted by Churn Probability)", yaxis_title="Lift Multiplier")
        st.plotly_chart(fig_lift, use_container_width=True)

    st.markdown("### 📊 Decile-by-Decile Lift Table")
    formatted_gains = gains_df.copy()
    formatted_gains["pct_customers"] = formatted_gains["pct_customers"].apply(lambda x: f"{x:.1f}%")
    formatted_gains["cum_gains_pct"] = formatted_gains["cum_gains_pct"].apply(lambda x: f"{x:.1f}%")
    formatted_gains["churn_rate_pct"] = formatted_gains["churn_rate_pct"].apply(lambda x: f"{x:.1f}%")
    formatted_gains["lift"] = formatted_gains["lift"].apply(lambda x: f"{x:.2f}x")
    st.dataframe(formatted_gains, use_container_width=True)

# =========================================================
# MODULE 4: Global SHAP Feature Explainability
# =========================================================
elif nav_choice == "🔍 4. Global SHAP Feature Explainability":
    st.markdown('<div class="main-header">🔍 Global Model Explainability (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interpreting behavioral drivers that systematically dictate customer cancellations.</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([3, 2])

    with c_left:
        fig_shap_bar = plot_shap_global_bar(shap_importance_df)
        st.plotly_chart(fig_shap_bar, use_container_width=True)

    with c_right:
        st.markdown("### 📋 Top SHAP Feature Ranking")
        st.dataframe(shap_importance_df, height=360, use_container_width=True)
        st.markdown("""
        **Interpretation Summary:**
        - **Contract Commitment**: Month-to-month contracts have zero lock-in, driving top churn attribution.
        - **Tenure Non-Linearity**: Churn risk is concentrated in months 1–12 before dropping sharply.
        - **Tech Support & Security**: Accounts lacking security services churn at over double the rate.
        - **Fiber Pricing Discrepancies**: High monthly fiber bills without tech support lead to churn.
        """)

# =========================================================
# MODULE 5: Real-Time Predictor & SHAP Force Plot
# =========================================================
elif nav_choice == "🎯 5. Real-Time Predictor & SHAP Force Plot":
    st.markdown('<div class="main-header">🎯 Real-Time Churn Predictor & SHAP Force Plot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate individual customer risk, inspect interactive SHAP force plots, and simulate counterfactuals.</div>', unsafe_allow_html=True)

    input_mode = st.radio("Select Customer Profiling Method:", ["Choose Existing Customer from Test Set", "Configure Custom Customer Attributes"], horizontal=True)

    if input_mode == "Choose Existing Customer from Test Set":
        test_ids = list(splits["id_test"])
        selected_cust_id = st.selectbox("Search Customer ID:", test_ids[:600])
        idx = list(splits["id_test"]).index(selected_cust_id)
        cust_raw = splits["X_test_raw"].iloc[idx].copy()
    else:
        st.markdown("#### 👤 Custom Subscriber Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure_val = st.slider("Tenure (Months):", 0, 72, 8)
            contract_val = st.selectbox("Contract Type:", ["Month-to-month", "One year", "Two year"], index=0)
            monthly_val = st.slider("Monthly Charges ($):", 18.0, 120.0, 82.0)
            tech_val = st.selectbox("Tech Support:", ["No", "Yes", "No internet service"], index=0)
        with c2:
            internet_val = st.selectbox("Internet Service:", ["Fiber optic", "DSL", "No"], index=0)
            security_val = st.selectbox("Online Security:", ["No", "Yes", "No internet service"], index=0)
            backup_val = st.selectbox("Online Backup:", ["No", "Yes", "No internet service"], index=0)
            protection_val = st.selectbox("Device Protection:", ["No", "Yes", "No internet service"], index=0)
        with c3:
            payment_val = st.selectbox("Payment Method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
            paperless_val = st.selectbox("Paperless Billing:", ["Yes", "No"], index=0)
            streaming_tv = st.selectbox("Streaming TV:", ["Yes", "No", "No internet service"], index=1)
            streaming_movies = st.selectbox("Streaming Movies:", ["Yes", "No", "No internet service"], index=1)

        cust_raw = pd.Series({
            "gender": "Female",
            "SeniorCitizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "tenure": tenure_val,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet_val,
            "OnlineSecurity": security_val,
            "OnlineBackup": backup_val,
            "DeviceProtection": protection_val,
            "TechSupport": tech_val,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract_val,
            "PaperlessBilling": paperless_val,
            "PaymentMethod": payment_val,
            "MonthlyCharges": monthly_val,
            "TotalCharges": monthly_val * tenure_val
        })

    # Predict
    cust_df = pd.DataFrame([cust_raw])
    cust_proc = splits["preprocessor"].transform(cust_df)
    prob_churn = float(best_model.predict_proba(cust_proc)[0][1])

    risk_tier = assign_risk_tier(prob_churn)
    value_tier = assign_value_tier(float(cust_raw.get("MonthlyCharges", 0)))
    strategy = get_matrix_strategy(risk_tier, value_tier)

    # Score Cards
    col_pred1, col_pred2, col_pred3 = st.columns([2, 2, 3])
    with col_pred1:
        st.markdown("#### Churn Probability")
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
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=25, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_pred2:
        st.markdown("#### Segmentation & Stance")
        badge_class = "badge-high" if risk_tier == "High Risk" else ("badge-medium" if risk_tier == "Medium Risk" else "badge-low")
        st.markdown(f"""
        <div style="margin-top: 20px;">
            <p><strong>Risk Level:</strong> <span class="{badge_class}">{risk_tier}</span></p>
            <p><strong>Customer Value:</strong> <strong>{value_tier}</strong> (${cust_raw.get('MonthlyCharges', 0):.2f}/mo)</p>
            <p><strong>Strategy:</strong> <strong>{strategy['strategy']}</strong></p>
            <p><strong>Channel:</strong> {strategy['channel']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_pred3:
        st.markdown("#### Retention Playbook Directive")
        st.info(f"**{strategy['strategy']} ({strategy['budget_tier']})**\n\n{strategy['summary']}")

    # SHAP FORCE PLOT (Official JS HTML)
    st.markdown("---")
    st.markdown("### ⚡ Interactive SHAP Force Plot")
    st.caption("Visualizing how each feature shifts the baseline expected churn log-odds (Higher = Red, Lower = Blue).")

    explanation = explain_customer_prediction(explainer, cust_proc[0], splits["feature_names"])
    force_html = generate_shap_force_html(explanation, splits["feature_names"])
    components.html(force_html, height=160, scrolling=True)

    # Waterfall Chart
    fig_waterfall = plot_individual_waterfall(explanation)
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # WHAT-IF COUNTERFACTUAL SIMULATOR
    st.markdown("---")
    st.markdown("### 🧪 What-If Counterfactual Tool (Slide 8)")
    st.write("Simulate retention interventions (e.g. upgrading to an annual contract or adding Tech Support) and see the churn risk drop in real-time.")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        whatif_contract = st.selectbox("What-If Contract:", ["Month-to-month", "One year", "Two year"], index=1 if "month" in str(cust_raw.get("Contract", "")).lower() else 0)
    with w2:
        whatif_tech = st.selectbox("What-If Tech Support:", ["Yes", "No"], index=0)
    with w3:
        whatif_security = st.selectbox("What-If Online Security:", ["Yes", "No"], index=0)
    with w4:
        whatif_payment = st.selectbox("What-If Payment:", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], index=0)

    whatif_raw = cust_raw.copy()
    whatif_raw["Contract"] = whatif_contract
    whatif_raw["TechSupport"] = whatif_tech
    whatif_raw["OnlineSecurity"] = whatif_security
    whatif_raw["PaymentMethod"] = whatif_payment

    whatif_df = pd.DataFrame([whatif_raw])
    whatif_proc = splits["preprocessor"].transform(whatif_df)
    whatif_prob = float(best_model.predict_proba(whatif_proc)[0][1])

    delta = whatif_prob - prob_churn
    whatif_risk = assign_risk_tier(whatif_prob)

    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Original Churn Risk", f"{prob_churn * 100:.1f}%")
    res_c2.metric("Counterfactual Risk", f"{whatif_prob * 100:.1f}%", delta=f"{delta * 100:.1f}%", delta_color="inverse")
    res_c3.metric("New Risk Tier", whatif_risk)

# =========================================================
# MODULE 6: Retention Strategy & Campaign Export
# =========================================================
elif nav_choice == "📋 6. Retention Strategy & Campaign Export":
    st.markdown('<div class="main-header">📋 Retention Strategy & Campaign List Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mapping risk and customer value to concrete retention actions, with one-click campaign roster export.</div>', unsafe_allow_html=True)

    # 2x2 Matrix Cards
    st.markdown("### 🗂️ 2x2 Risk-Value Retention Framework (Slide 7)")
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.markdown("""
        <div style="background-color:#FEE2E2; border-left:4px solid #EF4444; border-radius:8px; padding:14px; min-height:160px;">
            <strong style="color:#991B1B;">🔥 Priority Save</strong><br>
            <span style="font-size:0.8rem; color:#7F1D1D;">High Risk × High Value</span>
            <p style="font-size:0.82rem; color:#450A0A; margin-top:8px;">
                Dedicated VIP phone call + 20% discount on 1-year contract lock-in + free security bundle.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with m_c2:
        st.markdown("""
        <div style="background-color:#FEF3C7; border-left:4px solid #F59E0B; border-radius:8px; padding:14px; min-height:160px;">
            <strong style="color:#92400E;">⚡ Low-Cost Save</strong><br>
            <span style="font-size:0.8rem; color:#78350F;">High Risk × Low Value</span>
            <p style="font-size:0.82rem; color:#451A03; margin-top:8px;">
                Automated email/SMS sequence with $5 statement credit and free 3-month TechSupport trial.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with m_c3:
        st.markdown("""
        <div style="background-color:#EFF6FF; border-left:4px solid #3B82F6; border-radius:8px; padding:14px; min-height:160px;">
            <strong style="color:#1E40AF;">💎 Nurture & Upsell</strong><br>
            <span style="font-size:0.8rem; color:#1E3A8A;">Low Risk × High Value</span>
            <p style="font-size:0.82rem; color:#172554; margin-top:8px;">
                Loyalty club perks, preview of higher fiber speed tier, and relationship check-ins.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with m_c4:
        st.markdown("""
        <div style="background-color:#F1F5F9; border-left:4px solid #64748B; border-radius:8px; padding:14px; min-height:160px;">
            <strong style="color:#334155;">🛡️ Monitor & Retain</strong><br>
            <span style="font-size:0.8rem; color:#475569;">Low Risk × Low Value</span>
            <p style="font-size:0.82rem; color:#0F172A; margin-top:8px;">
                Standard self-service maintenance; quarterly NPS survey; zero proactive marketing expense.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📥 Campaign List Generator & Exporter")

    with st.spinner("Scoring batch customers..."):
        test_raw = splits["X_test_raw"].copy()
        test_proc = splits["X_test_proc"]
        y_probs = best_model.predict_proba(test_proc)[:, 1]

        batch_results = []
        for i in range(len(test_raw)):
            row = test_raw.iloc[i]
            prob = float(y_probs[i])
            r_tier = assign_risk_tier(prob)
            v_tier = assign_value_tier(float(row.get("MonthlyCharges", 0)))
            strat = get_matrix_strategy(r_tier, v_tier)
            actions = generate_personalized_actions(row, prob)
            primary_action = actions[0]["action"] if actions else "Standard Retention"

            batch_results.append({
                "CustomerID": splits["id_test"][i],
                "Churn_Probability": round(prob, 4),
                "Risk_Tier": r_tier,
                "Value_Tier": v_tier,
                "Strategic_Playbook": strat["strategy"],
                "Assigned_Channel": strat["channel"],
                "Primary_Action": primary_action,
                "MonthlyCharges": row.get("MonthlyCharges", 0),
                "Tenure": row.get("tenure", 0),
                "Contract": row.get("Contract", "")
            })

        batch_df = pd.DataFrame(batch_results).sort_values(by="Churn_Probability", ascending=False)

    # Filter controls
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        campaign_filter = st.radio(
            "Filter Campaign Target Audience:",
            ["Top 20% Highest Risk Accounts (Recommended - Maximizes ROI)", "High Risk & Medium Risk Accounts", "All Scored Customers"],
            horizontal=True
        )

    if "Top 20%" in campaign_filter:
        cutoff_n = int(len(batch_df) * 0.20)
        export_df = batch_df.iloc[:cutoff_n].copy()
    elif "High Risk & Medium" in campaign_filter:
        export_df = batch_df[batch_df["Risk_Tier"].isin(["High Risk", "Medium Risk"])].copy()
    else:
        export_df = batch_df.copy()

    # Pre-download Campaign Metrics
    camp_m1, camp_m2, camp_m3 = st.columns(3)
    camp_m1.metric("Accounts in Campaign", f"{len(export_df):,}")
    camp_m2.metric("Monthly Billing in Campaign", f"${export_df['MonthlyCharges'].sum():,.2f}")
    camp_m3.metric("Avg Churn Risk", f"{export_df['Churn_Probability'].mean() * 100:.1f}%")

    # PROMINENT DOWNLOAD BUTTON AT TOP
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Campaign List (CSV)",
        data=csv_buffer.getvalue(),
        file_name="targeted_retention_campaign_list.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.dataframe(export_df, use_container_width=True)

# =========================================================
# MODULE 7: Business Impact & ROI Planner
# =========================================================
elif nav_choice == "💰 7. Business Impact & ROI Planner":
    st.markdown('<div class="main-header">💰 Business Impact & Campaign ROI Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating financial returns of model-driven targeting vs contacting everyone (Slide 9).</div>', unsafe_allow_html=True)

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        total_subscribers = st.number_input("Total Subscriber Base:", min_value=1000, max_value=500000, value=7043, step=500)
        baseline_churn = st.slider("Baseline Churn Rate (%):", 5.0, 50.0, 26.5, step=0.5) / 100.0
    with c_p2:
        contact_cost = st.slider("Retention Cost per Contact ($):", 5.0, 150.0, 40.0, step=5.0)
        save_success_rate = st.slider("Offer Save / Acceptance Rate (%):", 5.0, 60.0, 25.0, step=1.0) / 100.0
    with c_p3:
        avg_annual_clv = st.number_input("Annual Revenue per Saved Customer ($):", min_value=200.0, max_value=3000.0, value=780.0, step=50.0)
        target_top_pct = st.slider("Model Targeting Cut-off (% of Base):", 5.0, 50.0, 20.0, step=5.0) / 100.0

    roi_data = calculate_campaign_roi(
        total_customers=total_subscribers,
        baseline_churn_rate=baseline_churn,
        avg_annual_revenue=avg_annual_clv,
        contact_cost=contact_cost,
        save_rate=save_success_rate,
        top_pct_targeted=target_top_pct,
        top_pct_churn_capture_rate=0.60
    )

    ev = roi_data["everyone"]
    md = roi_data["model"]
    cmp = roi_data["comparison"]

    st.markdown("### ⚖️ Strategy Financial Comparison")
    col_e, col_m = st.columns(2)

    with col_e:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 2px solid #CBD5E1; border-radius: 8px; padding: 20px;">
            <h3 style="color: #475569; margin-top: 0;">Strategy A: Contact Everyone (Mass Blast)</h3>
            <p><strong>Customers Contacted:</strong> {ev['contacted']:,} (100%)</p>
            <p><strong>Campaign Expenditure:</strong> <span style="color:#EF4444; font-weight:700;">${ev['campaign_cost']:,.0f}</span></p>
            <p><strong>Churners Saved:</strong> {ev['churners_saved']:,}</p>
            <p><strong>Gross Revenue Kept:</strong> ${ev['revenue_kept']:,.0f}</p>
            <hr style="margin: 10px 0;">
            <h4 style="color: #0F172A; margin: 0;">Net Financial Gain: <span style="color: #2563EB;">${ev['net_gain']:,.0f}</span></h4>
            <p style="margin: 5px 0 0 0; color: #64748B;">Campaign ROI: <strong>{ev['roi_pct']:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col_m:
        st.markdown(f"""
        <div style="background-color: #ECFDF5; border: 2px solid #10B981; border-radius: 8px; padding: 20px;">
            <h3 style="color: #065F46; margin-top: 0;">Strategy B: Model-Driven Targeting (Top {md['target_pct']:.0f}%)</h3>
            <p><strong>Customers Contacted:</strong> {md['contacted']:,} ({md['target_pct']:.0f}%)</p>
            <p><strong>Campaign Expenditure:</strong> <span style="color:#10B981; font-weight:700;">${md['campaign_cost']:,.0f}</span></p>
            <p><strong>Churners Saved:</strong> {md['churners_saved']:,}</p>
            <p><strong>Gross Revenue Kept:</strong> ${md['revenue_kept']:,.0f}</p>
            <hr style="margin: 10px 0;">
            <h4 style="color: #065F46; margin: 0;">Net Financial Gain: <span style="color: #059669;">${md['net_gain']:,.0f}</span></h4>
            <p style="margin: 5px 0 0 0; color: #047857;">Campaign ROI: <strong>{md['roi_pct']:.1f}%</strong> ({cmp['roi_multiplier']:.1f}x Higher ROI!)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    strat_df = pd.DataFrame([
        {"Strategy": "Contact Everyone", "Metric": "Spend ($)", "Amount": ev["campaign_cost"]},
        {"Strategy": "Contact Everyone", "Metric": "Net Revenue Kept ($)", "Amount": ev["net_gain"]},
        {"Strategy": "Target Top 20% (Model)", "Metric": "Spend ($)", "Amount": md["campaign_cost"]},
        {"Strategy": "Target Top 20% (Model)", "Metric": "Net Revenue Kept ($)", "Amount": md["net_gain"]}
    ])

    fig_cmp = px.bar(
        strat_df,
        x="Strategy",
        y="Amount",
        color="Metric",
        barmode="group",
        title="Cost vs Net Revenue Retained: Mass Contact vs Precision Model Targeting",
        color_discrete_map={"Spend ($)": "#EF4444", "Net Revenue Kept ($)": "#10B981"}
    )
    fig_cmp.update_layout(template="plotly_white", yaxis_title="Amount ($)")
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.success(f"""
    💡 **Bottom-Line Business Conclusion:**  
    By prioritizing high-risk accounts via XGBoost rather than mass-discounting all subscribers:  
    - You save **${cmp['cost_savings']:,.0f} in wasted retention spend** (spending 80% less).  
    - You generate **${cmp['additional_profit']:,.0f} more net profit**.  
    - Campaign ROI increases from **{ev['roi_pct']:.0f}%** to **{md['roi_pct']:.0f}%**!
    """)
