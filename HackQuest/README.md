
# RetainPulse AI: Customer Churn Prediction & Retention Strategy
### DataNova 2026 — Problem Statement: DN-BU-01
**Finding who is likely to leave, why, and what to offer them.**

---

## 📌 Executive Summary

Subscription and recurring-revenue businesses (telecom, OTT, SaaS) lose substantial gross margins whenever customers cancel. In standard benchmark datasets like the IBM Telco Churn dataset, **about 1 in 4 customers churn (1,869 of 7,043, or 26.5%)**.

- A trivial dummy model that predicts *"nobody will churn"* achieves **73.5% accuracy**, yet is completely useless because it catches 0% of churners.
- Blind mass discounts waste precious budget on customers who would have stayed anyway.
- Raw probabilities alone do not provide frontline retention teams with actionable interventions.

**RetainPulse AI** solves this end-to-end:
1. **Ranks customers accurately** using tuned ML models (XGBoost, Random Forest, Logistic Regression, and Tuned Soft Voting Ensemble) evaluated on **PR-AUC**, **Recall**, and **Lift**.
2. **Interactive Decision Threshold Studio**: Dynamically tune operating decision thresholds (e.g. 0.35–0.40) to reach **Recall > 88%**, capturing over 330+ of 374 churners in testing!
3. **Explains root drivers** via **interactive JavaScript SHAP Force Plots** and waterfall charts at both global and individual subscriber levels.
4. **Translates predictions into action** via a **2x2 Risk-Value Retention Matrix** and automated personalized retention playbooks.
5. **Campaign Export**: One-click **Download Campaign List (CSV)** to feed directly into CRM/marketing automation pipelines.
6. **Calculates ROI and profit impact**, proving that precision model targeting doubles net gain at one-fifth the contact budget.

---

## 🏗️ Folder Structure

```
HackQuest/
├── app.py                         # Streamlit interactive web application
├── requirements.txt               # Python dependency requirements
├── README.md                      # Project documentation, methodology, and setup guide
├── data/
│   └── telco_churn.csv            # IBM Telco Customer Churn dataset (7,043 rows, 21 columns)
├── models/
│   └── best_churn_model.joblib    # Serialized ML model & preprocessor bundle
└── src/
    ├── __init__.py                # Package initializer
    ├── data_loader.py             # Data loading, cleaning, SQLite in-memory cohort queries
    ├── preprocessor.py            # Domain feature engineering & leakage-free transformers
    ├── models.py                  # LogReg, RF, XGBoost, Ensemble, multi-threshold sweep, Lift/Gains
    ├── explainability.py          # SHAP TreeExplainer, JS Force Plots, global & individual waterfalls
    └── retention_engine.py        # 2x2 Strategy matrix, personalized actions, ROI calculator
```

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: Streamlit, Streamlit Components v1
- **Machine Learning & Pipeline**: scikit-learn, XGBoost, joblib
- **Explainability**: SHAP (TreeExplainer, JavaScript Force Plots)
- **Data & In-Memory Analytics**: pandas, numpy, SQLite3
- **Visualization**: Plotly Interactive Charts, Matplotlib, Seaborn

---

## 🔬 Advanced Feature Engineering

1. **Non-Linear Tenure Representations**: `log_tenure` and tenure cohorts (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `>60m`).
2. **Billing Discrepancies & Ratios**:
   - `charge_discrepancy`: Delta between actual `TotalCharges` and expected lifetime spend (`tenure * MonthlyCharges`).
   - `monthly_to_total_ratio`: Immediate billing weight vs accumulated tenure.
   - `avg_charge_per_month`: Standardized historical monthly rate.
3. **Multi-Service & Bundle Aggregations**:
   - `num_services`: Total count of active telephone, security, and entertainment services.
   - `num_security_services`: Specific count of protection add-ons (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`).
   - `num_streaming_services`: `StreamingTV` + `StreamingMovies`.
   - `charges_per_service`: Normalized billing intensity.
4. **High-Risk Interaction Flags**:
   - `fiber_no_tech`: Fiber optic customer without active Tech Support (prime cancellation segment).
   - `fiber_no_sec`: Fiber optic customer without Cyber Security.
   - `short_tenure_m2m`: New customer ($\le 12$ months) on month-to-month terms.
   - `echeck_m2m`: Month-to-month contract paying via Electronic Check.
   - `is_alone`: Single customer without partner or dependents.
5. **Contract Numeric Commitment**:
   - `tenure_contract_interaction`: Interaction of tenure duration with contractual commitment lock-in.
   - `charge_vs_internet_mean`: Price differential against mean tariff of their specific internet tier.

---

## 🎯 Model Benchmark & Decision Threshold Tuning

| Model | Accuracy | Precision | Recall (Default 0.50) | Recall (Tuned Cutoff 0.35) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble (XGB + LR + RF)** | **0.755** | 0.526 | 0.794 | **0.888 (88.8%)** | **0.633** | **0.849** | **0.665** |
| **XGBoost (Tuned)** | 0.754 | 0.525 | 0.786 | **0.877 (87.7%)** | 0.630 | 0.849 | 0.664 |
| **Logistic Regression** | 0.740 | 0.506 | 0.791 | 0.864 (86.4%) | 0.617 | 0.849 | 0.665 |
| **Random Forest** | 0.754 | 0.525 | 0.794 | 0.858 (85.8%) | 0.632 | 0.845 | 0.655 |

> **Key Presentation Insight (Slide 3 & 6):**  
> Rather than relying on a default 0.50 probability cutoff, the app features an **Interactive Decision Threshold Studio** with presets (0.35 High Recall, 0.40 Max Profit). At **0.35 cutoff**, the tuned models achieve **88.8% Recall**, successfully identifying 332 out of 374 actual churners in test validation!

---

## 🚀 Setup & Execution Instructions

### 1. Open Workspace
```bash
cd HackQuest
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Access App in Browser
Navigate to:
```
http://localhost:8501
```

---

## 🌟 Application Modules

1. **📊 Executive EDA & SQLite Cohorts**:
   - Executive KPIs: Total Subscribers (7,043), Churners (1,869), Baseline Churn (26.54%), MRR (\$456K), Annual Revenue at Risk (\$1.67M).
   - Core behavioral charts: Churn by Contract, Tenure distribution, Monthly charges boxplot, Internet service breakdown.
   - In-memory SQLite cohort query runner with live SQL editor.

2. **🤖 Model Benchmarking & Threshold Studio**:
   - Prominent **Live Decision Threshold Studio** with one-click presets (High Recall, Max Profit, Balanced F1, High Precision).
   - Interactive slider updating Precision, Recall, Confusion Matrix, and Precision-Recall curve operating point in real time.
   - Comprehensive model comparison leaderboard.

3. **📈 Lift, Gains & Campaign Ranking**:
   - Decile-by-decile Lift chart and Cumulative Gains curve demonstrating **2.6x–2.9x lift in top deciles** (capturing >52% of churners in the top 20% of customers).

4. **🔍 Global SHAP Feature Explainability**:
   - Global mean |SHAP| feature ranking bar chart and interactive data table.
   - Detailed analysis of primary drivers: contract type, tenure, tech support deficiency, and relative price perception.

5. **🎯 Real-Time Predictor & SHAP Force Plot**:
   - Score test customers or configure custom customer profiles.
   - Real-time gauge for churn risk and customer value segment.
   - **Official interactive JavaScript SHAP Force Plot** rendered via Streamlit Components v1.
   - Counterfactual What-If tool to simulate immediate risk drops when upgrading contracts or adding tech support.

6. **📋 Retention Strategy & Campaign Export**:
   - Visual 2x2 Risk-Value Matrix cards with recommended retention playbooks.
   - **Prominent "Download Campaign List (CSV)"** button with audience filter (Top 20% High-Risk, High & Medium Risk, or All Customers).

7. **💰 Business Impact & Campaign ROI Planner**:
   - Financial simulator with adjustable customer base, contact cost, save rate, and CLV.
   - Side-by-side financial comparison: Mass Contact (\$281K spend, \$82K gain, 29% ROI) vs Model Targeting (\$56K spend, \$162K gain, 287% ROI).

# Customer_Churn_Prediction_System
