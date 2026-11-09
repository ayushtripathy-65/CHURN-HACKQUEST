"""
create_presentation.py
Generates the complete 16:9 PowerPoint Presentation (.pptx)
for DataNova 2026 Problem Statement DN-BU-01 based on the updated project.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Theme Colors
    COLOR_PRIMARY = RGBColor(15, 23, 42)      # Slate 900
    COLOR_ACCENT = RGBColor(37, 99, 235)      # Blue 600
    COLOR_SUCCESS = RGBColor(16, 185, 129)    # Emerald 500
    COLOR_DANGER = RGBColor(239, 68, 68)      # Red 500
    COLOR_MUTED = RGBColor(100, 116, 139)     # Slate 500
    COLOR_BG_CARD = RGBColor(248, 250, 252)   # Slate 50
    COLOR_WHITE = RGBColor(255, 255, 255)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="DataNova 2026 · Problem DN-BU-01"):
        # Category Banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY

    def add_card(slide, left, top, width, height, bg_color=COLOR_BG_CARD, border_color=COLOR_ACCENT):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
        return shape

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_PRIMARY
    bg1.line.fill.background()

    t_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "DATANOVA 2026  ·  PROBLEM DN-BU-01"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_SUCCESS
    p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "Customer Churn Prediction &\nTargeted Retention Strategy"
    p1.font.size = Pt(38)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_after = Pt(18)

    p2 = tf1.add_paragraph()
    p2.text = "Finding Who is Likely to Leave, Why, and What Exactly to Offer Them"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_after = Pt(36)

    p3 = tf1.add_paragraph()
    p3.text = "[Your Name]  ·  B.Tech Computer Science & Engineering  ·  [College / University Name]  ·  [Team Name]"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(148, 163, 184)

    # ==========================================
    # SLIDE 2: Why Churn Matters & Why Accuracy is Wrong
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Why Churn Matters, and Why Accuracy is the Wrong Score")

    # Left Column Card: The Problem
    add_card(slide2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), COLOR_BG_CARD, COLOR_DANGER)
    box2_l = slide2.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tf2_l = box2_l.text_frame
    tf2_l.word_wrap = True

    p = tf2_l.paragraphs[0]
    p.text = "The High Cost of Customer Churn"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_DANGER
    p.space_after = Pt(12)

    bullets_l = [
        ("Subscription recurring revenue", "Subscription businesses (Telecom, SaaS, OTT) bleed lifetime customer value each time a subscriber cancels."),
        ("Significant baseline churn", "In our dataset, ~1 in every 4 customers left (1,869 of 7,043, a 26.54% churn rate)."),
        ("Mass discounts waste margins", "Offering retention discounts to everyone burns budget on loyal customers who would have stayed anyway."),
        ("Probability alone is useless", "A bare probability score doesn't tell frontline retention agents what concrete offer will prevent cancellation.")
    ]
    for title, desc in bullets_l:
        p = tf2_l.add_paragraph()
        p.text = f"• {title}: {desc}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(8)

    # Right Column Card: Why Accuracy Fails
    add_card(slide2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), COLOR_BG_CARD, COLOR_ACCENT)
    box2_r = slide2.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf2_r = box2_r.text_frame
    tf2_r.word_wrap = True

    p = tf2_r.paragraphs[0]
    p.text = "The Accuracy Paradox in Imbalanced Churn"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p.space_after = Pt(12)

    bullets_r = [
        ("The 'Dummy' baseline trap", "A trivial dummy model predicting 'Nobody will churn' achieves 73.5% accuracy—yet catches 0% of churners and saves $0."),
        ("PR-AUC & Recall are true metrics", "Because only 26.5% churn, PR-AUC and Recall properly penalize missed churners and wasted campaigns."),
        ("Cut-off by expected profit", "We replace the naive 0.50 threshold with an interactive profit-maximizing threshold (0.35–0.40) to maximize net saved revenue."),
        ("Key Presentation Takeaway", "The ultimate goal is spending the retention budget with precision, not maximizing a deceptive accuracy score.")
    ]
    for title, desc in bullets_r:
        p = tf2_r.add_paragraph()
        p.text = f"• {title}: {desc}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(8)

    # ==========================================
    # SLIDE 3: Goals & Success Criteria
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "What I Set Out to Do, and How I Judge It")

    # Table layout
    rows, cols = 6, 3
    left, top, width, height = Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8)
    table_shape = slide3.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(3.5)
    table.columns[2].width = Inches(5.0)

    headers = ["Goal / Objective", "How We Measure It", "Core Engineering Focus"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(14)
            p.font.color.rgb = COLOR_WHITE

    goals_data = [
        ("Catch At-Risk Subscribers", "Recall, Precision, PR-AUC, ROC-AUC", "Tuned Soft Voting Ensemble & class weighting to maximize churn capture."),
        ("Rank Limited-Budget Campaign", "Decile Lift & Cumulative Gains (Top 20%)", "Concentrate 80% of retention budget where over 52% of churners reside."),
        ("Explain Every Prediction", "Global & Individual SHAP Attributions", "SHAP TreeExplainer JavaScript Force Plots + Plotly feature waterfalls."),
        ("Turn Insights into Action", "2x2 Risk-Value Matrix & Playbook", "Differentiating high-touch VIP saves from automated digital nudges."),
        ("Prove Business ROI", "Net Revenue Retained & Campaign ROI %", "Interactive financial simulator validating +$162K net gain vs mass contact.")
    ]

    for i, row in enumerate(goals_data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_WHITE if i % 2 == 0 else COLOR_BG_CARD
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 4: Data & Preprocessing Insights
    # ==========================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Dataset Architecture & Preprocessing (IBM Telco Churn)")

    add_card(slide4, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8), COLOR_BG_CARD, COLOR_ACCENT)
    b4_1 = slide4.shapes.add_textbox(Inches(0.9), Inches(1.9), Inches(3.4), Inches(4.5))
    tf4_1 = b4_1.text_frame
    tf4_1.word_wrap = True
    p = tf4_1.paragraphs[0]
    p.text = "Dataset Dimensions"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_ACCENT
    p.space_after = Pt(8)

    pts1 = [
        "7,043 Total Customers across 21 original columns.",
        "Target Balance: 1,869 churners (26.5%) vs 5,174 retained (73.5%).",
        "Total Monthly Revenue: $456,116 / mo.",
        "Annual Revenue at Risk: $1,669,570 from churned accounts.",
        "Demographics, account tenure, contract terms, payment, & service add-ons."
    ]
    for pt in pts1:
        p = tf4_1.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(12)
        p.space_after = Pt(6)

    add_card(slide4, Inches(4.8), Inches(1.8), Inches(3.6), Inches(4.8), COLOR_BG_CARD, COLOR_DANGER)
    b4_2 = slide4.shapes.add_textbox(Inches(4.9), Inches(1.9), Inches(3.4), Inches(4.5))
    tf4_2 = b4_2.text_frame
    tf4_2.word_wrap = True
    p = tf4_2.paragraphs[0]
    p.text = "Data Cleaning Nuances"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_DANGER
    p.space_after = Pt(8)

    pts2 = [
        "TotalCharges whitespace bug: 11 records stored as blank text string ' '.",
        "Root cause identified: Brand new customers with tenure = 0 months.",
        "Solution: Cleanly coerced to float and imputed to 0.0.",
        "Leakage Prevention: Dropped customerID and target-leaking fields.",
        "Checked duplicate IDs and standardized SeniorCitizen binary mappings."
    ]
    for pt in pts2:
        p = tf4_2.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(12)
        p.space_after = Pt(6)

    add_card(slide4, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.8), COLOR_BG_CARD, COLOR_SUCCESS)
    b4_3 = slide4.shapes.add_textbox(Inches(8.9), Inches(1.9), Inches(3.5), Inches(4.5))
    tf4_3 = b4_3.text_frame
    tf4_3.word_wrap = True
    p = tf4_3.paragraphs[0]
    p.text = "Telco Churn Patterns"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_SUCCESS
    p.space_after = Pt(8)

    pts3 = [
        "Contract Risk: Month-to-month contracts churn at ~42.7% (vs 2.8% on 2-year).",
        "Tenure Vulnerability: Highest attrition occurs in months 1–12.",
        "Service Disconnect: Fiber optic users churn at 41.9% when lacking tech support.",
        "Payment Friction: Electronic check users show 45.3% churn.",
        "Autopay Stickiness: Automatic bank/card users churn at only ~16%."
    ]
    for pt in pts3:
        p = tf4_3.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(12)
        p.space_after = Pt(6)

    # ==========================================
    # SLIDE 5: Advanced Feature Engineering
    # ==========================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "End-to-End Pipeline & High-Leverage Feature Engineering")

    add_card(slide5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), COLOR_BG_CARD, COLOR_ACCENT)
    b5_l = slide5.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tf5_l = b5_l.text_frame
    tf5_l.word_wrap = True
    p = tf5_l.paragraphs[0]
    p.text = "10-Step Leakage-Free Workflow"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_ACCENT
    p.space_after = Pt(10)

    steps = [
        "1. In-Memory SQLite: Cohort SQL querying by tenure, contract, and plans.",
        "2. Stratified 80/20 Train-Test Split: 5,634 train / 1,409 test records.",
        "3. Leakage Guard: Statistics (plan averages) fit strictly on train split.",
        "4. Feature Transformations: Log scales, interaction terms, and binning.",
        "5. ColumnTransformer: StandardScaler for numerics, OneHotEncoder for categoricals.",
        "6. Class Imbalance: scale_pos_weight = 2.76 in XGBoost, balanced class weights.",
        "7. Hyperparameter Tuning: Tree depth regularized to prevent overfitting.",
        "8. Soft Voting Ensemble: Blending XGBoost, LogReg, and Random Forest.",
        "9. Decision Threshold Studio: Profit-optimized cutoff tuning (0.35–0.40).",
        "10. SHAP Attribution & Automated 2x2 Retention Playbook Generation."
    ]
    for s in steps:
        p = tf5_l.add_paragraph()
        p.text = s
        p.font.size = Pt(11)
        p.space_after = Pt(3)

    add_card(slide5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), COLOR_BG_CARD, COLOR_PRIMARY)
    b5_r = slide5.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf5_r = b5_r.text_frame
    tf5_r.word_wrap = True
    p = tf5_r.paragraphs[0]
    p.text = "Engineered Domain Features"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    feats = [
        ("log_tenure", "Captures diminishing churn sensitivity as customer maturity grows."),
        ("charge_discrepancy", "TotalCharges - (tenure * MonthlyCharges) detecting sudden tariff spikes."),
        ("monthly_to_total_ratio", "Measures current price pressure relative to historical commitment."),
        ("num_services & num_security", "Counts active stickiness features (TechSupport, OnlineSecurity, Backup)."),
        ("fiber_no_tech", "Critical interaction: Fiber Optic customer without Tech Support."),
        ("short_tenure_m2m", "High-risk flag for accounts under 12 months on monthly terms."),
        ("echeck_m2m", "Captures highest-friction billing combination in the dataset."),
        ("charge_vs_internet_mean", "Price differential compared to customer's specific internet plan mean.")
    ]
    for name, desc in feats:
        p = tf5_r.add_paragraph()
        p.text = f"• {name}: {desc}"
        p.font.size = Pt(11.5)
        p.space_after = Pt(4)

    # ==========================================
    # SLIDE 6: Model Results & Benchmarking
    # ==========================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Model Benchmarking, Threshold Tuning & Decile Lift")

    # Table
    left, top, width, height = Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.8)
    table_shape = slide6.shapes.add_table(5, 7, left, top, width, height)
    t6 = table_shape.table
    for j in range(7):
        t6.columns[j].width = Inches(1.67)

    m_headers = ["Model Architecture", "Accuracy", "Precision", "Recall (0.50)", "Recall (0.35)", "ROC-AUC", "PR-AUC"]
    for j, h in enumerate(m_headers):
        cell = t6.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(12)
            p.font.color.rgb = COLOR_WHITE

    m_rows = [
        ("Tuned Ensemble (XGB+LR+RF)", "0.755", "0.526", "0.794", "0.888 (88.8%)", "0.849", "0.665"),
        ("XGBoost (Tuned Classifier)", "0.754", "0.525", "0.786", "0.877 (87.7%)", "0.849", "0.664"),
        ("Logistic Regression", "0.740", "0.506", "0.791", "0.864 (86.4%)", "0.849", "0.665"),
        ("Random Forest (Balanced)", "0.754", "0.525", "0.794", "0.858 (85.8%)", "0.845", "0.655")
    ]
    for i, row in enumerate(m_rows):
        for j, val in enumerate(row):
            cell = t6.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_WHITE if i % 2 == 0 else COLOR_BG_CARD
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.bold = (j in [0, 4, 6])
                p.font.color.rgb = COLOR_PRIMARY

    # Bottom Cards: Threshold Studio & Lift
    add_card(slide6, Inches(0.8), Inches(4.8), Inches(5.6), Inches(2.1), COLOR_BG_CARD, COLOR_SUCCESS)
    b6_l = slide6.shapes.add_textbox(Inches(0.9), Inches(4.9), Inches(5.4), Inches(1.9))
    tf6_l = b6_l.text_frame
    tf6_l.word_wrap = True
    p = tf6_l.paragraphs[0]
    p.text = "🎯 Threshold Studio Findings (Slide 3 & 6)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = COLOR_SUCCESS
    p.space_after = Pt(4)

    pts6_l = [
        "At default 0.50 cutoff: Recall is ~78.6% (misses ~80 churners).",
        "At profit-tuned 0.35 cutoff: Recall reaches 88.8% (332 of 374 churners caught!).",
        "Provides frontline teams the freedom to tune between maximum churn capture vs campaign spend control."
    ]
    for pt in pts6_l:
        p = tf6_l.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11)

    add_card(slide6, Inches(6.8), Inches(4.8), Inches(5.7), Inches(2.1), COLOR_BG_CARD, COLOR_ACCENT)
    b6_r = slide6.shapes.add_textbox(Inches(6.9), Inches(4.9), Inches(5.5), Inches(1.9))
    tf6_r = b6_r.text_frame
    tf6_r.word_wrap = True
    p = tf6_r.paragraphs[0]
    p.text = "📈 Decile Lift & Cumulative Gains (Slide 6 & 9)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = COLOR_ACCENT
    p.space_after = Pt(4)

    pts6_r = [
        "Top 10% Risk Decile: 78.0% churn rate, delivering a 2.94x Lift.",
        "Top 20% Risk Decile: Captures >52.1% of all churners (2.61x Lift over random).",
        "Validation: Contacting just the top 2 quintiles catches majority of churners at 20% of the contact budget."
    ]
    for pt in pts6_r:
        p = tf6_r.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11)

    # ==========================================
    # SLIDE 7: 2x2 Risk-Value Matrix & Playbook
    # ==========================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "From Prediction to Action: 2x2 Retention Playbook")

    # 4 Quadrants
    quads = [
        (Inches(0.8), Inches(1.8), Inches(5.6), Inches(2.4), COLOR_BG_CARD, COLOR_DANGER,
         "🔥 Priority Save (High Risk × High Value)",
         "Monthly Bill >= $70 | Churn Risk >= 60%",
         "Action: Personal VIP retention call from senior specialist + 20% discount on 12-month contract lock-in + free security bundle."),

        (Inches(6.8), Inches(1.8), Inches(5.7), Inches(2.4), COLOR_BG_CARD, COLOR_ACCENT,
         "💎 Nurture & Upsell (Low Risk × High Value)",
         "Monthly Bill >= $70 | Churn Risk < 35%",
         "Action: VIP loyalty perks, preview of faster fiber speeds, annual loyalty rewards, and quarterly account satisfaction review."),

        (Inches(0.8), Inches(4.4), Inches(5.6), Inches(2.4), COLOR_BG_CARD, RGBColor(245, 158, 11),
         "⚡ Low-Cost Save (High Risk × Low Value)",
         "Monthly Bill < $70 | Churn Risk >= 60%",
         "Action: Automated digital email/SMS campaign + $5 autopay statement credit + complimentary 3-month TechSupport trial."),

        (Inches(6.8), Inches(4.4), Inches(5.7), Inches(2.4), COLOR_BG_CARD, COLOR_MUTED,
         "🛡️ Monitor & Retain (Low Risk × Low Value)",
         "Monthly Bill < $70 | Churn Risk < 35%",
         "Action: Standard self-service portal, automated quarterly NPS survey, zero outbound promotional expenditure.")
    ]

    for l, t, w, h, bg, border, title, sub, act in quads:
        add_card(slide7, l, t, w, h, bg, border)
        tb = slide7.shapes.add_textbox(l + Inches(0.15), t + Inches(0.15), w - Inches(0.3), h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = border
        p.space_after = Pt(2)

        p = tf.add_paragraph()
        p.text = sub
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_MUTED
        p.space_after = Pt(6)

        p = tf.add_paragraph()
        p.text = act
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 8: System Architecture & Web Application
    # ==========================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "End-to-End System Architecture & Streamlit Application")

    # Workflow strip
    add_card(slide8, Inches(0.8), Inches(1.8), Inches(11.7), Inches(1.0), COLOR_BG_CARD, COLOR_ACCENT)
    tb_flow = slide8.shapes.add_textbox(Inches(0.9), Inches(1.9), Inches(11.5), Inches(0.8))
    tf_flow = tb_flow.text_frame
    tf_flow.word_wrap = True
    p = tf_flow.paragraphs[0]
    p.text = "DATA PIPELINE:  Raw Telco CSV  ➔  SQLite DB  ➔  Feature Engineering  ➔  Ensemble ML  ➔  SHAP Explainer  ➔  Streamlit UI"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_ACCENT

    # 4 Module Cards
    mods = [
        (Inches(0.8), Inches(3.0), Inches(5.6), Inches(1.9),
         "📊 Executive EDA & SQLite Cohorts",
         "KPIs for MRR & revenue at risk ($1.67M), interactive donut & tenure distributions, plus in-memory SQL cohort query editor."),
        (Inches(6.8), Inches(3.0), Inches(5.7), Inches(1.9),
         "🤖 Threshold Studio & Model Diagnostics",
         "Interactive threshold tuning with live Confusion Matrix, PR Tradeoff curve, and automated leaderboard across 4 models."),
        (Inches(0.8), Inches(5.1), Inches(5.6), Inches(1.9),
         "⚡ Real-Time Predictor & SHAP Force Plot",
         "Interactive JavaScript SHAP Force Plot + Counterfactual What-If tool to simulate contract upgrades and see risk drop."),
        (Inches(6.8), Inches(5.1), Inches(5.7), Inches(1.9),
         "📥 Batch Campaign Export & ROI Planner",
         "One-click 'Download Campaign List (CSV)' filtered by risk tier + financial simulator comparing targeting vs contacting everyone.")
    ]
    for l, t, w, h, title, desc in mods:
        add_card(slide8, l, t, w, h, COLOR_BG_CARD, COLOR_PRIMARY)
        tb = slide8.shapes.add_textbox(l + Inches(0.15), t + Inches(0.15), w - Inches(0.3), h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(4)

        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_MUTED

    # ==========================================
    # SLIDE 9: Business Impact & Campaign ROI
    # ==========================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Business Impact: Model Targeting vs Mass Contacting (Slide 9)")

    # Table
    left, top, width, height = Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.5)
    table_shape = slide9.shapes.add_table(3, 7, left, top, width, height)
    t9 = table_shape.table
    for j in range(7):
        t9.columns[j].width = Inches(1.67)

    roi_headers = ["Strategy", "Contacted", "Campaign Cost", "Saved Churners", "Revenue Kept", "Net Profit / Gain", "Campaign ROI %"]
    for j, h in enumerate(roi_headers):
        cell = t9.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(12)
            p.font.color.rgb = COLOR_WHITE

    roi_rows = [
        ("Strategy A: Contact Everyone (Mass Blast)", "7,043 (100%)", "$281,720", "467 accounts", "$364,260", "+$82,540", "29.3% ROI"),
        ("Strategy B: Target Top 20% (Model-Driven)", "1,409 (20%)", "$56,360", "280 accounts", "$218,556", "+$162,196", "287.8% ROI (10x!)")
    ]
    for i, row in enumerate(roi_rows):
        for j, val in enumerate(row):
            cell = t9.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_WHITE if i == 0 else RGBColor(236, 253, 245)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.bold = (j in [0, 5, 6])
                p.font.color.rgb = COLOR_SUCCESS if (i == 1 and j in [5, 6]) else COLOR_PRIMARY

    # Bottom Callout Card
    add_card(slide9, Inches(0.8), Inches(4.5), Inches(11.7), Inches(2.4), COLOR_BG_CARD, COLOR_SUCCESS)
    b9 = slide9.shapes.add_textbox(Inches(1.0), Inches(4.6), Inches(11.3), Inches(2.2))
    tf9 = b9.text_frame
    tf9.word_wrap = True
    p = tf9.paragraphs[0]
    p.text = "💡 Financial Assumptions & Bottom-Line Executive Takeaway"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_SUCCESS
    p.space_after = Pt(6)

    pts9 = [
        "Assumptions: Baseline 26.5% churn, $40 contact cost, 25% offer save rate, $780 annual CLV (12 × $65/mo).",
        "Wasted Spend Eliminated: Target Top 20% saves $225,360 in wasted discount spend (spending 80% less).",
        "Higher Net Profit: Delivers +$162,196 in net profit—nearly DOUBLE the net gain of contacting everyone.",
        "10x Campaign ROI Multiplier: ROI jumps from 29.3% (mass spray-and-pray) to 287.8% (precision ML targeting).",
        "Operational Feasibility: Batch scoring 100K+ subscribers takes minutes. Model retrained quarterly to monitor drift."
    ]
    for pt in pts9:
        p = tf9.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11.5)
        p.space_after = Pt(3)

    # ==========================================
    # SLIDE 10: Conclusion & Future Roadmap
    # ==========================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Conclusion and Future Work")

    add_card(slide10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), COLOR_BG_CARD, COLOR_PRIMARY)
    b10_l = slide10.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tf10_l = b10_l.text_frame
    tf10_l.word_wrap = True
    p = tf10_l.paragraphs[0]
    p.text = "What We Built (Delivered Scope)"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(12)

    bullets10_l = [
        ("Production ML Pipeline", "Leakage-free transformers, domain feature engineering, and tuned soft voting ensemble."),
        ("Threshold Studio", "Flexible decision cutoff reaching 88.8% Recall and 0.849 ROC-AUC."),
        ("Explainable AI (SHAP)", "Interactive JavaScript force plots explaining per-customer risk drivers."),
        ("Retention Playbook Matrix", "2x2 Risk-Value matrix with automated personalized interventions."),
        ("Working Streamlit Application", "Complete 7-module dashboard with What-If simulator and CSV campaign export.")
    ]
    for title, desc in bullets10_l:
        p = tf10_l.add_paragraph()
        p.text = f"✓ {title}: {desc}"
        p.font.size = Pt(12.5)
        p.space_after = Pt(8)

    add_card(slide10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), COLOR_BG_CARD, COLOR_ACCENT)
    b10_r = slide10.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf10_r = b10_r.text_frame
    tf10_r.word_wrap = True
    p = tf10_r.paragraphs[0]
    p.text = "Future Extensions & Roadmap"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_ACCENT
    p.space_after = Pt(12)

    bullets10_r = [
        ("Uplift Modeling", "Transition from churn risk to treatment uplift—targeting only 'Persuadable' subscribers."),
        ("Survival Analysis", "Predict when a customer will churn (hazard curves) rather than just a binary whether."),
        ("Continuous Usage Logs", "Incorporate daily clickstream, latency drops, customer support tickets, and NPS feedback."),
        ("Real-Time CRM API", "Deploy FastAPI endpoints to trigger automated retention offers inside Salesforce / HubSpot."),
        ("A/B Feedback Loop", "Feed retention campaign offer redemption data back into retraining models quarterly.")
    ]
    for title, desc in bullets10_r:
        p = tf10_r.add_paragraph()
        p.text = f"➔ {title}: {desc}"
        p.font.size = Pt(12.5)
        p.space_after = Pt(8)

    # ==========================================
    # SLIDE 11: Thank You & Q&A
    # ==========================================
    slide11 = prs.slides.add_slide(blank_layout)
    bg11 = slide11.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg11.fill.solid()
    bg11.fill.fore_color.rgb = COLOR_PRIMARY
    bg11.line.fill.background()

    t_box11 = slide11.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.3), Inches(3.8))
    tf11 = t_box11.text_frame
    tf11.word_wrap = True

    p = tf11.paragraphs[0]
    p.text = "Thank You!"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.space_after = Pt(12)

    p = tf11.add_paragraph()
    p.text = "Questions & Discussions are Welcome"
    p.font.size = Pt(22)
    p.font.color.rgb = COLOR_SUCCESS
    p.space_after = Pt(28)

    p = tf11.add_paragraph()
    p.text = "Interactive Working Demo: http://localhost:8501"
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor(148, 163, 184)
    p.space_after = Pt(6)

    p = tf11.add_paragraph()
    p.text = "[Your Name]  ·  [Your Email]  ·  [LinkedIn Profile]  ·  [GitHub Repository]"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(203, 213, 225)

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataNova_2026_Customer_Churn_Retention_Strategy.pptx")
    prs.save(out_path)
    print("Presentation saved successfully at:", out_path)
    return out_path

if __name__ == "__main__":
    create_deck()
