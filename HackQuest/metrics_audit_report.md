# Metrics Audit & Contradiction Resolution Report

### RetainPulse AI — DataNova 2026

---

## 1. Executive Summary: The Contradiction Resolved

In the original project materials, there was a discrepancy regarding the Top-20% customer risk segment capture rate:
- **Slide 6 (and README line 134)** stated: `>52.1% of all churners` (or `>52%`).
- **Slide 9 (and app.py line 961)** assumed: `59.9%` (using `280 accounts saved` out of `467.25` total churners, and `top_pct_churn_capture_rate=0.60` in code).

### The True Single Source of Truth (`artifacts/metrics.json`):
- Model: Tuned Soft Voting Ensemble (XGBoost + Logistic Regression + Random Forest), evaluated on the held-out test set ($N=1,409$, $374$ churners, seeded random_state=42).
- **Actual Top 20% Churn Capture Rate: 51.07%** (Lift: **2.55x**; 95% Bootstrap CI: **[47.0% – 54.9%]**).
- **Actual Top 10% Churn Capture Rate: 28.61%** (Lift: **2.86x**).

---

## 2. Line-by-Line Inventory of Affected Files and Changes

| File | Location | Old / Contradictory Value | New Single-Source-of-Truth Value | Reason for Change |
| :--- | :--- | :--- | :--- | :--- |
| **create_presentation.py** | Slide 2 (Goal strip) | `over 52% of churners reside` | `51.1% of all churners (2.55x Lift)` | Synchronized with held-out test cumulative gains |
| **create_presentation.py** | Slide 2 (ROI strip) | `+$162K net gain vs mass contact` | `+$47K additional profit (+57% higher net profit) & $225K cost savings` | Mathematically derived from 51.07% capture rate |
| **create_presentation.py** | Slide 6 (Decile card) | `Top 10%: 78.0%, 2.94x Lift` | `Top 10%: 75.9% churn rate, 2.86x Lift (28.6% capture)` | Empirically computed decile 1 metrics |
| **create_presentation.py** | Slide 6 (Decile card) | `Top 20%: Captures >52.1% (2.61x Lift)` | `Top 20%: Captures 51.1% of all churners (2.55x Lift)` | Exact test-set cumulative gains value |
| **create_presentation.py** | Slide 9 (Table Row 2) | `Saved: 280 accounts` (implied 59.9%) | `Saved: 239 accounts` (51.07% capture rate × 25% save rate) | Consistent downstream mathematical calculation |
| **create_presentation.py** | Slide 9 (Table Row 2) | `Revenue Kept: $218,556` | `Revenue Kept: $186,125` | 238.6 saved accounts × $780 CLV |
| **create_presentation.py** | Slide 9 (Table Row 2) | `Net Profit: +$162,196` | `Net Profit: +$129,781` | $186,125 revenue kept - $56,344 campaign cost |
| **create_presentation.py** | Slide 9 (Table Row 2) | `ROI: 287.8% (10x!)` | `ROI: 230.3% (7.8x ROI multiplier!)` | $129,781 / $56,344 |
| **create_presentation.py** | Slide 9 (Bullets) | `Delivers +$162,196 in net profit` | `Delivers +$129,781 in net profit (+57% higher profit than mass blast)` | Harmonized with single source of truth |
| **create_presentation.py** | Slide 9 (Bullets) | `ROI jumps from 29.3% to 287.8%` | `ROI jumps from 29.4% to 230.3% (7.8x ROI Multiplier)` | Harmonized with single source of truth |
| **app.py** | Line 961 | `top_pct_churn_capture_rate=0.60` | Dynamically read from `metrics.json` (`51.07%`) | Removed hardcoding |
| **README.md** | Line 134 | `capturing >52% of churners` | `capturing 51.1% of churners [47.0–54.9%]` | Harmonized with test metrics & bootstrap CI |

---

## 3. Financial Impact Comparison Table (Updated for Deck Slide 9)

| Strategy | Target Audience | Campaign Cost ($40/contact) | Saved Churners (25% save rate) | Revenue Kept ($780 CLV) | Net Financial Gain | Campaign ROI % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy A: Contact Everyone (Mass Blast)** | 7,043 (100%) | $281,720 | 467 accounts | $364,455 | +$82,735 | **29.4%** |
| **Strategy B: Target Top 20% (Model-Driven)** | 1,408 (20%) | $56,344 | 239 accounts | $186,125 | **+$129,781** | **230.3% (7.8x Multiplier)** |

### Key Executive Takeaways:
1. **$225,376 Budget Saved**: Contacting only the high-risk quintile cuts contact/discount budget by 80%.
2. **+$47,046 Extra Net Profit (+56.9%)**: Net profit increases from $82,735 to $129,781 despite contacting 80% fewer accounts.
3. **7.8x ROI Multiplier**: Precision ML targeting delivers 230.3% ROI vs 29.4% for mass campaign outreach.
