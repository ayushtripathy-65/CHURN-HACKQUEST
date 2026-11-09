"""
retention_engine.py
Implements the 2x2 Risk-Value Retention Matrix, granular rule-based action
recommendations, and the DataNova 2026 Business Impact / Campaign ROI Simulator.
"""

import pandas as pd
import numpy as np

def assign_risk_tier(prob: float, high_thresh: float = 0.60, med_thresh: float = 0.35) -> str:
    """Classifies churn probability into Low, Medium, or High Risk tiers."""
    if prob >= high_thresh:
        return "High Risk"
    elif prob >= med_thresh:
        return "Medium Risk"
    else:
        return "Low Risk"

def assign_value_tier(monthly_charges: float, threshold: float = 70.0) -> str:
    """Classifies customer value based on monthly spend."""
    return "High Value" if monthly_charges >= threshold else "Low Value"

def get_matrix_strategy(risk_tier: str, value_tier: str) -> dict:
    """
    Returns strategic playbook category based on the 2x2 Risk x Value Matrix (Slide 7).
    """
    if risk_tier == "High Risk" and value_tier == "High Value":
        return {
            "strategy": "Priority Save",
            "channel": "Dedicated Personal Call + Executive Offer",
            "budget_tier": "High Spend ($50 - $100 per customer)",
            "summary": "Urgent intervention required. High CLTV customer at immediate risk of leaving. Deploy personalized retention specialist with maximum contract discounting authority."
        }
    elif risk_tier == "High Risk" and value_tier == "Low Value":
        return {
            "strategy": "Low-Cost Save",
            "channel": "Automated Personalized Email / SMS Campaign",
            "budget_tier": "Low Spend ($5 - $15 per customer)",
            "summary": "High risk of leaving but lower margin. Use cost-efficient digital nudges, small bundle discounts, or free service trials without excessive promotional spend."
        }
    elif risk_tier in ["Medium Risk", "Low Risk"] and value_tier == "High Value":
        return {
            "strategy": "Nurture & Upsell",
            "channel": "VIP Loyalty Program & Relationship Management",
            "budget_tier": "Medium Spend ($20 - $35 per customer)",
            "summary": "Valuable customer currently stable. Protect margin with loyalty perks, exclusive early access to faster fiber tiers, and quarterly relationship reviews."
        }
    else:
        return {
            "strategy": "Monitor & Retain",
            "channel": "Standard Self-Service & Periodic Newsletters",
            "budget_tier": "Zero Spend ($0)",
            "summary": "Low risk, modest spend. Maintain high baseline service satisfaction without allocating proactive outbound retention budget."
        }

def generate_personalized_actions(customer_row: pd.Series, churn_prob: float, shap_reasons: list = None) -> list:
    """
    Generates 2-4 concrete, actionable retention offers based on customer features (Slide 7 playbook).
    """
    actions = []
    contract = str(customer_row.get("Contract", ""))
    tenure = float(customer_row.get("tenure", 0))
    tech_support = str(customer_row.get("TechSupport", ""))
    security = str(customer_row.get("OnlineSecurity", ""))
    payment = str(customer_row.get("PaymentMethod", ""))
    internet = str(customer_row.get("InternetService", ""))
    monthly = float(customer_row.get("MonthlyCharges", 0))

    # Action 1: Contract Upgrade Incentive
    if "month" in contract.lower():
        actions.append({
            "trigger": "Month-to-Month Contract",
            "action": "Offer a 15% discount for committing to a 12-month or 24-month contract lock-in.",
            "expected_impact": "Significantly locks in customer tenure and slashes baseline churn by up to 50%."
        })

    # Action 2: Onboarding & Early Life Care
    if tenure <= 12:
        actions.append({
            "trigger": f"New Customer Tenure ({int(tenure)} months)",
            "action": "Trigger automated 30/60/90-day onboarding satisfaction check-in and dedicated concierge support.",
            "expected_impact": "Resolves early friction points that cause 40%+ of first-year cancellations."
        })

    # Action 3: Value-Add Services (Tech Support & Security)
    if "no" in tech_support.lower() or "no" in security.lower():
        actions.append({
            "trigger": "Missing Tech Support / Cyber Security",
            "action": "Provide a complimentary 3-month trial of 24/7 Premium Tech Support & Online Security Shield.",
            "expected_impact": "Increases product stickiness and service adoption, reducing churn propensity."
        })

    # Action 4: Payment Friction / Electronic Check
    if "electronic check" in payment.lower():
        actions.append({
            "trigger": "Payment via Electronic Check (High Friction)",
            "action": "Offer a $5 monthly bill credit for switching to automated Credit Card or Bank Transfer autopay.",
            "expected_impact": "Reduces billing churn and involuntary drop-offs from payment failure."
        })

    # Action 5: High Bill / Fiber Plan Optimization
    if monthly > 80.0 or "fiber" in internet.lower():
        actions.append({
            "trigger": f"High Monthly Spend (${monthly:.2f})",
            "action": "Conduct a plan right-sizing review; offer customized entertainment bundle (e.g. free streaming add-on).",
            "expected_impact": "Restores perceived value without slashing top-line headline tariffs."
        })

    # Fallback if customer is low risk
    if len(actions) == 0 or churn_prob < 0.25:
        actions.append({
            "trigger": "Loyal, Stable Customer",
            "action": "Enroll in Annual Customer Loyalty Club with reward points and complimentary speed boosts.",
            "expected_impact": "Reinforces long-term brand advocacy and defensive loyalty against competitors."
        })

    return actions[:4]

def calculate_campaign_roi(
    total_customers: int,
    baseline_churn_rate: float,
    avg_annual_revenue: float,
    contact_cost: float,
    save_rate: float,
    top_pct_targeted: float = 0.20,
    top_pct_churn_capture_rate: float = 0.60
) -> dict:
    """
    DataNova Business Impact Calculator (Slide 9):
    Compares 'Contact Everyone' vs 'Model-Driven Targeting (Top X%)' vs 'No Campaign'.
    """
    total_churners = total_customers * baseline_churn_rate

    # Strategy 1: Contact Everyone (Mass Spray-and-Pray)
    cost_everyone = total_customers * contact_cost
    saved_everyone = total_churners * save_rate
    revenue_kept_everyone = saved_everyone * avg_annual_revenue
    net_gain_everyone = revenue_kept_everyone - cost_everyone
    roi_everyone = (net_gain_everyone / cost_everyone * 100) if cost_everyone > 0 else 0

    # Strategy 2: Model-Driven Targeting (Target Top X% High-Risk)
    contacted_model = total_customers * top_pct_targeted
    cost_model = contacted_model * contact_cost
    churners_targeted = total_churners * top_pct_churn_capture_rate
    saved_model = churners_targeted * save_rate
    revenue_kept_model = saved_model * avg_annual_revenue
    net_gain_model = revenue_kept_model - cost_model
    roi_model = (net_gain_model / cost_model * 100) if cost_model > 0 else 0

    # Lift in efficiency
    cost_savings = cost_everyone - cost_model
    additional_profit = net_gain_model - net_gain_everyone

    return {
        "everyone": {
            "contacted": int(total_customers),
            "campaign_cost": cost_everyone,
            "churners_saved": int(saved_everyone),
            "revenue_kept": revenue_kept_everyone,
            "net_gain": net_gain_everyone,
            "roi_pct": roi_everyone
        },
        "model": {
            "target_pct": top_pct_targeted * 100,
            "contacted": int(contacted_model),
            "campaign_cost": cost_model,
            "churners_saved": int(saved_model),
            "revenue_kept": revenue_kept_model,
            "net_gain": net_gain_model,
            "roi_pct": roi_model
        },
        "comparison": {
            "cost_savings": cost_savings,
            "additional_profit": additional_profit,
            "roi_multiplier": (roi_model / roi_everyone) if roi_everyone > 0 else 1.0
        }
    }
