"""
data_loader.py
Handles data loading, cleaning, validation, and in-memory SQLite cohort querying.
"""

import os
import sqlite3
import pandas as pd
import numpy as np

def get_default_data_path():
    """Returns the default path to the IBM Telco Customer Churn dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "telco_churn.csv")

def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Telco dataset according to project specifications:
    - Strips whitespace
    - Converts TotalCharges from text to numeric; 11 blanks are new customers (tenure 0) -> filled with 0.0
    - Converts SeniorCitizen to category or readable string if numeric
    - Cleans Churn column to standard 'Yes'/'No'
    """
    df_clean = df.copy()

    # Strip column names
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # Handle TotalCharges blanks (Telco limitation noted in slide 4)
    if 'TotalCharges' in df_clean.columns:
        # Replace empty spaces with NaN and convert to float
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].astype(str).str.strip(), errors='coerce')
        # Blanks correspond to brand new customers (tenure == 0); fill with 0.0
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)

    # Standardize SeniorCitizen as Yes/No for uniform EDA
    if 'SeniorCitizen' in df_clean.columns:
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({1: 'Yes', 0: 'No', '1': 'Yes', '0': 'No'}).fillna(df_clean['SeniorCitizen'])

    # Standardize tenure as int
    if 'tenure' in df_clean.columns:
        df_clean['tenure'] = pd.to_numeric(df_clean['tenure'], errors='coerce').fillna(0).astype(int)

    # Standardize MonthlyCharges
    if 'MonthlyCharges' in df_clean.columns:
        df_clean['MonthlyCharges'] = pd.to_numeric(df_clean['MonthlyCharges'], errors='coerce').fillna(0.0)

    return df_clean

def load_data(file_source=None) -> pd.DataFrame:
    """
    Loads dataset from uploaded file buffer or default data path.
    """
    if file_source is not None:
        if isinstance(file_source, str):
            df = pd.read_csv(file_source)
        else:
            df = pd.read_csv(file_source)
    else:
        path = get_default_data_path()
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"Default dataset not found at {path}")

    return clean_telco_data(df)

def get_data_summary(df: pd.DataFrame) -> dict:
    """Calculates key overview statistics."""
    total_customers = len(df)
    churn_col = 'Churn' if 'Churn' in df.columns else None

    if churn_col:
        churn_count = (df[churn_col] == 'Yes').sum()
        churn_rate = (churn_count / total_customers) * 100 if total_customers > 0 else 0
        stay_count = total_customers - churn_count
    else:
        churn_count, churn_rate, stay_count = 0, 0, total_customers

    monthly_revenue = df['MonthlyCharges'].sum() if 'MonthlyCharges' in df.columns else 0
    revenue_at_risk = df[df[churn_col] == 'Yes']['MonthlyCharges'].sum() * 12 if churn_col and 'MonthlyCharges' in df.columns else 0

    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "stay_count": stay_count,
        "churn_rate": churn_rate,
        "monthly_revenue": monthly_revenue,
        "revenue_at_risk_annual": revenue_at_risk,
        "features_count": len(df.columns)
    }

class SQLiteCohortExplorer:
    """In-memory SQLite database helper for cohort queries (Step 1 in presentation)."""

    def __init__(self, df: pd.DataFrame):
        self.conn = sqlite3.connect(":memory:")
        # Write df to SQLite
        df.to_sql("customers", self.conn, index=False, if_exists="replace")

    def query(self, sql_query: str) -> pd.DataFrame:
        """Runs a user or predefined SQL query on the dataset."""
        try:
            return pd.read_sql_query(sql_query, self.conn)
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    def get_preset_queries(self) -> dict:
        """Returns standard cohort analysis queries matching Slide 5."""
        return {
            "Churn Rate by Contract Type": """
                SELECT 
                    Contract,
                    COUNT(*) AS total_customers,
                    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
                    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
                    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue,
                    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges * 12 ELSE 0 END), 2) AS annual_revenue_at_risk
                FROM customers
                GROUP BY Contract
                ORDER BY churn_rate_pct DESC;
            """,
            "Churn by Internet Service & Tech Support": """
                SELECT 
                    InternetService,
                    TechSupport,
                    COUNT(*) AS customer_count,
                    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
                    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
                    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill
                FROM customers
                GROUP BY InternetService, TechSupport
                ORDER BY churn_rate_pct DESC;
            """,
            "Tenure Cohorts Churn & Spend": """
                SELECT 
                    CASE 
                        WHEN tenure <= 12 THEN '01. 0 - 12 Months'
                        WHEN tenure <= 24 THEN '02. 13 - 24 Months'
                        WHEN tenure <= 48 THEN '03. 25 - 48 Months'
                        WHEN tenure <= 60 THEN '04. 49 - 60 Months'
                        ELSE '05. > 60 Months'
                    END AS tenure_cohort,
                    COUNT(*) AS total_customers,
                    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
                    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
                    ROUND(AVG(TotalCharges), 2) AS avg_cltv
                FROM customers
                GROUP BY tenure_cohort
                ORDER BY tenure_cohort;
            """,
            "Payment Method Risk Analysis": """
                SELECT 
                    PaymentMethod,
                    COUNT(*) AS total_customers,
                    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churners,
                    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
                    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill
                FROM customers
                GROUP BY PaymentMethod
                ORDER BY churn_rate_pct DESC;
            """
        }
