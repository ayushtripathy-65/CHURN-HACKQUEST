"""
preprocessor.py
Advanced domain feature engineering, preprocessing pipelines,
and stratified train-test splitting without data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Service column definitions
SEC_COLS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
STREAM_COLS = ['StreamingTV', 'StreamingMovies']
ALL_SERVICE_COLS = ['PhoneService', 'MultipleLines'] + SEC_COLS + STREAM_COLS

def engineer_features(df: pd.DataFrame, internet_plan_means: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Performs high-leverage domain feature engineering for customer churn:
    - Log transforms & tenure non-linearities
    - Billing discrepancies & monthly-to-total ratios
    - Multi-service counts & security vs streaming bundles
    - Key Telco churn risk combos (e.g. Fiber without TechSupport, Month-to-month with Electronic Check)
    - Relative pricing benchmarks
    """
    data = df.copy()

    # 1. Non-linear tenure representation
    if 'tenure' in data.columns:
        data['log_tenure'] = np.log1p(data['tenure'])
        bins = [-1, 12, 24, 48, 60, 100]
        labels = ['0-12m', '13-24m', '25-48m', '49-60m', '>60m']
        data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels).astype(str)
    else:
        data['log_tenure'] = 0.0
        data['tenure_group'] = '0-12m'

    # 2. Billing shock & charge discrepancies
    if 'TotalCharges' in data.columns and 'tenure' in data.columns and 'MonthlyCharges' in data.columns:
        expected_total = data['tenure'] * data['MonthlyCharges']
        data['charge_discrepancy'] = data['TotalCharges'] - expected_total
        data['monthly_to_total_ratio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1.0)
        data['avg_charge_per_month'] = data['TotalCharges'] / (data['tenure'] + 1.0)
    elif 'MonthlyCharges' in data.columns:
        data['charge_discrepancy'] = 0.0
        data['monthly_to_total_ratio'] = 1.0
        data['avg_charge_per_month'] = data['MonthlyCharges']
    else:
        data['charge_discrepancy'] = 0.0
        data['monthly_to_total_ratio'] = 0.0
        data['avg_charge_per_month'] = 0.0

    # 3. Service bundle metrics
    data['num_services'] = sum((data[col] == 'Yes').astype(int) for col in ALL_SERVICE_COLS if col in data.columns)
    data['num_security_services'] = sum((data[col] == 'Yes').astype(int) for col in SEC_COLS if col in data.columns)
    data['num_streaming_services'] = sum((data[col] == 'Yes').astype(int) for col in STREAM_COLS if col in data.columns)
    
    if 'MonthlyCharges' in data.columns:
        data['charges_per_service'] = data['MonthlyCharges'] / (data['num_services'] + 1.0)
    else:
        data['charges_per_service'] = 0.0

    # 4. Critical Telco Churn Risk Combinations
    if 'InternetService' in data.columns and 'TechSupport' in data.columns:
        data['fiber_no_tech'] = ((data['InternetService'] == 'Fiber optic') & (data['TechSupport'] == 'No')).astype(int)
    else:
        data['fiber_no_tech'] = 0

    if 'InternetService' in data.columns and 'OnlineSecurity' in data.columns:
        data['fiber_no_sec'] = ((data['InternetService'] == 'Fiber optic') & (data['OnlineSecurity'] == 'No')).astype(int)
    else:
        data['fiber_no_sec'] = 0

    if 'tenure' in data.columns and 'Contract' in data.columns:
        data['short_tenure_m2m'] = ((data['tenure'] <= 12) & (data['Contract'] == 'Month-to-month')).astype(int)
    else:
        data['short_tenure_m2m'] = 0

    if 'PaymentMethod' in data.columns and 'Contract' in data.columns:
        data['echeck_m2m'] = ((data['PaymentMethod'] == 'Electronic check') & (data['Contract'] == 'Month-to-month')).astype(int)
        data['has_autopay'] = data['PaymentMethod'].astype(str).str.contains('automatic|bank transfer|credit card', case=False).astype(int)
    else:
        data['echeck_m2m'] = 0
        data['has_autopay'] = 0

    if 'Partner' in data.columns and 'Dependents' in data.columns:
        data['is_alone'] = ((data['Partner'] == 'No') & (data['Dependents'] == 'No')).astype(int)
    else:
        data['is_alone'] = 0

    # 5. Contract numerical weight & interaction
    if 'Contract' in data.columns:
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        data['contract_numeric'] = data['Contract'].map(contract_map).fillna(0)
        data['tenure_contract_interaction'] = data.get('tenure', 0) * (data['contract_numeric'] + 1)
    else:
        data['contract_numeric'] = 0
        data['tenure_contract_interaction'] = 0

    # 6. Relative pricing benchmarks (leakage-free)
    if 'MonthlyCharges' in data.columns and 'InternetService' in data.columns:
        if internet_plan_means is None:
            internet_plan_means = data.groupby('InternetService')['MonthlyCharges'].mean().to_dict()
        data['charge_vs_internet_mean'] = data.apply(
            lambda r: r['MonthlyCharges'] - internet_plan_means.get(r['InternetService'], data['MonthlyCharges'].mean()),
            axis=1
        )
        data['charge_ratio_internet'] = data.apply(
            lambda r: r['MonthlyCharges'] / (internet_plan_means.get(r['InternetService'], data['MonthlyCharges'].mean()) + 1e-5),
            axis=1
        )
    else:
        internet_plan_means = {}
        data['charge_vs_internet_mean'] = 0.0
        data['charge_ratio_internet'] = 1.0

    return data, internet_plan_means

class ChurnPreprocessor:
    """Preprocesses raw data using standard ColumnTransformer with robust feature set."""

    def __init__(self):
        self.column_transformer = None
        self.feature_names = []
        self.internet_plan_means = {}
        self.numerical_cols = []
        self.categorical_cols = []

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        X_eng, self.internet_plan_means = engineer_features(X_train)

        # Numerical columns
        self.numerical_cols = [
            'tenure', 'log_tenure', 'MonthlyCharges', 'TotalCharges',
            'charge_discrepancy', 'monthly_to_total_ratio', 'avg_charge_per_month',
            'num_services', 'num_security_services', 'num_streaming_services',
            'charges_per_service', 'fiber_no_tech', 'fiber_no_sec',
            'short_tenure_m2m', 'echeck_m2m', 'has_autopay', 'is_alone',
            'contract_numeric', 'tenure_contract_interaction',
            'charge_vs_internet_mean', 'charge_ratio_internet'
        ]
        self.numerical_cols = [c for c in self.numerical_cols if c in X_eng.columns]

        # Categorical columns
        self.categorical_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'tenure_group'
        ]
        self.categorical_cols = [c for c in self.categorical_cols if c in X_eng.columns]

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_cols)
            ]
        )

        X_processed = self.column_transformer.fit_transform(X_eng)

        cat_encoder = self.column_transformer.named_transformers_['cat']
        cat_names = list(cat_encoder.get_feature_names_out(self.categorical_cols))
        self.feature_names = self.numerical_cols + cat_names

        return X_processed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_eng, _ = engineer_features(X, self.internet_plan_means)
        return self.column_transformer.transform(X_eng)

def prepare_train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Stratified 80/20 train-test split avoiding data leakage.
    """
    clean_df = df.copy()
    customer_ids = clean_df['customerID'] if 'customerID' in clean_df.columns else clean_df.index
    
    if 'Churn' not in clean_df.columns:
        raise ValueError("Dataset must contain 'Churn' target column.")

    y = (clean_df['Churn'] == 'Yes').astype(int)
    X = clean_df.drop(columns=['Churn', 'customerID'], errors='ignore')

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = ChurnPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "id_train": id_train.values,
        "id_test": id_test.values,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.feature_names
    }
