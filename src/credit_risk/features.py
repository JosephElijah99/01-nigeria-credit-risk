"""Feature engineering + sklearn preprocessing.

Imputation lives INSIDE the sklearn pipeline (fit on train only) to avoid leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "monthly_income_ngn", "loan_amount_ngn", "tenor_months",
           "monthly_interest_rate", "prior_loans", "prior_defaults", "mobile_money_txn_30d",
           "avg_wallet_balance_ngn", "salary_account", "existing_debt_ngn", "bureau_score",
           "inflation_rate", "gdp_growth", "unemployment_rate",
           "loan_to_income", "debt_to_income", "monthly_repayment_ratio",
           "default_history_rate", "bureau_missing", "wallet_to_income"]
CATEGORICAL = ["gender", "education", "employment_type", "loan_purpose", "channel",
               "device_type", "region"]
FEATURES = NUMERIC + CATEGORICAL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level, application-time-only features (no leakage from outcomes)."""
    out = df.copy()
    inc = out["monthly_income_ngn"].replace(0, np.nan)
    out["loan_to_income"] = out["loan_amount_ngn"] / inc
    out["debt_to_income"] = out["existing_debt_ngn"] / inc
    total = out["loan_amount_ngn"] * (1 + out["monthly_interest_rate"] * out["tenor_months"])
    out["monthly_repayment_ratio"] = (total / out["tenor_months"]) / inc
    out["default_history_rate"] = (out["prior_defaults"] / out["prior_loans"].replace(0, np.nan)).fillna(0)
    out["bureau_missing"] = out["bureau_score"].isna().astype(int)
    out["wallet_to_income"] = out["avg_wallet_balance_ngn"] / inc
    return out


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=False)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
