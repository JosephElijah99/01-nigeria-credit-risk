"""Validation and cleaning. Returns a clean frame plus a data-quality report."""
from __future__ import annotations
import pandas as pd

from .generate import STATES

CANONICAL = {s.lower(): s for s in STATES}
REQUIRED = ["loan_id", "application_date", "age", "state", "monthly_income_ngn",
            "loan_amount_ngn", "tenor_months", "default"]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    report = {"rows_in": len(df)}
    df = df.copy()
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")

    n0 = len(df)
    df = df.drop_duplicates(subset="loan_id", keep="first")
    report["duplicates_removed"] = n0 - len(df)

    df["state"] = df["state"].astype(str).str.strip().str.lower().map(CANONICAL)
    report["unmapped_states"] = int(df["state"].isna().sum())

    bad_age = ~df["age"].between(18, 80)
    report["invalid_age_set_to_nan"] = int(bad_age.sum())
    df.loc[bad_age, "age"] = pd.NA
    df["age"] = pd.to_numeric(df["age"])

    report["missing_income"] = int(df["monthly_income_ngn"].isna().sum())
    report["missing_bureau_score"] = int(df["bureau_score"].isna().sum())
    df = df.dropna(subset=["application_date", "state", "loan_amount_ngn"])
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report
