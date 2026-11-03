"""Synthetic loan-book generator for a Nigerian digital lender.

Real lender data is confidential, so this simulates a realistic portfolio with:
  * Nigerian geography (states -> geopolitical zones) and NGN amounts
  * macro conditions (inflation etc.) that genuinely affect default risk
  * deliberately messy fields (typos in state names, missing income, duplicates)
    so the cleaning step is a real step, not a formality.
The data-generating process is documented so results can be audited.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STATES = {  # state: (zone, sampling weight, income multiplier)
    "Lagos": ("South West", 0.26, 1.30), "Ogun": ("South West", 0.06, 1.00),
    "Oyo": ("South West", 0.07, 0.95), "FCT": ("North Central", 0.10, 1.30),
    "Plateau": ("North Central", 0.03, 0.90), "Kano": ("North West", 0.08, 0.90),
    "Kaduna": ("North West", 0.05, 0.95), "Rivers": ("South South", 0.08, 1.15),
    "Delta": ("South South", 0.04, 1.00), "Edo": ("South South", 0.03, 0.95),
    "Enugu": ("South East", 0.05, 0.95), "Anambra": ("South East", 0.06, 1.05),
    "Abia": ("South East", 0.04, 0.95), "Borno": ("North East", 0.02, 0.75),
    "Bauchi": ("North East", 0.03, 0.80),
}
EMPLOYMENT = {  # type: (weight, median monthly income NGN)
    "salaried": (0.38, 260_000), "self_employed": (0.22, 190_000),
    "trader": (0.22, 150_000), "gig_worker": (0.10, 95_000), "student": (0.08, 45_000),
}


def generate_loans(macro: pd.DataFrame, n: int = 20_000, seed: int = 42,
                   start: str = "2021-01-01", end: str = "2025-06-30") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.integers(pd.Timestamp(start).value, pd.Timestamp(end).value, n))
    df = pd.DataFrame({"loan_id": [f"LN{100000 + i}" for i in range(n)],
                       "application_date": dates.normalize()})

    names = list(STATES)
    df["state"] = rng.choice(names, n, p=np.array([STATES[s][1] for s in names]) /
                             sum(STATES[s][1] for s in names))
    df["region"] = df["state"].map(lambda s: STATES[s][0])
    df["employment_type"] = rng.choice(list(EMPLOYMENT), n,
                                       p=[v[0] for v in EMPLOYMENT.values()])
    df["age"] = np.clip(rng.normal(32, 8, n), 18, 65).round().astype(int)
    df["gender"] = rng.choice(["female", "male"], n, p=[0.47, 0.53])
    df["education"] = rng.choice(["secondary", "ond_nce", "bsc_hnd", "postgraduate"], n,
                                 p=[0.22, 0.20, 0.45, 0.13])

    base = df["employment_type"].map(lambda e: EMPLOYMENT[e][1])
    mult = df["state"].map(lambda s: STATES[s][2])
    edu_boost = df["education"].map({"secondary": 0.85, "ond_nce": 0.95, "bsc_hnd": 1.1,
                                     "postgraduate": 1.35})
    df["monthly_income_ngn"] = (base * mult * edu_boost *
                                rng.lognormal(0, 0.35, n)).round(-3).clip(20_000, 3_000_000)
    df["loan_amount_ngn"] = (df["monthly_income_ngn"] * rng.uniform(0.2, 1.3, n)
                             ).round(-3).clip(10_000, 2_000_000)
    df["tenor_months"] = rng.choice([1, 2, 3, 6, 12], n, p=[0.25, 0.2, 0.25, 0.2, 0.1])
    df["monthly_interest_rate"] = rng.choice([0.04, 0.06, 0.08, 0.10, 0.12], n,
                                             p=[0.15, 0.25, 0.30, 0.20, 0.10])
    df["loan_purpose"] = rng.choice(["business", "emergency", "education", "rent", "consumer"],
                                    n, p=[0.30, 0.25, 0.10, 0.15, 0.20])
    df["channel"] = rng.choice(["app", "ussd", "agent", "web"], n, p=[0.55, 0.15, 0.10, 0.20])
    df["device_type"] = rng.choice(["android", "ios", "feature_phone"], n, p=[0.72, 0.18, 0.10])
    df["prior_loans"] = rng.poisson(2.2, n)
    df["prior_defaults"] = rng.binomial(df["prior_loans"], 0.09)
    df["mobile_money_txn_30d"] = rng.negative_binomial(5, 0.15, n)
    df["avg_wallet_balance_ngn"] = (df["monthly_income_ngn"] * rng.beta(1.5, 6, n)).round(-2)
    df["salary_account"] = ((df["employment_type"] == "salaried") &
                            (rng.random(n) < 0.7)).astype(int)
    df["existing_debt_ngn"] = (df["monthly_income_ngn"] * rng.exponential(0.5, n) *
                               (rng.random(n) < 0.45)).round(-3)
    bureau = np.clip(rng.normal(610, 70, n) - 45 * df["prior_defaults"], 300, 850).round()
    df["bureau_score"] = np.where(rng.random(n) < 0.28, np.nan, bureau)

    df["year"] = df["application_date"].dt.year
    df = df.merge(macro, on="year", how="left")

    # ---- Data-generating process for default (documented ground truth) ----
    lti = df["loan_amount_ngn"] / df["monthly_income_ngn"]
    dti = df["existing_debt_ngn"] / df["monthly_income_ngn"]
    bureau_z = ((df["bureau_score"].fillna(580) - 600) / 70)
    logit = (-3.45
             + 1.0 * lti + 0.6 * dti
             + 0.9 * (df["prior_defaults"] > 0)
             - 0.55 * bureau_z
             + 0.25 * df["bureau_score"].isna()               # informative missingness
             - 0.55 * df["salary_account"]
             - 0.18 * np.log1p(df["mobile_money_txn_30d"])
             + 0.30 * (df["tenor_months"] >= 6)
             + 7.0 * df["monthly_interest_rate"]
             + 0.45 * (df["employment_type"] == "gig_worker")
             + 0.60 * (df["employment_type"] == "student")
             - 0.012 * (df["age"] - 32)
             + 0.30 * (df["device_type"] == "feature_phone")
             + 0.035 * (df["inflation_rate"] - 18)               # macro stress
             + rng.normal(0, 0.45, n))
    p = 1 / (1 + np.exp(-logit))
    df["default"] = (rng.random(n) < p).astype(int)

    # ---- Inject realistic messiness ----
    df = df.drop(columns=["year"])
    typo = rng.random(n)
    df.loc[typo < 0.03, "state"] = df.loc[typo < 0.03, "state"].str.upper()
    df.loc[(typo >= 0.03) & (typo < 0.05), "state"] = df.loc[(typo >= 0.03) & (typo < 0.05),
                                                              "state"].str.lower() + " "
    df.loc[rng.random(n) < 0.03, "monthly_income_ngn"] = np.nan
    df.loc[rng.random(n) < 0.002, "age"] = -1                      # impossible values
    df = pd.concat([df, df.sample(60, random_state=seed)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
