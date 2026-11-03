import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.clean import clean
from credit_risk.features import add_features
from credit_risk.generate import generate_loans
from credit_risk.ingest import load_macro
from credit_risk.models import ks_statistic, profit_curve
from credit_risk import stats as S


@pytest.fixture(scope="module")
def raw():
    return generate_loans(load_macro(), n=1500, seed=1)


def test_clean_removes_duplicates_and_standardises_states(raw):
    df, rep = clean(raw)
    assert df["loan_id"].is_unique
    assert rep["duplicates_removed"] > 0
    assert df["state"].isin(["Lagos", "Kano", "FCT"]).any()
    assert df["age"].dropna().between(18, 80).all()


def test_features_no_target_leakage(raw):
    df, _ = clean(raw)
    out = add_features(df)
    assert "default" not in [c for c in out.columns if c.startswith("loan_to")]
    assert (out["loan_to_income"].dropna() > 0).all()


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([.1, .2, .8, .9])) == 1.0


def test_profit_curve_monotone_approval():
    y = np.random.default_rng(0).integers(0, 2, 200)
    p = np.random.default_rng(1).random(200)
    pc = profit_curve(y, p, np.full(200, 1e5), np.full(200, 1e4), 0.85)
    assert pc["approval_rate"].is_monotonic_increasing


def test_wilson_interval_contains_estimate():
    lo, hi = S.proportion_ci(30, 200)
    assert lo < 0.15 < hi


# ---- API tests (skipped automatically if fastapi/model not available) ----
def test_api_predict():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from app.main import app
    payload = dict(age=34, gender="female", education="bsc_hnd", employment_type="salaried",
                   region="South West", monthly_income_ngn=250000, loan_amount_ngn=150000,
                   tenor_months=3, monthly_interest_rate=0.08, loan_purpose="business",
                   channel="app", device_type="android", prior_loans=2, prior_defaults=0,
                   mobile_money_txn_30d=25, avg_wallet_balance_ngn=20000, salary_account=1,
                   existing_debt_ngn=0, bureau_score=640)
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json=payload)
        assert r.status_code == 200
        assert 0 <= r.json()["probability_of_default"] <= 1
        bad = {**payload, "age": 12}
        assert c.post("/predict", json=bad).status_code == 422
