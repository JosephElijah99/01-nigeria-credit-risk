"""FastAPI scoring service.

Run:  uvicorn app.main:app --reload      then open  http://localhost:8000/docs
"""
from __future__ import annotations
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from credit_risk.predictor import CreditScorer

logger = logging.getLogger("credit-api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scorer"] = CreditScorer()          # load model once at startup
    logger.info("Model loaded: %s", state["scorer"].meta["algorithm"])
    yield
    state.clear()


app = FastAPI(title="Nigeria Credit Risk API", version="1.0.0",
              description="Probability-of-default scoring for digital loan applications.",
              lifespan=lifespan)


class Applicant(BaseModel):
    age: int = Field(ge=18, le=80, examples=[34])
    gender: Literal["female", "male"]
    education: Literal["secondary", "ond_nce", "bsc_hnd", "postgraduate"]
    employment_type: Literal["salaried", "self_employed", "trader", "gig_worker", "student"]
    region: Literal["South West", "South East", "South South", "North Central",
                    "North West", "North East"]
    monthly_income_ngn: float = Field(gt=0, examples=[250000])
    loan_amount_ngn: float = Field(gt=0, le=5_000_000, examples=[150000])
    tenor_months: int = Field(ge=1, le=24, examples=[3])
    monthly_interest_rate: float = Field(ge=0, le=0.3, examples=[0.08])
    loan_purpose: Literal["business", "emergency", "education", "rent", "consumer"]
    channel: Literal["app", "ussd", "agent", "web"]
    device_type: Literal["android", "ios", "feature_phone"]
    prior_loans: int = Field(ge=0, examples=[2])
    prior_defaults: int = Field(ge=0, examples=[0])
    mobile_money_txn_30d: int = Field(ge=0, examples=[25])
    avg_wallet_balance_ngn: float = Field(ge=0, examples=[20000])
    salary_account: int = Field(ge=0, le=1, examples=[1])
    existing_debt_ngn: float = Field(ge=0, examples=[0])
    bureau_score: float | None = Field(default=None, ge=300, le=850,
                                       description="Null if no bureau record")


class Prediction(BaseModel):
    probability_of_default: float
    risk_band: str
    decision: Literal["approve", "decline"]
    decision_threshold: float


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "scorer" in state}


@app.get("/model-info")
def model_info():
    m = state["scorer"].meta
    return {k: m[k] for k in ("model_version", "algorithm", "trained_at", "train_period",
                              "decision_threshold", "metrics_test_out_of_time")}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    try:
        return state["scorer"].score([applicant.model_dump()])[0]
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(applicants: list[Applicant]):
    if not 0 < len(applicants) <= 1000:
        raise HTTPException(status_code=422, detail="Send between 1 and 1000 applicants")
    return state["scorer"].score([a.model_dump() for a in applicants])
