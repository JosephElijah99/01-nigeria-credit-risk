"""Framework-independent scoring logic (used by the API, batch jobs and tests)."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import ROOT, load_config
from .features import FEATURES, add_features
from .ingest import load_macro


def risk_band(pd_hat: float) -> str:
    return ("A - Low" if pd_hat < 0.08 else "B - Moderate" if pd_hat < 0.18
            else "C - Elevated" if pd_hat < 0.30 else "D - High")


class CreditScorer:
    def __init__(self, model_path: str | Path | None = None, meta_path: str | Path | None = None):
        cfg = load_config()
        self.model = joblib.load(model_path or ROOT / cfg["paths"]["model"])
        self.meta = json.load(open(meta_path or ROOT / cfg["paths"]["metadata"]))
        macro = load_macro(cfg).sort_values("year")
        self.latest_macro = macro.iloc[-1].to_dict()
        self.threshold = self.meta["decision_threshold"]

    def _frame(self, records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        for k in ("inflation_rate", "gdp_growth", "unemployment_rate"):
            if k not in df or df[k].isna().all():
                df[k] = self.latest_macro[k]          # default to latest macro conditions
        return add_features(df)

    def score(self, records: list[dict]) -> list[dict]:
        X = self._frame(records)[FEATURES]
        probs = self.model.predict_proba(X)[:, 1].clip(0.002, 0.98)  # isotonic can output hard 0/1
        return [{"probability_of_default": round(float(p), 4),
                 "risk_band": risk_band(p),
                 "decision": "approve" if p <= self.threshold else "decline",
                 "decision_threshold": self.threshold} for p in probs]
