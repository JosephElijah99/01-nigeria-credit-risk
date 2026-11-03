"""End-to-end pipeline CLI.

    python -m credit_risk.pipeline generate [--refresh-macro]
    python -m credit_risk.pipeline train
    python -m credit_risk.pipeline all
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .clean import clean
from .config import load_config, resolve
from .features import FEATURES, add_features
from .generate import generate_loans
from .ingest import load_macro
from .models import (evaluate, feature_importance, fairness_report, fit_best, profit_curve)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pipeline")


def step_generate(cfg, refresh_macro=False):
    macro = load_macro(cfg, refresh=refresh_macro)
    d = cfg["data"]
    raw = generate_loans(macro, d["n_rows"], cfg["seed"], d["start_date"], d["end_date"])
    raw.to_csv(resolve(cfg["paths"]["raw"]), index=False)
    clean_df, report = clean(raw)
    clean_df.to_csv(resolve(cfg["paths"]["processed"]), index=False)
    json.dump(report, open(resolve("reports/data_quality.json"), "w"), indent=2)
    log.info("Raw %d rows -> clean %d rows | %s", len(raw), len(clean_df), report)


def step_train(cfg):
    df = pd.read_csv(resolve(cfg["paths"]["processed"]), parse_dates=["application_date"])
    df = add_features(df).sort_values("application_date").reset_index(drop=True)
    target = cfg["model"]["target"]

    # Out-of-time split: train on the past, test on the most recent period (like production).
    test_mask = df["application_date"] >= cfg["split"]["test_start"]
    dev, test = df[~test_mask], df[test_mask]
    cut = int(len(dev) * 0.8)
    train, val = dev.iloc[:cut], dev.iloc[cut:]
    log.info("train=%d  val=%d  test(out-of-time)=%d", len(train), len(val), len(test))

    best, model, table = fit_best(train, train[target], val, val[target], cfg["seed"])
    log.info("Validation comparison:\n%s", table.to_string(index=False))

    pd_val = model.predict_proba(val[FEATURES])[:, 1]
    pd_test = model.predict_proba(test[FEATURES])[:, 1]

    def econ(frame):
        return (frame["loan_amount_ngn"] * frame["monthly_interest_rate"]
                * frame["tenor_months"]).to_numpy()

    pc = profit_curve(val[target].to_numpy(), pd_val, val["loan_amount_ngn"].to_numpy(),
                      econ(val), cfg["economics"]["lgd"])
    threshold = float(pc.loc[pc["profit_ngn"].idxmax(), "threshold"])
    test_pc = profit_curve(test[target].to_numpy(), pd_test, test["loan_amount_ngn"].to_numpy(),
                           econ(test), cfg["economics"]["lgd"], thresholds=[threshold])
    log.info("Profit-optimal PD threshold (chosen on validation): %.2f", threshold)

    test_metrics = evaluate(test[target], pd_test)
    log.info("OUT-OF-TIME TEST: %s", test_metrics)

    # Artifacts
    imp = feature_importance(model, test, test[target], cfg["seed"])
    fair = fairness_report(test, pd_test, threshold)
    rep = resolve("reports/tables/placeholder").parent
    table.to_csv(rep / "model_comparison.csv", index=False)
    imp.to_csv(rep / "feature_importance.csv", index=False)
    fair.to_csv(rep / "fairness.csv", index=False)
    pc.to_csv(rep / "profit_curve_validation.csv", index=False)
    pd.DataFrame({"y": test[target].to_numpy(), "pd": pd_test}).to_csv(
        rep / "test_predictions.csv", index=False)

    joblib.dump(model, resolve(cfg["paths"]["model"]))
    meta = {"model_version": __version__, "algorithm": best,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__, "features": FEATURES,
            "decision_threshold": threshold,
            "metrics_validation": table.iloc[0].to_dict(),
            "metrics_test_out_of_time": test_metrics,
            "test_profit_at_threshold_ngn": float(test_pc["profit_ngn"].iloc[0]),
            "test_approval_rate": float(test_pc["approval_rate"].iloc[0]),
            "train_period": [str(train["application_date"].min().date()),
                             str(train["application_date"].max().date())],
            "n_train": int(len(train))}
    json.dump(meta, open(resolve(cfg["paths"]["metadata"]), "w"), indent=2, default=str)
    log.info("Saved model + metadata. Test profit @threshold: NGN %s",
             f"{meta['test_profit_at_threshold_ngn']:,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "train", "all"])
    ap.add_argument("--refresh-macro", action="store_true",
                    help="pull fresh macro data from the World Bank API")
    args = ap.parse_args()
    cfg = load_config()
    if args.command in ("generate", "all"):
        step_generate(cfg, args.refresh_macro)
    if args.command in ("train", "all"):
        step_train(cfg)


if __name__ == "__main__":
    main()
