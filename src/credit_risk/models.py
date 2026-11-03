"""Model training, evaluation, threshold optimisation, explainability."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline

from .features import FEATURES, build_preprocessor


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5))]),
        "gradient_boosting": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=300, max_leaf_nodes=15,
                l2_regularization=1.0, early_stopping=True, random_state=seed))]),
    }


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov: max separation of cumulative good/bad distributions
    (the standard credit-scoring discrimination metric)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_prob) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    return {"roc_auc": round(auc, 4), "gini": round(2 * auc - 1, 4),
            "ks": round(ks_statistic(y_true, y_prob), 4),
            "pr_auc": round(average_precision_score(y_true, y_prob), 4),
            "brier": round(brier_score_loss(y_true, y_prob), 4),
            "base_rate": round(float(np.mean(y_true)), 4)}


def fit_best(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame,
             y_val: pd.Series, seed: int = 42):
    """Fit every candidate, pick best validation AUC, calibrate it. Returns (model, table)."""
    rows, fitted = [], {}
    for name, pipe in candidate_models(seed).items():
        pipe.fit(X_train[FEATURES], y_train)
        m = evaluate(y_val, pipe.predict_proba(X_val[FEATURES])[:, 1])
        rows.append({"model": name, **m})
        fitted[name] = pipe
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best = table.iloc[0]["model"]
    # Probability calibration matters in lending: PD feeds pricing & provisioning.
    calibrated = _calibrate(fitted[best], X_val, y_val)
    return best, calibrated, table


def _calibrate(pipe: Pipeline, X_val, y_val):
    """Isotonic calibration on a held-out validation window (prefit pipeline)."""
    from sklearn.frozen import FrozenEstimator
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    cal.fit(X_val[FEATURES], y_val)
    return cal


def profit_curve(y_true: np.ndarray, pd_hat: np.ndarray, amount: np.ndarray,
                 interest_income: np.ndarray, lgd: float, thresholds=None) -> pd.DataFrame:
    """Expected profit if we approve everyone with PD <= threshold."""
    thresholds = np.linspace(0.05, 0.95, 91) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        approve = pd_hat <= t
        gain = interest_income[approve & (y_true == 0)].sum()
        loss = (lgd * amount[approve & (y_true == 1)]).sum()
        rows.append({"threshold": round(float(t), 3), "approval_rate": approve.mean(),
                     "bad_rate_approved": y_true[approve].mean() if approve.any() else np.nan,
                     "profit_ngn": gain - loss})
    return pd.DataFrame(rows)


def feature_importance(model, X: pd.DataFrame, y: pd.Series, seed: int = 42,
                       n_repeats: int = 5, sample: int = 3000) -> pd.DataFrame:
    """Model-agnostic permutation importance (AUC drop) on a sample."""
    idx = X.sample(min(sample, len(X)), random_state=seed).index
    r = permutation_importance(model, X.loc[idx, FEATURES], y.loc[idx], scoring="roc_auc",
                               n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def fairness_report(df: pd.DataFrame, pd_hat: np.ndarray, threshold: float,
                    group_cols=("gender", "region")) -> pd.DataFrame:
    """Approval rate, observed bad rate and disparate-impact ratio per group."""
    d = df.assign(pd_hat=pd_hat, approved=pd_hat <= threshold)
    out = []
    for col in group_cols:
        g = d.groupby(col).agg(n=("approved", "size"), approval_rate=("approved", "mean"),
                               mean_pd=("pd_hat", "mean"), observed_default=("default", "mean"))
        g["disparate_impact_vs_best"] = g["approval_rate"] / g["approval_rate"].max()
        g["attribute"] = col
        out.append(g.reset_index().rename(columns={col: "group"}))
    return pd.concat(out, ignore_index=True)
