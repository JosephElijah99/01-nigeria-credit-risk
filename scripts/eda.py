# %% [markdown]
# # Nigeria Digital Lending - EDA, Descriptive & Inferential Statistics
# Percent-format script: open in VS Code / Jupyter / JupyterLab (jupytext) as a notebook,
# or run directly:  `python scripts/eda.py`  (saves figures to reports/figures).

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from credit_risk import stats as S
from credit_risk.config import load_config, resolve
from credit_risk.features import CATEGORICAL, add_features
from credit_risk.models import ks_statistic
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

sns.set_theme(style="whitegrid", context="notebook")
cfg = load_config()
FIG = resolve(cfg["paths"]["figures"] + "/x").parent
TAB = resolve("reports/tables/x").parent

def save(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", dpi=130); plt.close()

df = add_features(pd.read_csv(resolve(cfg["paths"]["processed"]),
                              parse_dates=["application_date"]))
print(df.shape)

# %% [markdown]
# ## 1. Data overview & descriptive statistics
# %%
num_cols = ["age", "monthly_income_ngn", "loan_amount_ngn", "loan_to_income",
            "debt_to_income", "bureau_score", "mobile_money_txn_30d", "inflation_rate"]
desc = S.describe(df, num_cols)
desc.to_csv(TAB / "descriptive_stats.csv")
print(desc[["mean", "50%", "std", "skew", "missing_pct"]])
print(f"\nOverall default rate: {df['default'].mean():.2%}")

# %% [markdown]
# ## 2. Distributions (note the right-skew in money variables -> log transform / non-parametric tests)
# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, c in zip(ax.ravel(), ["monthly_income_ngn", "loan_amount_ngn", "loan_to_income",
                              "bureau_score", "age", "mobile_money_txn_30d"]):
    sns.histplot(df[c].dropna(), bins=50, kde=True, ax=a, color="#1b7f5c")
    a.set_title(f"{c}  (skew={df[c].skew():.2f})")
save("01_distributions")

# %% [markdown]
# ## 3. Default rate by segment, with Wilson 95% confidence intervals
# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, c in zip(ax, ["region", "employment_type", "tenor_months"]):
    t = S.default_rate_by(df, c)
    a.barh(t[c].astype(str), t["default_rate"],
           xerr=[t["default_rate"] - t["ci_low"], t["ci_high"] - t["default_rate"]],
           color="#c0392b", alpha=.85)
    a.axvline(df["default"].mean(), ls="--", c="k"); a.set_title(f"Default rate by {c}")
save("02_default_by_segment")

# %% [markdown]
# ## 4. Default rate vs. macro conditions (time dimension)
# %%
q = df.assign(q=df["application_date"].dt.to_period("Q").dt.to_timestamp()).groupby("q").agg(
    default_rate=("default", "mean"), inflation=("inflation_rate", "mean"), n=("default", "size"))
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(q.index, q["default_rate"] * 100, marker="o", c="#c0392b", label="Default rate %")
ax.set_ylabel("Default rate %")
ax2 = ax.twinx(); ax2.plot(q.index, q["inflation"], c="#2c3e50", ls="--", label="Inflation %")
ax2.set_ylabel("Inflation % (annual)"); ax.set_title("Quarterly default rate vs inflation")
save("03_default_vs_inflation")

# %% [markdown]
# ## 5. Correlations
# %%
corr = df[["default", "loan_to_income", "debt_to_income", "bureau_score", "age",
           "mobile_money_txn_30d", "monthly_interest_rate", "inflation_rate",
           "salary_account", "prior_defaults"]].corr(method="spearman")
plt.figure(figsize=(9, 7)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Spearman correlation"); save("04_correlation")

# %% [markdown]
# ## 6. Inferential statistics
# **H0:** the feature is independent of default. We report p-values (Bonferroni-adjusted),
# but also EFFECT SIZES - with n=20k almost everything is 'significant'; size is what matters.
# %%
cat_tests = S.categorical_tests(df, CATEGORICAL + ["tenor_months"])
num_tests = S.numeric_tests(df, ["loan_to_income", "debt_to_income", "bureau_score", "age",
                                 "mobile_money_txn_30d", "monthly_income_ngn",
                                 "monthly_interest_rate", "wallet_to_income"])
cat_tests.to_csv(TAB / "inference_categorical.csv", index=False)
num_tests.to_csv(TAB / "inference_numeric.csv", index=False)
print("\nChi-square + Cramer's V\n", cat_tests)
print("\nMann-Whitney U + rank-biserial\n", num_tests)

trend = S.trend_test(df)
print("\nInflation -> default trend:", trend)

# Two-proportion comparison: salary-account holders vs others
from scipy import stats as sp
a = df.loc[df.salary_account == 1, "default"]; b = df.loc[df.salary_account == 0, "default"]
z = (a.mean() - b.mean()) / np.sqrt(df.default.mean() * (1 - df.default.mean()) *
                                    (1 / len(a) + 1 / len(b)))
print(f"\nSalary account default {a.mean():.2%} vs {b.mean():.2%}; z={z:.2f}, p={2*sp.norm.sf(abs(z)):.2e}")

# %%
try:
    feats = ["loan_to_income", "debt_to_income", "bureau_score", "age", "prior_defaults",
             "mobile_money_txn_30d", "monthly_interest_rate", "inflation_rate", "salary_account"]
    odds = S.logit_odds_ratios(df, feats)
    odds.to_csv(TAB / "logit_odds_ratios.csv")
    print("\nLogit odds ratios (per +1 SD)\n", odds)
    plt.figure(figsize=(8, 5))
    o = odds.sort_values("odds_ratio")
    plt.errorbar(o["odds_ratio"], range(len(o)),
                 xerr=[o["odds_ratio"] - o["ci_low"], o["ci_high"] - o["odds_ratio"]], fmt="o")
    plt.yticks(range(len(o)), o.index); plt.axvline(1, ls="--", c="k")
    plt.title("Odds ratios per 1 SD (95% CI)"); save("05_odds_ratios")
except ImportError:
    print("statsmodels not installed - skipping odds-ratio analysis")

# %% [markdown]
# ## 7. Model results (run `python -m credit_risk.pipeline train` first)
# %%
pred = pd.read_csv(TAB / "test_predictions.csv")
fpr, tpr, _ = roc_curve(pred.y, pred.pd)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k--")
ax[0].set_title(f"ROC (out-of-time) KS={ks_statistic(pred.y, pred.pd):.2f}")
fp, mp = calibration_curve(pred.y, pred.pd, n_bins=10, strategy="quantile")
ax[1].plot(mp, fp, marker="o"); ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set_title("Calibration")
pc = pd.read_csv(TAB / "profit_curve_validation.csv")
ax[2].plot(pc.threshold, pc.profit_ngn / 1e6); ax[2].set_title("Profit vs PD cut-off (validation)")
ax[2].set_ylabel("NGN millions"); ax[2].set_xlabel("Approve if PD <=")
save("06_model_performance")

imp = pd.read_csv(TAB / "feature_importance.csv").head(12)
plt.figure(figsize=(8, 5)); sns.barplot(data=imp, y="feature", x="importance", color="#1b7f5c")
plt.title("Permutation importance (AUC drop)"); save("07_feature_importance")
print(pd.read_csv(TAB / "fairness.csv").round(3))
