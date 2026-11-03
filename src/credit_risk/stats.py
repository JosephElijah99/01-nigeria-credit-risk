"""Descriptive + inferential statistics for the credit portfolio."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def describe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df[cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
    d["skew"] = df[cols].skew()
    d["kurtosis"] = df[cols].kurt()
    d["missing_pct"] = df[cols].isna().mean() * 100
    return d.round(3)


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Chi-square test of independence + bias-corrected Cramér's V effect size."""
    ct = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    phi2 = max(0, chi2 / n - (ct.shape[0] - 1) * (ct.shape[1] - 1) / (n - 1))
    r = ct.shape[0] - (ct.shape[0] - 1) ** 2 / (n - 1)
    k = ct.shape[1] - (ct.shape[1] - 1) ** 2 / (n - 1)
    return float(chi2), float(p), float(np.sqrt(phi2 / max(min(k - 1, r - 1), 1e-9)))


def categorical_tests(df: pd.DataFrame, cats: list[str], target="default") -> pd.DataFrame:
    rows = []
    for c in cats:
        chi2, p, v = cramers_v(df[c], df[target])
        rows.append({"feature": c, "chi2": chi2, "p_value": p, "cramers_v": v,
                     "significant_5pct": p < 0.05})
    out = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    out["p_adj_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    return out.round(5).reset_index(drop=True)


def numeric_tests(df: pd.DataFrame, nums: list[str], target="default") -> pd.DataFrame:
    """Mann-Whitney U (non-parametric; credit variables are skewed) + effect size
    (rank-biserial correlation) comparing defaulters vs non-defaulters."""
    rows = []
    for c in nums:
        a = df.loc[df[target] == 1, c].dropna()
        b = df.loc[df[target] == 0, c].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 1 - 2 * u / (len(a) * len(b))
        rows.append({"feature": c, "median_default": a.median(), "median_good": b.median(),
                     "u_stat": u, "p_value": p, "rank_biserial": rbc})
    out = pd.DataFrame(rows)
    out["abs_effect"] = out["rank_biserial"].abs()
    return out.sort_values("abs_effect", ascending=False).drop(columns="abs_effect") \
              .round(5).reset_index(drop=True)


def proportion_ci(successes: int, n: int, alpha=0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def default_rate_by(df: pd.DataFrame, col: str, target="default") -> pd.DataFrame:
    g = df.groupby(col)[target].agg(["sum", "count"])
    g["default_rate"] = g["sum"] / g["count"]
    ci = [proportion_ci(int(s), int(n)) for s, n in zip(g["sum"], g["count"])]
    g["ci_low"], g["ci_high"] = zip(*ci)
    return g.sort_values("default_rate", ascending=False).round(4).reset_index()


def trend_test(df: pd.DataFrame, target="default") -> dict:
    """Does default rate rise with inflation? Spearman correlation across years + OLS slope."""
    yr = df.assign(year=df["application_date"].dt.year).groupby("year").agg(
        default_rate=(target, "mean"), inflation=("inflation_rate", "mean")).reset_index()
    rho, p = stats.spearmanr(yr["inflation"], yr["default_rate"])
    slope = stats.linregress(yr["inflation"], yr["default_rate"])
    return {"spearman_rho": float(rho), "p_value": float(p),
            "slope_per_pp_inflation": float(slope.slope), "n_years": int(len(yr))}


def logit_odds_ratios(df: pd.DataFrame, features: list[str], target="default") -> pd.DataFrame:
    """Inference-grade logistic regression (odds ratios, 95% CI, p-values) via statsmodels.
    Numeric features are z-scored so odds ratios are 'per 1 SD'."""
    import statsmodels.api as sm  # imported lazily: only needed for this analysis
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    X = (X - X.mean()) / X.std()
    X = sm.add_constant(X)
    fit = sm.Logit(df[target], X).fit(disp=False)
    ci = fit.conf_int()
    out = pd.DataFrame({"odds_ratio": np.exp(fit.params), "ci_low": np.exp(ci[0]),
                        "ci_high": np.exp(ci[1]), "p_value": fit.pvalues})
    return out.drop(index="const").sort_values("odds_ratio", ascending=False).round(4)


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    return tuple(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
