# 🏦 Nigeria Digital Lending - Credit Risk Scoring (End-to-End)

> Predict the probability that a Nigerian digital-loan applicant will default, choose the
> **profit-maximising approval cut-off**, audit the model for **fairness**, and serve it
> through a **FastAPI** microservice (Docker-ready).

**Domain:** Finance / FinTech  **Skills shown:** EDA · descriptive & inferential statistics ·
feature engineering · ML · calibration · out-of-time validation · explainability · fairness ·
MLOps (config, modular code, tests, Docker, CI) · REST API deployment

---

## 1. Business problem
Nigerian digital lenders (Carbon, FairMoney, Renmoney-style products) approve thousands of
small loans a day with thin credit files - only ~70% of applicants have a bureau score.
Approving a bad loan loses up to ~85% of principal; declining a good one loses interest income.
The business question is **"whom do we approve, so that expected profit is maximised?"**,
not just "what is the AUC?".

## 2. Data
| Source | How it is obtained | Notes |
|---|---|---|
| Loan book (20k loans, 2021-2025) | `generate.py` - simulated | Real lender data is confidential. The data-generating process is documented in code, includes Nigerian states/zones, NGN amounts, informative missing bureau scores, typos and duplicates. |
| Macro indicators (inflation, GDP growth, unemployment) | **World Bank API** (`ingest.py`, `--refresh-macro`) | Cached locally; falls back to `data/external/macro_fallback.csv` when offline. The fallback values are **approximate placeholders** - run `make ingest` for official figures. |

> ⚠️ **Honesty note.** Because the loans are synthetic, the *numbers* below demonstrate the
> methodology, not real-world Nigerian default rates. Swap in a real portfolio via
> `ingest.load_csv` / `ingest.load_sql` - the rest of the pipeline is unchanged.

## 3. Project structure
```
config/config.yaml        # all paths, seeds, split date, economics (LGD)
src/credit_risk/
  ingest.py               # World Bank API + CSV + SQL loaders, caching, fallback
  generate.py             # synthetic portfolio with documented DGP
  clean.py                # validation + data-quality report
  features.py             # leakage-safe features + sklearn ColumnTransformer
  models.py               # candidates, KS/Gini/AUC, calibration, profit curve, fairness
  stats.py                # chi-square, Mann-Whitney, Wilson CI, logit odds ratios, bootstrap
  predictor.py            # framework-free scoring class (shared by API/batch/tests)
  pipeline.py             # CLI: generate | train | all
app/main.py               # FastAPI service (validation, batch endpoint, timing middleware)
scripts/eda.py            # notebook-style EDA (percent format -> Jupyter/VS Code)
tests/                    # unit + API tests
Dockerfile · Makefile · CI workflow
```

## 4. Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all          # generate data -> clean -> train -> save model + metadata
make eda          # figures -> reports/figures, tables -> reports/tables
make test
make serve        # http://localhost:8000/docs
# or:  make docker
```
Example request:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
 "age":34,"gender":"female","education":"bsc_hnd","employment_type":"salaried",
 "region":"South West","monthly_income_ngn":250000,"loan_amount_ngn":150000,
 "tenor_months":3,"monthly_interest_rate":0.08,"loan_purpose":"business","channel":"app",
 "device_type":"android","prior_loans":2,"prior_defaults":0,"mobile_money_txn_30d":25,
 "avg_wallet_balance_ngn":20000,"salary_account":1,"existing_debt_ngn":0,"bureau_score":640}'
# {"probability_of_default":0.0557,"risk_band":"A - Low","decision":"approve",...}
```

## 5. Method highlights
* **Cleaning:** 60 duplicate IDs removed, state-name typos standardised, 42 impossible ages
  nulled, missingness reported (`reports/data_quality.json`).
* **No leakage:** only application-time features; imputation/scaling live *inside* the sklearn
  pipeline so they are fit on training data only.
* **Out-of-time validation:** train on the past, tune on a later window, test on the latest
  6 months - how the model would really be used.
* **Descriptive stats:** skewness, kurtosis, missingness; distributions motivate
  non-parametric tests.
* **Inferential stats:** chi-square + Cramér's V (Bonferroni-adjusted), Mann-Whitney U with
  rank-biserial effect size, two-proportion z-test, Wilson CIs, Spearman trend test,
  logistic-regression odds ratios with CIs. *With n = 20k nearly everything is "significant",
  so I report effect sizes, not just p-values.*
* **Models:** regularised Logistic Regression vs Gradient Boosting; best chosen on
  validation AUC, then **isotonic-calibrated** (PDs drive pricing and provisioning).
* **Decisioning:** approval cut-off chosen to maximise expected NGN profit on validation
  (not an arbitrary 0.5), then evaluated once on the untouched test set.
* **Fairness:** approval-rate and disparate-impact ratios by gender and region.

## 6. Results (out-of-time test, ~4.4k loans)
| Metric | Value |
|---|---|
| ROC AUC / Gini | 0.715 / 0.431 |
| KS | 0.325 |
| Brier score | 0.129 (base rate 17.3%) |
| Profit-optimal PD cut-off | 0.27 → approves ~89% of applicants |
| Fairness (disparate-impact ratio) | ≥ 0.95 across gender and regions |

Logistic regression matched/beat gradient boosting - unsurprising when the signal is mostly
additive. **Picking the simpler, more explainable model is the right call** in regulated
lending.

## 7. Limitations & next steps
* Synthetic data → replace with real portfolio; expect lower AUC and more drift.
* The inflation→default relationship rests on 5 annual points (inflation is constant within a
  year, so it is confounded with time). Treat it as illustrative; with real data use monthly macro data.
* Add reject inference (declined applicants have no outcome), PSI monitoring for drift,
  MLflow experiment tracking, and a scheduled retraining job (Airflow / GitHub Actions).
