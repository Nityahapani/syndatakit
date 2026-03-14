# syndatakit

**A synthetic data generator for finance & econometrics.**

```bash
pip install syndatakit
syndatakit generate fred_macro --rows 1000000 --output training_data.csv
```

One command. One million synthetic macroeconomic observations. Statistically realistic. No real individuals. No data use agreement.

---

## What syndatakit is

syndatakit is a **synthetic data generator** — not a collection of stored datasets.

When you run `syndatakit generate hmda --rows 1000000`, it creates one million brand new synthetic mortgage application records that never existed. They are statistically similar to real HMDA data but contain zero real people. The package does not store or ship those rows — it generates them on demand, every time, in seconds.

The package ships with **10 built-in dataset profiles** — statistical models learned from published government aggregate statistics (CFPB reports, Federal Reserve bulletins, BLS data releases, etc.). Each profile captures the distributions, correlations, and structure of a real public dataset without containing any actual records from it.

You can also **bring your own data**:

```python
import pandas as pd
from syndatakit.generators import GaussianCopulaGenerator

# Load your own real data
real = pd.read_csv("my_loan_book.csv")

# Fit the generator on it
gen = GaussianCopulaGenerator()
gen.fit(real)

# Generate unlimited synthetic versions
synthetic = gen.sample(100_000)
synthetic.to_csv("synthetic_loan_book.csv", index=False)
```

A bank feeds in their actual loan portfolio, fits the generator, and produces synthetic training data that matches their specific book — not a generic average. The 10 built-in profiles are the out-of-the-box experience so you can start immediately without any data of your own.

---

## Why syndatakit

Training ML models in finance requires realistic data. But real data is locked behind NDAs, data use agreements, and privacy regulations.

syndatakit learns the statistical structure of financial datasets — distributions, correlations, temporal dynamics, stylized facts — and generates unlimited synthetic records that preserve that structure without exposing any real individuals.

**No data use agreements. No PII. No legal review.**

---

## Install

```bash
pip install syndatakit                # core — Copula + VAR + Panel generators
pip install syndatakit[api]           # + Flask REST API server
pip install syndatakit[io]            # + Parquet, Arrow, Stata, SAS, Excel
pip install syndatakit[deep]          # + CTGAN deep generator (requires PyTorch)
pip install syndatakit[all]           # everything
```

Requires Python 3.9+. Core dependencies: `pandas`, `numpy`, `scipy`. No GPU required.

---

## Quick start

```bash
# See all 10 built-in dataset profiles
syndatakit list

# Generate 5,000 synthetic mortgage applications
syndatakit generate hmda --rows 5000 --output mortgages.csv

# Generate with filters — high-DTI denied applicants in CA/TX
syndatakit generate hmda \
  --rows 1000 \
  --filter state:CA,TX \
  --filter dti_min:45 \
  --filter action_taken:2,3,7 \
  --output highrisk.csv

# Generate macro data under a recession scenario
syndatakit generate fred_macro \
  --rows 2000 \
  --scenario recession \
  --intensity 0.8 \
  --output recession_macro.csv

# Full fidelity report — how well does the synthetic data match the real distribution?
syndatakit evaluate real.csv synthetic.csv --type time_series

# Full privacy audit — is there any risk of re-identification?
syndatakit audit real.csv synthetic.csv

# Validate your own data before fitting
syndatakit validate my_data.csv

# Start REST API
syndatakit serve --port 8080
```

---

## Python API

```python
from syndatakit.generators import GaussianCopulaGenerator
from syndatakit.generators.time_series import VARGenerator
from syndatakit.catalog import load_seed
from syndatakit.fidelity import fidelity_report
from syndatakit.privacy import privacy_audit
from syndatakit.calibration import apply_scenario

# ── Using a built-in profile ──────────────────────────────────────────────────

gen = GaussianCopulaGenerator()
gen.fit(load_seed("hmda"))

df = gen.sample(10_000)
df_highrisk = gen.sample(
    1000,
    filters={"state": ["CA", "TX"], "dti_min": 45, "action_taken": ["2", "3"]},
)

# ── Using your own data ───────────────────────────────────────────────────────

import pandas as pd
real = pd.read_csv("my_data.csv")

gen = GaussianCopulaGenerator()
gen.fit(real)
synthetic = gen.sample(100_000)

# ── Time series ───────────────────────────────────────────────────────────────

gen_ts = VARGenerator(lags=2, time_col="year")
gen_ts.fit(load_seed("fred_macro"))
df_macro = gen_ts.sample(500)

# Conditioned on recession scenario
df_recession = apply_scenario(df_macro, "recession", intensity=0.9)

# ── Fidelity report ───────────────────────────────────────────────────────────

real = load_seed("fred_macro")
report = fidelity_report(real, df_macro, dataset_type="time_series")
print(f"Overall fidelity: {report['summary']['overall_fidelity']}%")
print(f"Stationarity:     {report['temporal']['stationarity']['_summary']['agreement_rate']}%")
print(f"Cointegration:    {report['temporal']['cointegration']['_summary']['agreement_rate']}%")

# ── Privacy audit ─────────────────────────────────────────────────────────────

audit = privacy_audit(real, df_macro, n_attacks=500)
print(f"Risk level:       {audit['verdict']['overall_risk']}")
print(f"Recommendation:   {audit['verdict']['recommendation']}")
```

---

## Built-in dataset profiles

These 10 profiles are included so you can generate synthetic data immediately, without any source data of your own. Each profile was built from published government aggregate statistics — not from individual records.

| ID | Name | Vertical | Columns | Source |
|---|---|---|---|---|
| `hmda` | HMDA Mortgage Applications | Credit & Lending | 7 | CFPB 2022 |
| `fdic` | FDIC Bank Call Reports | Credit & Lending | 12 | FDIC SDI 2023 |
| `credit_risk` | Consumer Credit Risk | Credit & Lending | 10 | CFPB derived |
| `edgar` | SEC EDGAR Financial Statements | Capital Markets | 13 | SEC XBRL 2023 |
| `cftc` | CFTC Commitments of Traders | Capital Markets | 10 | CFTC COT 2023 |
| `fred_macro` | FRED Macroeconomic Indicators | Macro & Central Bank | 15 | Federal Reserve |
| `bls` | BLS Employment & Wages | Macro & Central Bank | 9 | BLS QCEW 2022 |
| `world_bank` | World Bank Development Indicators | Macro & Central Bank | 12 | WDI 2022 |
| `irs_soi` | IRS Statistics of Income | Tax & Income | 11 | IRS SOI 2021 |
| `census_acs` | Census ACS Income & Housing | Tax & Income | 11 | Census ACS 2022 |

You are not limited to these 10. Any tabular dataset can be used with `gen.fit(your_df)`.

---

## Generators

| Generator | Best for | Auto-selected for |
|---|---|---|
| `GaussianCopulaGenerator` | Cross-sectional tabular data | hmda, fdic, credit_risk, edgar, cftc, irs_soi, census_acs |
| `VARGenerator` | Multivariate time series | fred_macro, bls |
| `FixedEffectsGenerator` | Panel data (entity × time) | world_bank, fdic |
| `CTGANGenerator` | Complex non-linear relationships | any (`pip install syndatakit[deep]`) |

The CLI auto-selects the right generator for each built-in profile. When using your own data, pick the one that matches your data type.

---

## Fidelity metrics

Every generated dataset can be evaluated across six dimensions:

**Marginal** — per-column distributional similarity (KS test for numeric, TVD for categorical)

**Joint** — Spearman correlation matrix distance between real and synthetic

**Temporal** — stationarity (ADF), cointegration (Engle-Granger), structural breaks (Chow scan), Granger causality

**Stylized facts** — fat tails, skewness sign, autocorrelation structure, ARCH effects

**Downstream (TSTR)** — train a model on synthetic data, test on real held-out data, compare to training on real data

**Privacy** — exact copy check, membership inference, singling-out risk, linkability risk

---

## Privacy

```python
from syndatakit.privacy import privacy_audit, format_audit
from syndatakit.privacy.dp import PrivacyBudget, laplace_mechanism

# Full privacy audit
audit = privacy_audit(real_df, synthetic_df, n_attacks=500)
print(format_audit(audit))

# Differential privacy — add calibrated noise to statistics
budget = PrivacyBudget(epsilon=1.0)
noisy_mean = laplace_mechanism(
    real_df["loan_amount"].mean(),
    sensitivity=1_000_000,
    epsilon=0.5,
    budget=budget,
)
```

---

## Calibration

```python
from syndatakit.calibration import apply_scenario, match_moments
from syndatakit.calibration.priors import get_priors

# 5 built-in economic scenarios
df_recession      = apply_scenario(df, "recession",        intensity=0.9)
df_severe         = apply_scenario(df, "severe_recession",  intensity=1.0)
df_rate_shock     = apply_scenario(df, "rate_shock",        intensity=1.0)
df_credit_crisis  = apply_scenario(df, "credit_crisis",     intensity=0.8)
df_expansion      = apply_scenario(df, "expansion",         intensity=0.5)

# Post-hoc moment calibration
calibrated = match_moments(real_df, synthetic_df)

# Bayesian priors — stabilises generators on small datasets (< 500 rows)
priors = get_priors("hmda")
gen = GaussianCopulaGenerator(priors=priors)
gen.fit(small_df)   # works well even with 50 rows
```

---

## REST API

```bash
syndatakit serve --port 8080
# Interactive docs: http://localhost:8080/docs
```

```python
import requests

# Generate synthetic data
r = requests.post("http://localhost:8080/generate", json={
    "dataset":   "fred_macro",
    "rows":      2000,
    "scenario":  "recession",
    "intensity": 0.9,
})
df = pd.DataFrame(r.json()["data"])

# Fidelity evaluation
r = requests.post("http://localhost:8080/evaluate",
    files={"real": open("real.csv"), "synthetic": open("syn.csv")},
    data={"type": "time_series"},
)

# Privacy audit
r = requests.post("http://localhost:8080/audit",
    files={"real": open("real.csv"), "synthetic": open("syn.csv")},
    data={"attacks": "500"},
)
```

All endpoints: `GET /datasets` · `GET /datasets/{id}` · `GET /datasets/{id}/sample` · `POST /generate` · `POST /evaluate` · `POST /audit` · `GET /scenarios` · `POST /scenario/apply` · `POST /validate` · `GET /health`

---

## IO formats

```python
from syndatakit.io import write, read

write(df, "output.csv")        # CSV
write(df, "output.parquet")    # Parquet  (pip install syndatakit[io])
write(df, "output.dta")        # Stata
write(df, "output.feather")    # Apache Arrow
write(df, "output.xlsx")       # Excel
write(df, "output.json")       # JSON

df = read("my_data.dta")       # reads Stata, SAS, CSV, Parquet, Arrow, JSON, Excel
```

---

## Architecture

```
syndatakit/
├── generators/
│   ├── base.py                  BaseGenerator ABC — fit(), sample(), fit_sample()
│   ├── cross_sectional/         GaussianCopulaGenerator
│   ├── time_series/             VARGenerator
│   ├── panel/                   FixedEffectsGenerator
│   └── deep/                    CTGANGenerator (pip install syndatakit[deep])
├── fidelity/
│   ├── marginal.py              KS + TVD per-column scores
│   ├── joint.py                 Correlation matrix distance
│   ├── temporal/                Stationarity, cointegration, breaks, causality
│   ├── stylized_facts.py        Fat tails, ARCH, autocorrelation
│   ├── downstream.py            TSTR train/test evaluation
│   └── report.py                Unified report assembler
├── calibration/
│   ├── moment_matching.py       Post-hoc moment calibration
│   ├── priors.py                Bayesian priors + MAP blending
│   └── scenario.py              Economic scenario calibration
├── privacy/
│   ├── dp.py                    Laplace + Gaussian mechanisms, budget tracker
│   ├── disclosure.py            Membership inference attack
│   ├── singling_out.py          Quasi-identifier singling-out attack
│   ├── linkability.py           Nearest-neighbour linkability attack
│   └── audit.py                 Full privacy audit — runs all four tests
├── catalog/
│   └── loader.py                10 built-in dataset profiles + seed builders
└── io/
    ├── formats.py               CSV, Parquet, Arrow, Stata, SAS, Excel, JSON
    └── validators.py            Schema validation before fitting
```

---

## Contributing

**Adding a new dataset profile** — implement `_build_<id>()` in `catalog/loader.py` using published aggregate statistics (not individual records), add a `PriorSet` in `calibration/priors.py`, and add tests. See `CONTRIBUTING.md` for the full walkthrough.

**Adding a new generator** — subclass `BaseGenerator`, implement `fit()` and `sample()`, export from the relevant `__init__.py`. The interface is minimal by design.

---

## License

Business Source License 1.1 (BSL-1.1). Free to use for any purpose including commercial, except offering syndatakit itself as a hosted managed service without a commercial agreement. Self-hosting always free.

---

## Cloud version

Need higher row counts, managed infrastructure, compliance documentation, or team access?
A hosted cloud version is coming. [Join the waitlist →](https://syndatakit.com)
