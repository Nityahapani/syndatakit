# syndatakit

**Research-grade synthetic data for finance & econometrics.**

[![PyPI](https://img.shields.io/pypi/v/syndatakit)](https://pypi.org/project/syndatakit)
[![Python](https://img.shields.io/pypi/pyversions/syndatakit)](https://pypi.org/project/syndatakit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![JOSS](https://img.shields.io/badge/JOSS-under%20review-orange)](https://joss.theoj.org)
[![Tests](https://img.shields.io/badge/tests-168%20passing-brightgreen)](https://github.com/Nityahapani/syndatakit/actions)

```
pip install syndatakit
syndatakit generate fred_macro --rows 1000000 --output training_data.csv
```

One command. One million synthetic macroeconomic observations. Statistically realistic. No real individuals. No data use agreement.

---

## What syndatakit is

syndatakit is a **synthetic data generator** — not a collection of stored datasets.

When you run `syndatakit generate hmda --rows 1000000`, it creates one million brand-new synthetic mortgage application records that never existed. They are statistically identical to real HMDA data but contain zero real people. The package does not store those rows — it generates them on demand, every time, in seconds.

The package ships with **18 built-in dataset profiles** — statistical models learned from published government aggregate statistics (CFPB reports, Federal Reserve bulletins, BLS releases, and more). Each profile captures the distributions, correlations, and temporal dynamics of a real public dataset without containing any individual records from it.

You can also **bring your own data** — point it at any CSV and it fits the generator in seconds, then generates unlimited synthetic versions.

---

## Why syndatakit

Most synthetic data tools are built for ML engineers who need more rows. syndatakit is built for researchers who need rows that pass peer review.

Real financial microdata is locked behind NDAs, data use agreements, and IRB approvals that take months. Naive random data fails every econometric test you'd want to run — ADF rejects stationarity, Johansen finds no cointegration, ARCH tests see no volatility clustering. syndatakit learns the full statistical structure and generates data that preserves it.

**No data use agreements. No PII. No legal review. Full replication packages.**

---

## Install

```bash
pip install syndatakit                # core — all generators, 18 profiles, CLI, REST API
pip install syndatakit[timeseries]    # + VECM + GARCH time series generator
pip install syndatakit[vine]          # + Vine copula with asymmetric tail dependence
pip install syndatakit[io]            # + Parquet, Arrow, Stata .dta, SAS, Excel formats
pip install syndatakit[deep]          # + CTGAN deep tabular generator (requires PyTorch)
pip install syndatakit[all]           # everything
```

Requires Python ≥ 3.9. Core dependencies: `pandas`, `numpy`, `scipy`. No GPU required.

---

## Quick start

```bash
# List all 18 built-in dataset profiles
syndatakit list

# Generate 1M HMDA mortgage applications
syndatakit generate hmda --rows 1000000 --output mortgages.csv

# Generate with filters — high-DTI denied applicants in CA and TX
syndatakit generate hmda \
  --rows 10000 \
  --filter state:CA,TX \
  --filter dti_min:45 \
  --output highrisk.csv

# Generate macro data under a recession scenario (80% intensity)
syndatakit generate fred_macro \
  --scenario recession \
  --intensity 0.8 \
  --rows 2000 \
  --output recession_macro.csv

# Full fidelity report — six dimensions of statistical validity
syndatakit evaluate real.csv synthetic.csv --type time_series

# Full privacy audit — four ENISA tests
syndatakit audit real.csv synthetic.csv --attacks 500

# Use your own data
syndatakit generate --input my_loan_book.csv --rows 1000000 --output synthetic.csv

# Start the REST API server
syndatakit serve --port 8080
```

---

## Python API

```python
from syndatakit.generators import GaussianCopulaGenerator
from syndatakit.generators.time_series import VARGenerator
from syndatakit.catalog import load_seed
from syndatakit.calibration import get_priors, apply_scenario
from syndatakit.fidelity import fidelity_report
from syndatakit.privacy import privacy_audit

# ── Built-in profile ──────────────────────────────────────────────────────────

gen = GaussianCopulaGenerator(priors=get_priors("hmda"))
gen.fit(load_seed("hmda"))

df = gen.sample(10_000)
df_highrisk = gen.sample(
    1_000,
    filters={"state": ["CA", "TX"], "dti_min": 45, "action_taken": ["2", "3"]},
)

# ── Your own data ─────────────────────────────────────────────────────────────

import pandas as pd
real = pd.read_csv("my_loan_book.csv")

gen = GaussianCopulaGenerator()
gen.fit(real)
synthetic = gen.sample(100_000)
synthetic.to_csv("synthetic_loan_book.csv", index=False)

# ── Time series ───────────────────────────────────────────────────────────────

gen_ts = VARGenerator(lags=2)
gen_ts.fit(load_seed("fred_macro"))
df_macro = gen_ts.sample(500)

# Apply scenario calibration
df_recession = apply_scenario(df_macro, "recession", intensity=0.9)

# ── Fidelity report ───────────────────────────────────────────────────────────

report = fidelity_report(load_seed("fred_macro"), df_macro, dataset_type="time_series")
print(f"Overall fidelity:  {report['summary']['overall_fidelity']}%")
print(f"Stationarity:      {report['temporal']['stationarity']['_summary']['agreement_rate']}%")
print(f"Cointegration:     {report['temporal']['cointegration']['_summary']['agreement_rate']}%")

# ── Privacy audit ─────────────────────────────────────────────────────────────

audit = privacy_audit(load_seed("hmda"), df, n_attacks=500)
print(f"Risk level:       {audit['verdict']['overall_risk']}")
print(f"Recommendation:   {audit['verdict']['recommendation']}")
```

---

## 18 built-in dataset profiles

All profiles are built from published government aggregate statistics — not from individual records. Fidelity scores are computed by `validation/fidelity_engine.py` at N=50,000 synthetic rows.

| Profile | Dataset | Vertical | Cols | Source | Fidelity |
|---|---|---|---|---|---|
| `hmda` | HMDA Mortgage Applications | Credit & Lending | 7 | CFPB 2022 | 91.8% |
| `fdic` | FDIC Bank Call Reports | Credit & Lending | 12 | FDIC SDI 2023 | 95.0% |
| `credit_risk` | Consumer Credit Risk | Credit & Lending | 10 | CFPB derived | validated |
| `edgar` | SEC EDGAR Financial Statements | Capital Markets | 13 | SEC XBRL 2023 | 90.4% |
| `cftc` | CFTC Commitments of Traders | Capital Markets | 10 | CFTC COT 2023 | 95.0% |
| `equity_returns` | Equity Returns & Risk Factors | Capital Markets | 15 | CRSP/Compustat | 90.9% |
| `corporate_bonds` | Corporate Bond Market Data | Capital Markets | 15 | TRACE/FINRA | 90.8% |
| `fred_macro` | FRED Macroeconomic Indicators | Macro & Central Bank | 15 | Federal Reserve | 88.6% |
| `bls` | BLS Employment & Wages | Macro & Central Bank | 9 | BLS QCEW 2022 | 94.5% |
| `world_bank` | World Bank Dev. Indicators | Macro & Central Bank | 12 | WDI 2022 | 87.8% |
| `central_bank_rates` | Central Bank Policy Rates | Macro & Central Bank | 11 | BIS/central banks | 92.0% |
| `irs_soi` | IRS Statistics of Income | Tax & Income | 11 | IRS SOI 2021 | 81.7% |
| `census_acs` | Census ACS Income & Housing | Tax & Income | 11 | ACS 5-Year 2022 | 93.4% |
| `insurance_claims` | P&C Insurance Claims | Insurance | 13 | NAIC Schedule P | 91.0% |
| `life_insurance` | Life Insurance & Mortality | Insurance | 13 | SOA/LIMRA | 88.9% |
| `commercial_real_estate` | Commercial Real Estate | Real Estate | 15 | NCREIF/CoStar | 92.3% |
| `rental_market` | Residential Rental Market | Real Estate | 13 | HUD FMR/Zillow | 94.1% |
| `commodity_prices` | Commodity Price Returns | Commodities | 13 | EIA/USDA/LME | 94.4% |

*"validated" = profile built from published statistics; individual-record test data not publicly available. Fidelity score = 0.45 × marginal moment matching + 0.30 × KS distributional fit + 0.25 × Spearman correlation Frobenius distance. [Full reproduction code →](validation/)*

---

## Generators

| Generator | Best for | Install |
|---|---|---|
| `GaussianCopulaGenerator` | Cross-sectional tabular data | core |
| `VARGenerator` | Multivariate time series with VECM + GARCH | `pip install syndatakit[timeseries]` |
| `DPGaussianCopulaGenerator` | Formal (ε,δ)-differential privacy guarantee | core |
| `VineCopulaGenerator` | Asymmetric tail dependence (Clayton, Gumbel, Frank, Joe) | `pip install syndatakit[vine]` |
| `FixedEffectsGenerator` | Panel data (entity × time) | core |
| `CTGANGenerator` | Complex non-linear relationships | `pip install syndatakit[deep]` |

All generators share the same interface: `fit()`, `sample()`, `fit_sample()`. The CLI auto-selects the appropriate generator for each built-in profile.

---

## Fidelity metrics

Every generated dataset can be evaluated across six independent dimensions via `fidelity_report(real, synthetic)`.

**Marginal** — Kolmogorov-Smirnov test for numeric columns, Total Variation Distance for categorical. Per-column scores 0–100%.

**Joint** — Spearman rank correlation matrix distance. Frobenius norm between real and synthetic correlation matrices. Detects when the generator preserves marginals but breaks relationships.

**Temporal** — ADF stationarity, Engle-Granger cointegration, Chow structural breaks, Granger causality direction. Critical for time series and macro data.

**Stylized facts** — fat tails (kurtosis), skewness sign, autocorrelation, ARCH volatility clustering. Essential for financial returns data.

**Downstream (TSTR)** — train on synthetic, test on real. Compare OLS coefficients to training on real data. Ratio near 1.0 = valid substitute for research.

**Privacy** — exact copy check, membership inference AUC, singling-out rate, linkability risk. Full ENISA-framework verdict with recommendation.

**Combined score:** `0.45 × marginal + 0.30 × KS + 0.25 × correlation`

---

## Scenario calibration

Five built-in economic scenarios with configurable intensity (0.0–1.0):

```python
from syndatakit.calibration import apply_scenario

df_recession     = apply_scenario(df, "recession",       intensity=0.9)
df_severe        = apply_scenario(df, "severe_recession", intensity=1.0)
df_rate_shock    = apply_scenario(df, "rate_shock",       intensity=0.8)
df_credit_crisis = apply_scenario(df, "credit_crisis",    intensity=0.7)
df_expansion     = apply_scenario(df, "expansion",        intensity=0.5)
```

Scenarios interpolate all variable moments continuously from baseline to stressed values. Use intensity to represent partial stress — e.g., `intensity=0.5` on `recession` gives a mild slowdown rather than a full contraction.

---

## Differential privacy

```python
from syndatakit.generators.cross_sectional import DPGaussianCopulaGenerator
from syndatakit.privacy import privacy_audit, format_audit
from syndatakit.privacy.dp import PrivacyBudget, laplace_mechanism

# Formal (ε, δ)-DP generator — ε=1.0 balances utility and privacy
gen = DPGaussianCopulaGenerator(epsilon=1.0, delta=1e-5)
gen.fit(real_df)
syn = gen.sample(10_000)

# Budget is tracked — never accidentally exceed it
print(f"ε used: {gen.epsilon_used:.3f} / {gen.epsilon}")

# Laplace mechanism for individual statistics
budget = PrivacyBudget(epsilon=2.0)
noisy_mean = laplace_mechanism(
    real_df["loan_amount"].mean(),
    sensitivity=1_000_000,
    epsilon=0.5,
    budget=budget,
)

# Full four-test privacy audit
audit = privacy_audit(real_df, syn, n_attacks=500)
print(format_audit(audit))
```

The four ENISA privacy tests run automatically on every generated dataset:

1. **Exact copy check** — count of synthetic rows identical to any training row. Always zero for continuous distributions.
2. **Membership inference** — shadow model AUC test. Near 0.5 = no advantage; above 0.7 = memorisation risk.
3. **Singling-out** — random quasi-identifier attack. Can an adversary uniquely identify one individual by matching on 2–4 attributes?
4. **Linkability** — nearest-neighbour cross-dataset attack. Can a synthetic record be linked back to a specific real individual? Baseline 0.5.

---

## REST API

```bash
syndatakit serve --port 8080
# Interactive docs at http://localhost:8080/docs
```

```python
import requests, pandas as pd

# Generate
r = requests.post("http://localhost:8080/generate", json={
    "dataset":   "fred_macro",
    "rows":      2000,
    "scenario":  "recession",
    "intensity": 0.9,
})
df = pd.DataFrame(r.json()["data"])

# Evaluate fidelity
r = requests.post("http://localhost:8080/evaluate",
    files={"real": open("real.csv"), "synthetic": open("syn.csv")},
    data={"type": "time_series"},
)

# Privacy audit
r = requests.post("http://localhost:8080/audit",
    files={"real": open("real.csv"), "synthetic": open("syn.csv")},
    data={"attacks": "500"},
)
print(r.json()["data"]["verdict"]["recommendation"])

# Upload your own CSV and generate from it
r = requests.post("http://localhost:8080/generate/custom",
    files={"file": open("my_data.csv")},
    data={"rows": "100000"},
)
```

**Endpoints:** `GET /datasets` · `GET /datasets/{id}` · `POST /generate` · `POST /generate/custom` · `POST /evaluate` · `POST /audit` · `POST /scenario/apply` · `POST /validate` · `GET /scenarios` · `GET /health`

---

## IO formats

```python
from syndatakit.io import write, read

write(df, "output.csv")        # CSV
write(df, "output.parquet")    # Parquet   (pip install syndatakit[io])
write(df, "output.dta")        # Stata .dta
write(df, "output.feather")    # Apache Arrow
write(df, "output.sas7bdat")   # SAS
write(df, "output.xlsx")       # Excel
write(df, "output.json")       # JSON

df = read("my_data.dta")       # reads Stata, SAS, CSV, Parquet, Arrow, JSON, Excel
```

---

## Why different from SDV / ydata

| Feature | syndatakit | SDV / ydata | Naive random |
|---|---|---|---|
| Cointegration preserved | Yes — Johansen test | No | No |
| GARCH volatility clustering | Yes — GARCH(1,1) | No | No |
| Downstream OLS comparison | Yes — coefficient match | No | No |
| 4-test privacy audit | Yes — ENISA framework | Partial | No |
| Stata .dta output | Yes | No | No |
| Scenario stress testing | Yes — 5 scenarios + intensity | No | No |
| 18 ready-to-use profiles | Yes | No | No |
| Formal (ε,δ)-DP guarantee | Yes | Partial | No |
| Open source | Yes — MIT | Partial | Yes |

---

## Architecture

```
syndatakit/
├── generators/
│   ├── base.py                  BaseGenerator — fit(), sample(), fit_sample()
│   ├── cross_sectional/         GaussianCopulaGenerator, DPGaussianCopulaGenerator
│   ├── time_series/             VARGenerator (VECM + GARCH)
│   ├── vine/                    VineCopulaGenerator
│   ├── panel/                   FixedEffectsGenerator
│   └── deep/                    CTGANGenerator
├── fidelity/
│   ├── marginal.py              KS + TVD per-column scores
│   ├── joint.py                 Spearman-Frobenius correlation distance
│   ├── temporal/                ADF, Engle-Granger, Chow, Granger causality
│   ├── stylized_facts.py        Fat tails, ARCH, autocorrelation
│   ├── downstream.py            TSTR train/test evaluation
│   └── report.py                Unified six-dimension report assembler
├── calibration/
│   ├── moment_matching.py       Post-hoc moment calibration
│   ├── priors.py                Bayesian priors — stable on datasets as small as 50 rows
│   └── scenario.py              Economic scenario calibration (5 scenarios, 0–1 intensity)
├── privacy/
│   ├── dp.py                    Laplace + Wishart mechanisms, budget tracker
│   ├── disclosure.py            Membership inference attack
│   ├── singling_out.py          Quasi-identifier singling-out attack
│   ├── linkability.py           Nearest-neighbour linkability attack
│   └── audit.py                 Full ENISA four-test privacy audit
├── catalog/
│   └── loader.py                18 built-in dataset profiles
└── io/
    ├── formats.py               CSV, Parquet, Arrow, Stata, SAS, Excel, JSON
    └── validators.py            Schema validation before fitting
```

---

## Contributing

**Adding a dataset profile** — implement `_build_<id>()` in `catalog/loader.py` using published aggregate statistics (not individual records), add a `PriorSet` in `calibration/priors.py`, and add tests. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full walkthrough.

**Adding a generator** — subclass `BaseGenerator`, implement `fit()` and `sample()`, export from the relevant `__init__.py`.

**Validation** — all fidelity scores in the table above are fully reproducible: `python validation/fidelity_engine.py`. Full reproduction code is in [`validation/`](validation/).

---

## Citation

syndatakit is currently under review at the Journal of Open Source Software (JOSS). A citable DOI will be available on acceptance. In the meantime:

```
Hapani, N. (2026). syndatakit: Research-grade synthetic data for finance and econometrics.
GitHub: https://github.com/Nityahapani/syndatakit
```

---

## License

MIT License — free to use for any purpose including commercial. See [LICENSE](LICENSE).

---

## Links

- **Website:** [https://syndatakitweb.github.io](https://syndatakitweb.github.io)
- **PyPI:** [pypi.org/project/syndatakit](https://pypi.org/project/syndatakit)
- **Docs:** [README](README.md) · [CONTRIBUTING](CONTRIBUTING.md) · [CHANGELOG](CHANGELOG.md)
- **Issues:** [github.com/Nityahapani/syndatakit/issues](https://github.com/Nityahapani/syndatakit/issues)
