# syndatakit

**Research-grade synthetic data generator for finance & econometrics.**

[![PyPI version](https://img.shields.io/pypi/v/syndatakit)](https://pypi.org/project/syndatakit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](https://pypi.org/project/syndatakit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-168%20passing-brightgreen)](https://github.com/Nityahapani/syndatakit/actions)

```
pip install syndatakit
syndatakit generate fred_macro --rows 1000000 --scenario recession --output training_data.csv
```

One command. One million synthetic macroeconomic observations under a recession scenario. Statistically realistic. No real individuals. No data use agreement.

---

## What syndatakit is

syndatakit is a **synthetic data generator** — not a collection of stored datasets.

When you run `syndatakit generate hmda --rows 1000000`, it creates one million brand new synthetic mortgage application records that never existed. They are statistically similar to real HMDA data but contain zero real people. The package does not store or ship those rows — it generates them on demand, every time, in seconds.

The package ships with **18 built-in dataset profiles** across 8 financial and economic verticals — statistical models learned from published government aggregate statistics (CFPB reports, Federal Reserve bulletins, BLS data releases, etc.). Each profile captures the distributions, correlations, and structure of a real public dataset without containing any actual records from it.

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

---

## Why syndatakit

Training ML models in finance requires realistic data. But real data is locked behind NDAs, data use agreements, and privacy regulations.

syndatakit learns the statistical structure of financial datasets — distributions, correlations, temporal dynamics, stylized facts — and generates unlimited synthetic records that preserve that structure without exposing any real individuals.

**No data use agreements. No PII. No legal review.**

---

## Install

```
pip install syndatakit                     # core — all 18 datasets, 4 generators, CLI, REST API
pip install syndatakit[vine]               # + Vine Copula with asymmetric tail dependence
pip install syndatakit[timeseries]         # + VECM + GARCH time series generator
pip install syndatakit[deep]               # + CTGAN deep tabular generator (requires PyTorch)
pip install syndatakit[io]                 # + Parquet, Arrow, Stata .dta, SAS, Excel formats
```

**Requirements:** Python ≥ 3.9 · pandas ≥ 1.5 · numpy ≥ 1.23 · scipy ≥ 1.9 · No GPU required.

---

## Quick start

```bash
# List all 18 datasets grouped by vertical
syndatakit list

# Generate 1M HMDA mortgage rows with filters
syndatakit generate hmda \
  --rows 1000000 \
  --filter state:CA,TX \
  --filter dti_min:45 \
  --output mortgages.csv

# Macro data under a rate shock
syndatakit generate fred_macro \
  --scenario rate_shock \
  --intensity 0.8

# Generate 10,000 equity returns under recession
syndatakit generate equity_returns \
  --rows 10000 \
  --scenario recession \
  --output returns.csv

# Full fidelity report — how well does the synthetic match the real distribution?
syndatakit evaluate real.csv synthetic.csv --type time_series

# Full privacy audit with 500 attacks per test
syndatakit audit real.csv synthetic.csv --attacks 500

# Use your own file
syndatakit generate --input my_data.csv --rows 1000000

# Start REST API
syndatakit serve --port 8080
```

---

## Python API

```python
from syndatakit.generators import GaussianCopulaGenerator
from syndatakit.generators.time_series import VARGenerator
from syndatakit.generators.cross_sectional import DPGaussianCopulaGenerator
from syndatakit.catalog import load_seed
from syndatakit.fidelity import fidelity_report
from syndatakit.privacy import privacy_audit, format_audit
from syndatakit.privacy.dp import PrivacyBudget, laplace_mechanism
from syndatakit.calibration import apply_scenario, match_moments
from syndatakit.calibration.priors import get_priors

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

# ── Scenario calibration ──────────────────────────────────────────────────────

df_recession     = apply_scenario(df_macro, "recession",       intensity=0.9)
df_severe        = apply_scenario(df_macro, "severe_recession", intensity=1.0)
df_rate_shock    = apply_scenario(df_macro, "rate_shock",       intensity=1.0)
df_credit_crisis = apply_scenario(df_macro, "credit_crisis",    intensity=0.8)
df_expansion     = apply_scenario(df_macro, "expansion",        intensity=0.5)

# ── Fidelity report ───────────────────────────────────────────────────────────

real = load_seed("fred_macro")
report = fidelity_report(real, df_macro, dataset_type="time_series")
print(f"Overall fidelity: {report['summary']['overall_fidelity']}%")
print(f"Stationarity:     {report['temporal']['stationarity']['_summary']['agreement_rate']}%")
print(f"Cointegration:    {report['temporal']['cointegration']['_summary']['agreement_rate']}%")

# ── Differential privacy ──────────────────────────────────────────────────────

# Formal (ε, δ)-DP generator — ε=1.0 balances utility and privacy
gen_dp = DPGaussianCopulaGenerator(epsilon=1.0, delta=1e-5)
gen_dp.fit(real)
syn = gen_dp.sample(10_000)

# Budget is tracked — never accidentally exceed it
print(f"ε used: {gen_dp.epsilon_used:.3f} / {gen_dp.epsilon}")

# Explicit budget for aggregate statistics
budget = PrivacyBudget(epsilon=2.0)
noisy_mean = laplace_mechanism(
    real["loan_amount"].mean(),
    sensitivity=1_000_000,
    epsilon=0.5,
    budget=budget,
)

# ── Privacy audit ─────────────────────────────────────────────────────────────

audit = privacy_audit(real, syn, n_attacks=500)
print(format_audit(audit))

# ── Bayesian priors (stabilises generators on small datasets < 500 rows) ──────

priors = get_priors("hmda")
gen = GaussianCopulaGenerator(priors=priors)
gen.fit(small_df)   # works well even with 50 rows

# ── Post-hoc moment calibration ───────────────────────────────────────────────

calibrated = match_moments(real, synthetic)
```

---

## Built-in dataset profiles

18 profiles across 8 verticals. Each was built from published government aggregate statistics — not individual records.

### Credit & Lending

| ID | Name | Columns | Source | Fidelity |
|---|---|---|---|---|
| `hmda` | HMDA Mortgage Applications | 7 | CFPB 2022 | 98% |
| `fdic` | FDIC Bank Call Reports | 12 | FDIC SDI 2023 | 97% |
| `credit_risk` | Consumer Credit Risk | 10 | CFPB derived | 96% |

### Capital Markets

| ID | Name | Columns | Source | Fidelity |
|---|---|---|---|---|
| `edgar` | SEC EDGAR Financial Statements | 13 | SEC XBRL 2023 | 97% |
| `cftc` | CFTC Commitments of Traders | 10 | CFTC COT 2023 | 98% |
| `equity_returns` | Equity Returns & Risk Factors | 15 | CRSP/Compustat | 96% |
| `corporate_bonds` | Corporate Bond Market Data | 15 | TRACE/FINRA | 96% |

### Macro & Central Bank

| ID | Name | Columns | Source | Fidelity |
|---|---|---|---|---|
| `fred_macro` | FRED Macroeconomic Indicators | 15 | Federal Reserve | 97% |
| `bls` | BLS Employment & Wages | 9 | BLS QCEW 2022 | 97% |
| `world_bank` | World Bank Dev. Indicators | 12 | WDI 2022 | 96% |
| `central_bank_rates` | Central Bank Policy Rates | 11 | BIS/central banks | 97% |

### Tax & Income

| ID | Name | Columns | Source | Fidelity |
|---|---|---|---|---|
| `irs_soi` | IRS Statistics of Income | 11 | IRS SOI 2021 | 95% |
| `census_acs` | Census ACS Income & Housing | 11 | ACS 5-Year 2022 | 96% |

### Insurance

| ID | Name | Columns | Source | Fidelity |
|---|---|---|---|---|
| `insurance_claims` | P&C Insurance Claims | 13 | NAIC Schedule P | 95% |
| `life_insurance` | Life Insurance & Mortality | 13 | SOA/LIMRA | 96% |

### Real Estate

| ID | Name | Columns | Source | Fidelity |
|---|---|---|---|---|
| `commercial_real_estate` | Commercial Real Estate | 15 | NCREIF/CoStar | 95% |
| `rental_market` | Residential Rental Market | 13 | HUD FMR/Zillow | 96% |

### Retail Banking

| ID | Name | Columns | Source | Fidelity |
|---|---|---|---|---|
| `retail_transactions` | Retail Banking Transactions | 12 | Fed Payment Study | 96% |

### Commodities

| ID | Name | Columns | Source | Fidelity |
|---|---|---|---|---|
| `commodity_prices` | Commodity Price Returns | 13 | EIA/USDA/LME | 96% |

You are not limited to these 18. Any tabular dataset can be used with `gen.fit(your_df)`.

---

## Generators

| Generator | Best for | Auto-selected for |
|---|---|---|
| `GaussianCopulaGenerator` | Cross-sectional tabular data | hmda, fdic, credit_risk, edgar, cftc, irs_soi, census_acs, insurance, real_estate |
| `VineGenerator` | High-dimensional data (10+ correlated variables) with asymmetric tail dependence | equity_returns, corporate_bonds, commodity_prices |
| `VARGenerator` | Multivariate time series | fred_macro, bls, central_bank_rates |
| `VECMGARCHGenerator` | Cointegrated series + volatility clustering | equity_returns, fred_macro |
| `FixedEffectsGenerator` | Panel data (entity × time) | world_bank, fdic |
| `DPGaussianCopulaGenerator` | Any tabular data with formal DP guarantees | any, when compliance requires (ε, δ)-DP |
| `CTGANGenerator` | Complex non-linear relationships | any (`pip install syndatakit[deep]`) |

The CLI auto-selects the right generator for each built-in profile. When using your own data, pick the one that matches your data type.

### Generator selection guide

**Cross-sectional data** (one row per entity, no time dimension) → `GaussianCopulaGenerator`. For 10+ correlated variables with asymmetric behaviour (e.g. assets that co-crash but don't co-spike), upgrade to `VineGenerator`.

**Macroeconomic time series** (GDP, inflation, unemployment, interest rates) → `VARGenerator`. If your series share a long-run equilibrium (cointegrated), use `VECMGARCHGenerator` instead — standard VAR will destroy the long-run relationship.

**Financial returns** (equities, bonds, commodities) → `VECMGARCHGenerator`. GARCH is non-negotiable for financial returns: the volatility clustering stylized fact (calm periods followed by volatile bursts) is immediately obvious to any financial economist when it is absent.

**Panel data** (firms or countries observed over time) → `FixedEffectsGenerator`.

**Compliance-critical generation** (GDPR, CCPA, institutional DP requirements) → `DPGaussianCopulaGenerator` with your required ε.

---

## Fidelity evaluation

Every generated dataset can be evaluated across six independent dimensions via `fidelity_report(real, synthetic)`.

### The six dimensions

**Marginal** — Kolmogorov-Smirnov test for numeric columns, Total Variation Distance for categorical. Per-column scores 0–100%.

**Joint** — Spearman correlation matrix distance (Frobenius norm). Detects when the generator preserves marginals correctly but breaks the relationships between variables — the most common failure mode.

**Temporal** — ADF stationarity, Engle-Granger cointegration, Chow structural break detection, and Granger causality direction. If real series are cointegrated, synthetic series must be too; any regression on mis-integrated synthetic data produces invalid inference.

**Stylized Facts** — Fat tails (excess kurtosis), skewness sign, autocorrelation decay, ARCH volatility clustering. Critical for financial return data. Absence of GARCH-type clustering is immediately visible to any financial economist.

**Downstream / TSTR** — Train on synthetic, test on real (TSTR). Compare to training on real data (TRR). A ratio near 1.0 means the synthetic data is a perfect substitute for the real data in a model training context. This is the econometrician's gold standard.

**Privacy** — Exact copy check, membership inference AUC, singling-out rate, linkability risk. Full verdict with concrete recommendation.

### Live fidelity scores — HMDA Mortgage Applications (v2.1.0)

| Column | Score |
|---|---|
| `loan_amount` | 97.0% |
| `applicant_income` | 98.2% |
| `action_taken` | 99.1% |
| `loan_purpose` | 98.8% |
| `property_type` | 98.7% |
| `debt_to_income` | 97.2% |
| `state` | 97.2% |

---

## Privacy & compliance

Four independent attacks run automatically. Each returns a risk level. The overall verdict is the maximum risk across all tests, with a concrete recommendation.

### The four tests

**Exact Copy Check** — Hashes every synthetic row, checks against real data. Count must be 0. Always passes on properly fitted generators.

**Membership Inference** — Shadow model attack. AUC near 0.5 = no advantage over random guessing. AUC above 0.7 = memorisation risk detected.

**Singling-Out** — Random quasi-identifier subset attack. Tests whether an adversary can uniquely identify one individual by matching on 2–4 attributes from the synthetic dataset.

**Linkability** — Nearest-neighbour cross-dataset attack. Tests whether an adversary can link a synthetic record back to a specific real individual. Baseline 0.5 (random).

These four tests map directly to the EU ENISA synthetic data assessment framework, making the output suitable for compliance documentation at European research institutions.

### Differential privacy

Formal (ε, δ)-DP guarantee via Laplace mechanism on marginals and Wishart mechanism on the correlation matrix. Budget consumed and reported per operation.

| ε | Privacy level | Typical use |
|---|---|---|
| 0.1 | Strong | Highly sensitive individual data |
| 1.0 | Balanced | Standard compliance requirement |
| 10.0 | High utility | Internal model training, low sensitivity |

```python
from syndatakit.generators.cross_sectional import DPGaussianCopulaGenerator

gen = DPGaussianCopulaGenerator(epsilon=1.0, delta=1e-5)
gen.fit(real_df)
syn = gen.sample(10_000)

print(f"ε used: {gen.epsilon_used:.3f} / {gen.epsilon}")
```

---

## Scenario calibration

Five built-in economic scenarios with configurable intensity (0.0–1.0). Apply to any generated dataset.

```python
from syndatakit.calibration import apply_scenario

df_recession     = apply_scenario(df, "recession",        intensity=0.9)
df_severe        = apply_scenario(df, "severe_recession",  intensity=1.0)
df_rate_shock    = apply_scenario(df, "rate_shock",        intensity=1.0)
df_credit_crisis = apply_scenario(df, "credit_crisis",     intensity=0.8)
df_expansion     = apply_scenario(df, "expansion",         intensity=0.5)
```

### Scenario parameter shifts (at intensity 1.0)

**Recession**

| Metric | Baseline | Scenario |
|---|---|---|
| GDP Growth | +2.3% | −2.5% |
| Unemployment | 5.5% | 8.5% |
| VIX | 18.0 | 32.0 |
| Yield Curve | +0.9% | −0.3% |
| Housing Starts | 1,250K | 750K |
| NPL Ratio | 1.2% | 4.5% |

**Severe Recession (GFC-style)**

| Metric | Baseline | Scenario |
|---|---|---|
| GDP Growth | +2.3% | −5.0% |
| Unemployment | 5.5% | 12.0% |
| VIX | 18.0 | 55.0 |
| EBITDA Margin | 18% | 8.0% |
| NPL Ratio | 1.2% | 8.5% |
| Tier 1 Capital | 13.2% | 9.5% |

**Rate Shock (2022-style tightening)**

| Metric | Baseline | Scenario |
|---|---|---|
| Fed Funds Rate | 1.5% | 5.0% |
| 10Y Treasury | 2.5% | 4.5% |
| CPI YoY | 2.5% | 7.5% |
| Yield Curve | +0.9% | −0.3% |
| Housing Starts | 1,250K | 950K |
| Loan Amount | $280K | $238K |

**Credit Crisis (spreads widen)**

| Metric | Baseline | Scenario |
|---|---|---|
| NPL Ratio | 1.2% | 7.0% |
| Tier 1 Capital | 13.2% | 9.5% |
| Default Rate | 3% | 18% |
| NIM | 3.1% | 2.2% |
| ROA | 1.05% | 0.2% |
| Credit Spread | 150bps | 450bps |

**Expansion (above-trend growth)**

| Metric | Baseline | Scenario |
|---|---|---|
| GDP Growth | +2.3% | +4.0% |
| Unemployment | 5.5% | 3.5% |
| VIX | 18.0 | 14.0 |
| EBITDA Margin | 18% | 24% |
| Weekly Wage | $1,168 | $1,261 |
| ROA | 1.05% | 1.8% |

---

## REST API

```bash
syndatakit serve --port 8080
# Interactive docs: http://localhost:8080/docs
```

```python
import requests, pandas as pd

# Generate equity returns under a recession scenario
r = requests.post("http://localhost:8080/generate", json={
    "dataset":   "equity_returns",
    "rows":      5000,
    "scenario":  "recession",
    "intensity": 0.9,
})
df = pd.DataFrame(r.json()["data"])

# Upload your own CSV and generate from it
r2 = requests.post("http://localhost:8080/generate/custom",
    files={"file": open("my_data.csv")},
    data={"rows": "100000"},
)

# Full privacy audit
r3 = requests.post("http://localhost:8080/audit",
    files={"real": open("real.csv"), "synthetic": open("syn.csv")},
    data={"attacks": "500"},
)
print(r3.json()["data"]["verdict"]["recommendation"])
```

All endpoints: `GET /datasets` · `GET /datasets/{id}` · `GET /datasets/{id}/sample` · `POST /generate` · `POST /generate/custom` · `POST /evaluate` · `POST /audit` · `GET /scenarios` · `POST /scenario/apply` · `POST /validate` · `GET /health`

---

## IO formats

```python
from syndatakit.io import write, read

write(df, "output.csv")        # CSV
write(df, "output.parquet")    # Parquet  (pip install syndatakit[io])
write(df, "output.dta")        # Stata    (pip install syndatakit[io])
write(df, "output.feather")    # Apache Arrow
write(df, "output.xlsx")       # Excel
write(df, "output.json")       # JSON

df = read("my_data.dta")       # reads Stata, SAS, CSV, Parquet, Arrow, JSON, Excel
```

Stata `.dta` support is included specifically because the majority of academic economics research runs in Stata. The `read()` and `write()` functions preserve variable labels, value labels, and metadata.

---

## Architecture

```
syndatakit/
├── generators/
│   ├── base.py                    BaseGenerator ABC — fit(), sample(), fit_sample()
│   ├── cross_sectional/
│   │   ├── copula.py              GaussianCopulaGenerator, DPGaussianCopulaGenerator
│   │   └── vine.py                VineGenerator — pair copulas for high-dimensional data
│   ├── time_series/
│   │   ├── var.py                 VARGenerator — multivariate time series
│   │   └── vecm_garch.py          VECMGARCHGenerator — cointegrated systems + volatility
│   ├── panel/
│   │   └── fixed_effects.py       FixedEffectsGenerator — entity × time panel data
│   └── deep/
│       └── ctgan.py               CTGANGenerator (pip install syndatakit[deep])
├── fidelity/
│   ├── marginal.py                KS + TVD per-column scores
│   ├── joint.py                   Correlation matrix distance (Frobenius norm)
│   ├── temporal/
│   │   ├── stationarity.py        ADF, KPSS, Phillips-Perron tests
│   │   ├── cointegration.py       Engle-Granger cointegration tests
│   │   └── breaks.py              Chow scan for structural breaks
│   ├── stylized_facts.py          Fat tails, ARCH, autocorrelation decay
│   ├── causality.py               Granger causality preservation checks
│   ├── downstream.py              TSTR / TRR train-on-synthetic evaluation
│   └── report.py                  Unified report assembler
├── calibration/
│   ├── moment_matching.py         Post-hoc moment calibration
│   ├── priors.py                  Bayesian priors + MAP blending for small datasets
│   └── scenario.py                5 built-in economic scenarios with configurable intensity
├── privacy/
│   ├── dp.py                      Laplace + Wishart mechanisms, PrivacyBudget tracker
│   ├── disclosure.py              Membership inference shadow model attack
│   ├── singling_out.py            Quasi-identifier singling-out attack
│   ├── linkability.py             Nearest-neighbour linkability attack
│   └── audit.py                   Full 4-test audit — runs all tests, produces verdict
├── catalog/
│   └── loader.py                  18 built-in dataset profiles + seed builders
└── io/
    ├── formats.py                 CSV, Parquet, Arrow, Stata, SAS, Excel, JSON
    └── validators.py              Schema validation before fitting
```

---

## Contributing

**Adding a new dataset profile** — implement `_build_<id>()` in `catalog/loader.py` using published aggregate statistics (not individual records), add a `PriorSet` in `calibration/priors.py`, and add tests. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full walkthrough.

**Adding a new generator** — subclass `BaseGenerator`, implement `fit()` and `sample()`, export from the relevant `__init__.py`. The interface is minimal by design.

---

## Citation

If you use syndatakit in academic work, please cite it:

```bibtex
@software{syndatakit,
  title  = {syndatakit: Research-grade synthetic data generator for finance \& econometrics},
  author = {Nityahapani},
  url    = {https://github.com/Nityahapani/syndatakit},
  year   = {2025},
  version = {2.1.0}
}
```

---

## License

Business Source License 1.1 (BSL-1.1). Free to use for any purpose including commercial, except offering syndatakit itself as a hosted managed service without a commercial agreement. Self-hosting always free. See [LICENSE](LICENSE).

---

## Cloud version

Need higher row counts, managed infrastructure, compliance documentation, or team access? A hosted cloud version is coming.
