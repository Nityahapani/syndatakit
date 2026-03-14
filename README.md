# syndatakit

**Research-grade synthetic data for finance & econometrics.**

```bash
pip install syndatakit
syndatakit generate fred_macro --rows 10000 --scenario recession --output training_data.csv
```

10,000 synthetic macroeconomic observations, statistically learned from public Federal Reserve data,
conditioned on a recession scenario. Ready to train your model on in under 5 seconds.

---

## Why syndatakit

Training ML models in finance and economics requires realistic data. But real data is locked
behind NDAs, data use agreements, and privacy regulations.

syndatakit learns the statistical structure of public government datasets — distributions,
correlations, temporal dynamics, stylized facts — and generates unlimited synthetic records
that match that structure without containing any real individuals.

**No data use agreements. No PII. No legal review.**

---

## Install

```bash
pip install syndatakit          # core: Copula + VAR + Panel generators
pip install syndatakit[api]     # + Flask REST API server
pip install syndatakit[io]      # + Parquet, Arrow, Stata, SAS, Excel formats
pip install syndatakit[deep]    # + CTGAN deep generator (requires PyTorch)
pip install syndatakit[all]     # everything
```

Requires Python 3.9+. Core dependencies: `pandas`, `numpy`, `scipy`.

---

## Quick start

```bash
syndatakit list
syndatakit generate hmda --rows 5000 --output mortgages.csv
syndatakit generate hmda --rows 1000 --filter state:CA,TX --filter dti_min:45 --output highrisk.csv
syndatakit generate fred_macro --rows 2000 --scenario recession --intensity 0.8 --output recession.csv
syndatakit evaluate real.csv synthetic.csv --type time_series --target gdp_growth_yoy
syndatakit audit real.csv synthetic.csv --attacks 500
syndatakit validate my_data.csv
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

# Cross-sectional
gen = GaussianCopulaGenerator()
gen.fit(load_seed("hmda"))
df = gen.sample(10_000)
df_highrisk = gen.sample(1000, filters={"state": ["CA","TX"], "dti_min": 45})

# Time series
gen_ts = VARGenerator(lags=2, time_col="year")
gen_ts.fit(load_seed("fred_macro"))
df_macro = gen_ts.sample(500)
df_recession = apply_scenario(df_macro, "recession", intensity=0.9)

# Bayesian priors for small datasets
from syndatakit.calibration.priors import get_priors
gen_p = GaussianCopulaGenerator(priors=get_priors("hmda"))
gen_p.fit(load_seed("hmda").sample(50))

# Full fidelity report
report = fidelity_report(load_seed("fred_macro"), df_macro, dataset_type="time_series")
print(f"Overall fidelity: {report['summary']['overall_fidelity']}%")

# Privacy audit
audit = privacy_audit(load_seed("hmda"), df, n_attacks=500)
print(f"Privacy risk: {audit['verdict']['overall_risk']}")
```

---

## Datasets

| ID | Name | Vertical | Cols | Source | Fidelity |
|---|---|---|---|---|---|
| `hmda` | HMDA Mortgage Applications | Credit & Lending | 7 | CFPB 2022 | 98% |
| `fdic` | FDIC Bank Call Reports | Credit & Lending | 12 | FDIC SDI 2023 | 97% |
| `credit_risk` | Consumer Credit Risk | Credit & Lending | 10 | CFPB derived | 96% |
| `edgar` | SEC EDGAR Financials | Capital Markets | 13 | SEC XBRL 2023 | 97% |
| `cftc` | CFTC Commitments of Traders | Capital Markets | 10 | CFTC COT 2023 | 98% |
| `fred_macro` | FRED Macro Indicators | Macro & Central Bank | 15 | Federal Reserve | 97% |
| `bls` | BLS Employment & Wages | Macro & Central Bank | 9 | BLS QCEW 2022 | 97% |
| `world_bank` | World Bank WDI | Macro & Central Bank | 12 | WDI 2022 | 96% |
| `irs_soi` | IRS Statistics of Income | Tax & Income | 11 | IRS SOI 2021 | 95% |
| `census_acs` | Census ACS Income & Housing | Tax & Income | 11 | ACS 2022 | 96% |

---

## Generators

| Generator | Best for |
|---|---|
| `GaussianCopulaGenerator` | Cross-sectional tabular (hmda, edgar, credit_risk, ...) |
| `VARGenerator` | Multivariate time series (fred_macro, bls) |
| `FixedEffectsGenerator` | Panel data entity x time (world_bank, fdic) |
| `CTGANGenerator` | Complex nonlinear relationships (`pip install syndatakit[deep]`) |

---

## Fidelity

Every dataset is evaluated on six dimensions: marginal distributions (KS/TVD),
joint correlation structure, temporal properties (stationarity, cointegration,
structural breaks, Granger causality), stylized facts (fat tails, ARCH effects),
downstream utility (TSTR), and privacy (membership inference, singling-out, linkability).

---

## Architecture

```
syndatakit/
├── generators/           base.py + cross_sectional/ + time_series/ + panel/ + deep/
├── fidelity/             marginal, joint, temporal/, causality, stylized_facts, downstream, report
├── calibration/          moment_matching, priors, scenario
├── privacy/              dp, disclosure, singling_out, linkability, audit
├── catalog/              loader.py — 10 datasets, 4 verticals
└── io/                   formats (CSV/Parquet/Arrow/Stata/SAS/Excel), validators
```

---

## Contributing

Add a dataset: implement `_build_<id>()` in `catalog/loader.py` + add `PriorSet` in `calibration/priors.py`.

Add a generator: subclass `BaseGenerator`, implement `fit()` and `sample()`.

---

## License

Business Source License 1.1 — free for self-hosting, commercial agreement required for managed service.
