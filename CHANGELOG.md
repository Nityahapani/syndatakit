# Changelog

All notable changes to syndatakit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — 2026-03-14

Complete architectural rewrite. syndatakit v2 is a research-grade synthetic
data library for finance and econometrics, built around a modular plugin
architecture with formal fidelity reporting, privacy auditing, and calibration.

### Added — Generators
- `generators/base.py` — `BaseGenerator` abstract base class with `fit()`, `sample()`, `fit_sample()` interface
- `generators/cross_sectional/gaussian_copula.py` — Gaussian Copula generator with Cholesky correlation sampling, prior regularisation support
- `generators/time_series/var.py` — VAR(p) generator with bootstrapped initial conditions for temporal datasets
- `generators/panel/fixed_effects.py` — Fixed-effects panel generator with entity + time decomposition
- `generators/deep/ctgan.py` — CTGAN stub with clean upgrade path (`pip install syndatakit[deep]`)

### Added — Fidelity
- `fidelity/marginal.py` — KS test (numeric), TVD (categorical) per-column scores
- `fidelity/joint.py` — Spearman correlation matrix distance, pairwise delta report
- `fidelity/temporal/stationarity.py` — ADF-based stationarity agreement test
- `fidelity/temporal/cointegration.py` — Engle-Granger cointegration preservation
- `fidelity/temporal/breaks.py` — Chow-scan structural break matching
- `fidelity/causality.py` — Granger causality direction preservation
- `fidelity/stylized_facts.py` — Fat tails, skewness sign, autocorrelation, ARCH effects
- `fidelity/downstream.py` — TSTR (train-on-synthetic, test-on-real) utility evaluation
- `fidelity/report.py` — Unified fidelity report assembler across all metrics

### Added — Privacy
- `privacy/dp.py` — Laplace and Gaussian mechanisms, `PrivacyBudget` tracker with consumption log
- `privacy/disclosure.py` — Membership inference attack (shadow model, AUC-based)
- `privacy/singling_out.py` — Quasi-identifier subset singling-out attack
- `privacy/linkability.py` — Nearest-neighbour cross-dataset linkability attack
- `privacy/audit.py` — Full privacy audit: runs all four tests, returns unified verdict + recommendation

### Added — Calibration
- `calibration/moment_matching.py` — Post-hoc mean/std/skew moment calibration
- `calibration/priors.py` — `Prior`, `PriorSet`, MAP blending, built-in priors for all 10 datasets
- `calibration/scenario.py` — 5 built-in economic scenarios: `recession`, `severe_recession`, `rate_shock`, `credit_crisis`, `expansion`

### Added — Catalog
- `catalog/loader.py` — 10 finance/econometrics datasets across 4 verticals
  - Credit & Lending: `hmda`, `fdic`, `credit_risk`
  - Capital Markets: `edgar`, `cftc`
  - Macro & Central Bank: `fred_macro`, `bls`, `world_bank`
  - Tax & Income: `irs_soi`, `census_acs`

### Added — IO
- `io/formats.py` — Read/write CSV, Parquet, Arrow, JSON, Stata `.dta`, SAS `.sas7bdat`, Excel
- `io/validators.py` — Schema validation: null thresholds, cardinality, constant columns, duplicates

### Added — CLI (v2)
- `syndatakit list [--vertical]` — grouped by vertical with fidelity scores
- `syndatakit info <dataset>` — full metadata, columns, use cases
- `syndatakit generate <dataset>` — auto generator routing, `--scenario`, `--calibrate`, `--filter`, `--generator`
- `syndatakit evaluate <real> <syn>` — full fidelity report with temporal and TSTR support
- `syndatakit audit <real> <syn>` — full privacy audit with coloured risk levels
- `syndatakit scenario list / apply` — browse and apply scenarios to any CSV
- `syndatakit validate <file>` — schema validation before fitting

### Added — API (v2)
- 10 REST endpoints: `/health`, `/datasets`, `/datasets/{id}`, `/datasets/{id}/sample`,
  `/generate`, `/evaluate`, `/audit`, `/scenarios`, `/scenario/apply`, `/validate`
- Interactive docs page at `/docs`

### Added — Tests
- 130 tests across 17 test classes covering every module

---

## [0.1.0] — 2026-01-15

Initial release.

### Added
- Gaussian Copula generator for cross-sectional tabular data
- HMDA mortgage dataset (seed from CFPB 2022 statistics)
- Basic CLI: `list`, `info`, `generate`, `evaluate`, `serve`
- Flask REST API with `/generate` and `/evaluate` endpoints
- KS + TVD fidelity evaluation
- Exact copy privacy check
