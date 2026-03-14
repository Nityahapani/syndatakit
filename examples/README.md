# Examples

Four self-contained scripts showing real syndatakit workflows.

## Setup

```bash
pip install syndatakit
# or from source:
pip install -e ".[dev]"
```

## Run

```bash
# Credit risk PD model training
python examples/01_credit_risk.py

# Macro scenario analysis (baseline + recession + rate shock + ...)
python examples/02_macro_scenarios.py

# Full privacy audit walkthrough
python examples/03_privacy_audit.py

# REST API usage (start server first)
syndatakit serve --port 8080
python examples/04_api_usage.py
```

## What each example covers

| File | Dataset | Topics |
|---|---|---|
| `01_credit_risk.py` | `credit_risk` | Generator fitting, priors, TSTR downstream evaluation, privacy audit, scenario (credit crisis) |
| `02_macro_scenarios.py` | `fred_macro` | VAR generator, all 5 scenarios, temporal fidelity report |
| `03_privacy_audit.py` | `hmda` | Full privacy audit, membership inference, singling-out, linkability, differential privacy budget |
| `04_api_usage.py` | multiple | All 10 REST API endpoints with real requests |
