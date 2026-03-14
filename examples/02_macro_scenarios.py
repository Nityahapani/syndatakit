"""
Example 2: Macro Scenario Analysis
------------------------------------
Generate baseline and stressed macroeconomic time series.
Compare fidelity across scenarios. Useful for stress testing,
regime-switching models, and recession-probability classifiers.

Run: python examples/02_macro_scenarios.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from syndatakit.generators.time_series import VARGenerator
from syndatakit.catalog import load_seed
from syndatakit.calibration import apply_scenario, list_scenarios
from syndatakit.fidelity import fidelity_report
from syndatakit.io import write


def main():
    print("=" * 60)
    print("  Macro Scenario Analysis — FRED Indicators")
    print("=" * 60)

    # ── 1. Fit VAR generator ──────────────────────────────────────────────────
    print("\n[1/4] Fitting VAR(2) generator on fred_macro...")
    seed = load_seed("fred_macro")
    gen = VARGenerator(lags=2, time_col="year")
    gen.fit(seed)
    print(f"      {gen}")
    print(f"      Numeric columns: {gen._numeric_cols[:5]}...")

    # ── 2. Generate baseline ──────────────────────────────────────────────────
    print("\n[2/4] Generating 500 baseline macro observations...")
    baseline = gen.sample(500, seed=42)
    print(f"      GDP growth mean      : {baseline['gdp_growth_yoy'].mean():.2f}%")
    print(f"      Unemployment mean    : {baseline['unemployment_rate'].mean():.1f}%")
    print(f"      Fed Funds Rate mean  : {baseline['fed_funds_rate'].mean():.2f}%")
    print(f"      VIX mean             : {baseline['vix'].mean():.1f}")

    # ── 3. Apply all scenarios ────────────────────────────────────────────────
    print("\n[3/4] Applying all scenarios at intensity=1.0...")
    scenarios_df = list_scenarios()

    results = {}
    for _, row in scenarios_df.iterrows():
        name = row["name"]
        stressed = apply_scenario(baseline, name, intensity=1.0)
        results[name] = stressed

        gdp = stressed["gdp_growth_yoy"].mean() if "gdp_growth_yoy" in stressed.columns else float("nan")
        unemp = stressed["unemployment_rate"].mean() if "unemployment_rate" in stressed.columns else float("nan")
        vix = stressed["vix"].mean() if "vix" in stressed.columns else float("nan")
        print(f"\n  [{name}]")
        print(f"    GDP growth:    {gdp:.2f}%")
        print(f"    Unemployment:  {unemp:.1f}%")
        print(f"    VIX:           {vix:.1f}")

    # ── 4. Fidelity report on baseline ───────────────────────────────────────
    print("\n[4/4] Fidelity report on baseline vs seed...")
    syn_body = baseline.drop(columns=["syn_id"])
    report = fidelity_report(seed, syn_body, dataset_type="time_series")
    s = report["summary"]
    t = report["temporal"]
    print(f"      Overall fidelity       : {s['overall_fidelity']}%")
    print(f"      Stationarity agreement : {t['stationarity']['_summary']['agreement_rate']}%")
    print(f"      Cointegration agreement: {t['cointegration']['_summary']['agreement_rate']}%")
    print(f"      Causality agreement    : {t['causality']['_summary']['agreement_rate']}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    write(baseline, "examples/output_macro_baseline.csv")
    for name, df in results.items():
        write(df, f"examples/output_macro_{name}.csv")
    print(f"\n  Saved baseline + {len(results)} scenario files to examples/")
    print("\nDone.")


if __name__ == "__main__":
    main()
