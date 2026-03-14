"""
syndatakit.calibration.scenario
---------------------------------
Scenario-based calibration: generate synthetic data conditioned on
user-defined economic scenarios (recession, rate shock, credit crisis, etc.)

Scenarios work by shifting the synthetic distribution toward target parameter
values while preserving the correlation structure from the fitted generator.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Built-in scenarios ────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "recession": {
        "description": "Mild recession: negative GDP growth, rising unemployment",
        "shifts": {
            "gdp_growth_yoy":    {"target_mean": -2.5,  "target_std": 1.2},
            "unemployment_rate": {"target_mean":  8.5,  "target_std": 1.5},
            "vix":               {"target_mean": 32.0,  "target_std": 8.0},
            "yield_curve_spread":{"target_mean": -0.3,  "target_std": 0.4},
            "housing_starts":    {"target_mean": 750.0, "target_std": 150.0},
            "npl_ratio":         {"target_mean":  4.5,  "target_std": 1.2},
            "default_12m":       {"target_rate":  0.12},
        },
    },
    "severe_recession": {
        "description": "Severe recession: GFC-style contraction",
        "shifts": {
            "gdp_growth_yoy":    {"target_mean": -5.0,  "target_std": 1.5},
            "unemployment_rate": {"target_mean": 12.0,  "target_std": 2.0},
            "vix":               {"target_mean": 55.0,  "target_std": 12.0},
            "yield_curve_spread":{"target_mean": -0.8,  "target_std": 0.5},
            "npl_ratio":         {"target_mean":  8.5,  "target_std": 2.0},
            "ebitda_margin":     {"target_mean":  8.0,  "target_std": 6.0},
        },
    },
    "rate_shock": {
        "description": "Rapid rate hike cycle (2022-style tightening)",
        "shifts": {
            "fed_funds_rate": {"target_mean": 5.0,  "target_std": 0.4},
            "t10y_rate":      {"target_mean": 4.5,  "target_std": 0.6},
            "t2y_rate":       {"target_mean": 4.8,  "target_std": 0.5},
            "yield_curve_spread": {"target_mean": -0.3, "target_std": 0.3},
            "cpi_yoy":        {"target_mean": 7.5,  "target_std": 1.5},
            "housing_starts": {"target_mean": 950.0,"target_std": 150.0},
            "loan_amount":    {"scale_factor": 0.85},
        },
    },
    "credit_crisis": {
        "description": "Credit market stress: spreads widen, defaults surge",
        "shifts": {
            "npl_ratio":         {"target_mean": 7.0,  "target_std": 2.5},
            "tier1_capital_ratio":{"target_mean": 9.5, "target_std": 2.0},
            "default_12m":       {"target_rate": 0.18},
            "net_interest_margin":{"target_mean": 2.2, "target_std": 0.5},
            "roa":               {"target_mean": 0.2,  "target_std": 0.8},
        },
    },
    "expansion": {
        "description": "Strong expansion: above-trend growth, tight labour market",
        "shifts": {
            "gdp_growth_yoy":    {"target_mean": 4.0,  "target_std": 0.8},
            "unemployment_rate": {"target_mean": 3.5,  "target_std": 0.4},
            "vix":               {"target_mean": 14.0, "target_std": 3.0},
            "ebitda_margin":     {"target_mean": 24.0, "target_std": 6.0},
            "avg_weekly_wage":   {"scale_factor": 1.08},
        },
    },
}


def apply_scenario(
    synthetic: pd.DataFrame,
    scenario: str | dict,
    intensity: float = 1.0,
) -> pd.DataFrame:
    """
    Shift synthetic data toward a named scenario.

    Parameters
    ----------
    synthetic : generated synthetic DataFrame
    scenario  : name of a built-in scenario OR a custom dict of column shifts
    intensity : 0.0 = no shift, 1.0 = full shift, 0.5 = halfway

    Returns
    -------
    Scenario-conditioned DataFrame (copy).
    """
    if isinstance(scenario, str):
        if scenario not in SCENARIOS:
            available = ", ".join(SCENARIOS)
            raise ValueError(f"Unknown scenario '{scenario}'. Available: {available}")
        shifts = SCENARIOS[scenario]["shifts"]
    else:
        shifts = scenario

    syn = synthetic.copy()
    intensity = float(np.clip(intensity, 0.0, 1.0))

    for col, spec in shifts.items():
        if col not in syn.columns:
            continue
        arr = syn[col].astype(float).values.copy()

        if "target_mean" in spec:
            orig_mean = arr.mean()
            orig_std  = arr.std() + 1e-9
            new_mean  = spec["target_mean"]
            new_std   = spec.get("target_std", orig_std)
            # Blend: shift mean and std toward targets
            blend_mean = orig_mean + intensity * (new_mean - orig_mean)
            blend_std  = orig_std  + intensity * (new_std  - orig_std)
            arr = (arr - orig_mean) / orig_std * blend_std + blend_mean

        elif "scale_factor" in spec:
            factor = 1.0 + intensity * (spec["scale_factor"] - 1.0)
            arr = arr * factor

        elif "target_rate" in spec:
            # Binary column: resample to hit target rate
            target = spec["target_rate"]
            current_rate = arr.mean()
            if current_rate > 0 and current_rate < 1:
                blend_rate = current_rate + intensity * (target - current_rate)
                threshold  = np.percentile(arr, (1 - blend_rate) * 100)
                arr = (arr >= threshold).astype(float)

        syn[col] = arr

    return syn


def list_scenarios() -> pd.DataFrame:
    """Return a summary DataFrame of built-in scenarios."""
    rows = [
        {"name": k, "description": v["description"], "columns_affected": len(v["shifts"])}
        for k, v in SCENARIOS.items()
    ]
    return pd.DataFrame(rows)
