"""
syndatakit.fidelity.temporal.cointegration
-------------------------------------------
Engle-Granger cointegration test between column pairs.
Checks whether synthetic series preserve long-run equilibrium relationships
present in the real data (e.g. fed_funds_rate ↔ t10y_rate).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _eg_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Simplified Engle-Granger cointegration p-value via ADF on residuals."""
    if len(x) < 15 or len(y) < 15:
        return 0.5
    # Step 1: regress y on x
    X = np.column_stack([x, np.ones(len(x))])
    try:
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
    except Exception:
        return 0.5

    # Step 2: ADF on residuals
    diff_r = np.diff(resid)
    lag_r  = resid[:-1]
    Xr     = np.column_stack([lag_r, np.ones(len(lag_r))])
    try:
        br, _, _, _ = np.linalg.lstsq(Xr, diff_r, rcond=None)
        rho   = br[0]
        sigma = np.sqrt(np.sum((diff_r - Xr @ br) ** 2) / max(len(diff_r) - 2, 1))
        se    = sigma / (np.std(lag_r) * np.sqrt(len(lag_r)) + 1e-12)
        t     = rho / (se + 1e-12)
        # Critical value ~-3.34 @ 5% for residuals
        p = float(np.clip(0.5 + 0.5 * np.tanh((t + 3.34) / 0.9), 0.01, 0.99))
    except Exception:
        p = 0.5
    return p


def cointegration_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Test cointegration for column pairs.
    Score = fraction of pairs where real and synthetic agree on cointegration.
    """
    num_cols = [c for c in real.columns
                if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])]

    if pairs is None:
        # Auto-select up to 10 pairs from numeric columns
        pairs = []
        for i, ca in enumerate(num_cols[:6]):
            for cb in num_cols[i + 1:6]:
                pairs.append((ca, cb))

    results = {}
    agreements = 0

    for ca, cb in pairs:
        if ca not in real.columns or cb not in real.columns:
            continue
        r_x = real[ca].dropna().astype(float).values
        r_y = real[cb].dropna().astype(float).values
        s_x = synthetic[ca].dropna().astype(float).values
        s_y = synthetic[cb].dropna().astype(float).values
        n   = min(len(r_x), len(r_y), len(s_x), len(s_y))
        if n < 15:
            continue

        r_p = _eg_pvalue(r_x[:n], r_y[:n])
        s_p = _eg_pvalue(s_x[:n], s_y[:n])
        r_coint = r_p < alpha
        s_coint = s_p < alpha
        agree   = r_coint == s_coint
        if agree:
            agreements += 1

        results[f"{ca} ↔ {cb}"] = {
            "real_pvalue":      round(r_p, 4),
            "synthetic_pvalue": round(s_p, 4),
            "real_cointegrated":      r_coint,
            "synthetic_cointegrated": s_coint,
            "agreement": agree,
        }

    n_pairs = max(len(results), 1)
    results["_summary"] = {
        "agreement_rate": round(agreements / n_pairs * 100, 1),
        "pairs_tested":   len(results) - 1,
    }
    return results
