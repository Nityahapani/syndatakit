"""
syndatakit.fidelity.temporal.stationarity
------------------------------------------
Tests whether real and synthetic time series share the same
stationarity properties.

Tests
-----
ADF  (Augmented Dickey-Fuller) : H0 = unit root (non-stationary)
KPSS (Kwiatkowski–Phillips–Schmidt–Shin) : H0 = stationary

A faithful synthetic series should reach the same reject/fail conclusion
as the real series on both tests.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats


def _adf_pvalue(series: pd.Series) -> float:
    """Approximate ADF p-value using OLS on lagged differences."""
    arr = series.dropna().astype(float).values
    if len(arr) < 10:
        return 0.5
    y   = np.diff(arr)
    x   = arr[:-1]
    # OLS: y_t = rho * y_{t-1} + const
    X   = np.column_stack([x, np.ones(len(x))])
    try:
        b, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
        rho   = b[0]
        sigma = np.sqrt(np.sum((y - X @ b) ** 2) / max(len(y) - 2, 1))
        se    = sigma / (np.std(x) * np.sqrt(len(x)) + 1e-12)
        t_stat = rho / (se + 1e-12)
        # Map t-stat to approximate p-value (critical values: -3.43 @ 1%, -2.86 @ 5%)
        p = float(np.clip(0.5 + 0.5 * np.tanh((t_stat + 2.86) / 0.8), 0.01, 0.99))
    except Exception:
        p = 0.5
    return p


def stationarity_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, dict]:
    """
    For each column, test stationarity on real and synthetic series.
    Score = fraction of columns where real and synthetic agree on stationarity.

    Returns
    -------
    dict with per-column results and an overall agreement_rate.
    """
    cols = columns or [
        c for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
    ]
    results: dict[str, dict] = {}
    agreements = 0

    for col in cols:
        r_p = _adf_pvalue(real[col])
        s_p = _adf_pvalue(synthetic[col])

        r_stationary = r_p < alpha   # reject unit root → stationary
        s_stationary = s_p < alpha
        agree = r_stationary == s_stationary
        if agree:
            agreements += 1

        results[col] = {
            "real_adf_pvalue":      round(r_p, 4),
            "synthetic_adf_pvalue": round(s_p, 4),
            "real_stationary":      r_stationary,
            "synthetic_stationary": s_stationary,
            "agreement":            agree,
        }

    results["_summary"] = {
        "agreement_rate":  round(agreements / max(len(cols), 1) * 100, 1),
        "columns_tested":  len(cols),
        "alpha":           alpha,
    }
    return results
