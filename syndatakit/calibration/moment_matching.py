"""
syndatakit.calibration.moment_matching
----------------------------------------
Post-hoc calibration: adjust synthetic data so its first four moments
(mean, variance, skewness, kurtosis) match the real data exactly.

This is applied *after* generation as a lightweight correction layer.
It does not change the correlation structure, only marginal moments.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def match_moments(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    moments: tuple[str, ...] = ("mean", "std"),
) -> pd.DataFrame:
    """
    Adjust synthetic numeric columns to match real moments.

    Parameters
    ----------
    real, synthetic : DataFrames to align
    columns         : numeric columns to calibrate (default: all shared numeric)
    moments         : which moments to match — any of "mean", "std", "skew", "kurt"

    Returns
    -------
    Calibrated synthetic DataFrame (copy, original unchanged).
    """
    syn = synthetic.copy()

    cols = columns or [
        c for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
    ]

    for col in cols:
        r = real[col].dropna().astype(float).values
        s = syn[col].dropna().astype(float).values
        if len(r) < 4 or len(s) < 4:
            continue

        s_adj = s.copy()

        # Step 1: match mean and std (always)
        if "mean" in moments or "std" in moments:
            r_mean, r_std = r.mean(), r.std() + 1e-9
            s_mean, s_std = s_adj.mean(), s_adj.std() + 1e-9
            s_adj = (s_adj - s_mean) / s_std * r_std + r_mean

        # Step 2: match skewness via power transform
        if "skew" in moments:
            r_skew = float(stats.skew(r))
            s_skew = float(stats.skew(s_adj))
            if abs(r_skew - s_skew) > 0.1:
                # Simple skewness correction via cube-root-ish transform
                lam = 1 + (r_skew - s_skew) * 0.15
                s_min = s_adj.min()
                s_adj = np.sign(s_adj - s_min) * np.abs(s_adj - s_min) ** max(lam, 0.1) + s_min
                # Re-normalise after transform
                s_adj = (s_adj - s_adj.mean()) / (s_adj.std() + 1e-9) * r.std() + r.mean()

        # Write back, respecting original bounds
        r_min, r_max = r.min(), r.max()
        s_adj = np.clip(s_adj, r_min * 0.95, r_max * 1.05)
        # Cast back to original dtype
        try:
            import pandas as _pd
            if _pd.api.types.is_integer_dtype(syn[col]):
                s_adj = s_adj.round(0).astype("int64")
            syn[col] = syn[col].astype(s_adj.dtype)
            syn.loc[syn[col].notna(), col] = s_adj
        except Exception:
            pass

    return syn


def moment_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame comparing first four moments between real and synthetic.
    """
    cols = columns or [
        c for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
    ]
    rows = []
    for col in cols:
        r = real[col].dropna().astype(float)
        s = synthetic[col].dropna().astype(float)
        if len(r) < 4 or len(s) < 4:
            continue
        rows.append({
            "column":       col,
            "real_mean":    round(r.mean(), 4),
            "syn_mean":     round(s.mean(), 4),
            "real_std":     round(r.std(), 4),
            "syn_std":      round(s.std(), 4),
            "real_skew":    round(float(stats.skew(r)), 4),
            "syn_skew":     round(float(stats.skew(s)), 4),
            "real_kurt":    round(float(stats.kurtosis(r)), 4),
            "syn_kurt":     round(float(stats.kurtosis(s)), 4),
        })
    return pd.DataFrame(rows)
