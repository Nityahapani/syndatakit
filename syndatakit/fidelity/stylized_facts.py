"""
syndatakit.fidelity.stylized_facts
------------------------------------
Econometric stylized facts preservation tests.

Stylized facts are empirical regularities present in financial/macro data
that a good generator should preserve:

  1. Fat tails       — excess kurtosis > 0 (leptokurtic distributions)
  2. Volatility clustering — autocorrelation of squared returns (ARCH effects)
  3. Return autocorrelation — near-zero AC of raw returns
  4. Skewness sign   — left-skewed for equity returns, right for income data
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _excess_kurtosis(arr: np.ndarray) -> float:
    return float(stats.kurtosis(arr, fisher=True, nan_policy="omit"))

def _skewness(arr: np.ndarray) -> float:
    return float(stats.skew(arr, nan_policy="omit"))

def _autocorr(arr: np.ndarray, lag: int = 1) -> float:
    if len(arr) < lag + 2:
        return 0.0
    return float(np.corrcoef(arr[:-lag], arr[lag:])[0, 1])

def _arch_effect(arr: np.ndarray, lag: int = 5) -> float:
    """Autocorrelation of squared (de-meaned) series — proxy for ARCH."""
    dm = arr - arr.mean()
    sq = dm ** 2
    return _autocorr(sq, lag)


def stylized_facts_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict:
    """
    Compare stylized facts between real and synthetic data.

    Returns per-column fact comparison and an overall preservation score.
    """
    cols = columns or [
        c for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
        and pd.api.types.is_numeric_dtype(synthetic[c])
    ]

    results = {}
    scores = []

    for col in cols:
        r = real[col].dropna().astype(float).values
        s = synthetic[col].dropna().astype(float).values
        if len(r) < 10 or len(s) < 10:
            continue

        r_kurt = _excess_kurtosis(r)
        s_kurt = _excess_kurtosis(s)
        r_skew = _skewness(r)
        s_skew = _skewness(s)
        r_ac1  = _autocorr(r, 1)
        s_ac1  = _autocorr(s, 1)
        r_arch = _arch_effect(r)
        s_arch = _arch_effect(s)

        # Score each fact: 1 if sign/direction preserved, partial credit for magnitude
        fat_tail_match   = float((r_kurt > 0) == (s_kurt > 0))
        skew_sign_match  = float(np.sign(r_skew) == np.sign(s_skew))
        ac_sign_match    = float(np.sign(r_ac1) == np.sign(s_ac1))
        arch_sign_match  = float(np.sign(r_arch) == np.sign(s_arch))

        col_score = round((fat_tail_match + skew_sign_match + ac_sign_match + arch_sign_match) / 4 * 100, 1)
        scores.append(col_score)

        results[col] = {
            "kurtosis":          {"real": round(r_kurt, 3), "synthetic": round(s_kurt, 3), "match": bool(fat_tail_match)},
            "skewness":          {"real": round(r_skew, 3), "synthetic": round(s_skew, 3), "match": bool(skew_sign_match)},
            "autocorr_lag1":     {"real": round(r_ac1,  3), "synthetic": round(s_ac1,  3), "match": bool(ac_sign_match)},
            "arch_effect_lag5":  {"real": round(r_arch, 3), "synthetic": round(s_arch, 3), "match": bool(arch_sign_match)},
            "score": col_score,
        }

    results["_summary"] = {
        "mean_score":     round(float(np.mean(scores)), 1) if scores else 0.0,
        "columns_tested": len(scores),
    }
    return results
