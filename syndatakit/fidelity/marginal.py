"""
syndatakit.fidelity.marginal
-----------------------------
Per-column marginal distribution fidelity.

Numeric  → Kolmogorov-Smirnov two-sample test  (score = 1 - KS statistic)
Categorical → Total Variation Distance          (score = 1 - TVD/2)

Scores range 0–100. Higher = more faithful.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats


def marginal_scores(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute per-column marginal fidelity scores.

    Parameters
    ----------
    real, synthetic : DataFrames to compare
    columns         : subset of columns to score (default: all shared columns)

    Returns
    -------
    dict mapping column name → score (0–100)
    """
    cols = columns or [c for c in real.columns if c in synthetic.columns]
    scores: dict[str, float] = {}

    for col in cols:
        r = real[col].dropna()
        s = synthetic[col].dropna()
        if len(r) == 0 or len(s) == 0:
            continue

        if pd.api.types.is_numeric_dtype(r):
            ks, _ = stats.ks_2samp(r.astype(float), s.astype(float))
            scores[col] = round((1 - ks) * 100, 2)
        else:
            r_freq = r.value_counts(normalize=True)
            s_freq = s.value_counts(normalize=True)
            all_cats = set(r_freq.index) | set(s_freq.index)
            tvd = sum(abs(r_freq.get(c, 0) - s_freq.get(c, 0)) for c in all_cats) / 2
            scores[col] = round((1 - tvd) * 100, 2)

    return scores


def mean_marginal_score(scores: dict[str, float]) -> float:
    """Mean of all marginal scores."""
    return round(float(np.mean(list(scores.values()))), 2) if scores else 0.0
