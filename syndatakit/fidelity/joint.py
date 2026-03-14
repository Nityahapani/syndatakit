"""
syndatakit.fidelity.joint
--------------------------
Joint distribution fidelity — captures how well the generator
preserves inter-column relationships, not just marginals.

Metrics
-------
correlation_distance : Frobenius norm between real and synthetic
                       correlation matrices, normalised to 0–100.
pairwise_mi_score    : Average mutual information ratio across column pairs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def correlation_distance_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> float:
    """
    Score based on Frobenius distance between Spearman correlation matrices.
    Score = 100 means identical correlation structure.
    """
    cols = columns or [
        c for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
    ]
    if len(cols) < 2:
        return 100.0

    R_real = _spearman_matrix(real[cols])
    R_syn  = _spearman_matrix(synthetic[cols])

    max_possible = np.sqrt(2 * len(cols) * (len(cols) - 1))  # all ±1 → 0
    dist = np.linalg.norm(R_real - R_syn, "fro")
    score = max(0.0, 1 - dist / max(max_possible, 1e-8)) * 100
    return round(float(score), 2)


def _spearman_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute pairwise Spearman correlation matrix, handling NaNs."""
    num_df = df.select_dtypes(include="number")
    arr = num_df.fillna(num_df.median()).values.astype(float)
    n = arr.shape[1]
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(arr[:, i], arr[:, j])
            mat[i, j] = mat[j, i] = r if np.isfinite(r) else 0.0
    return mat


def pairwise_correlation_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """
    Per-pair Spearman correlation delta (real − synthetic).
    Returns dict of 'col_a × col_b' → delta.
    """
    cols = columns or [
        c for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
    ]
    # Ensure we only use truly numeric columns
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(real[c])]
    result = {}
    for i, ca in enumerate(cols):
        for cb in cols[i + 1:]:
            r_real, _ = spearmanr(
                real[ca].fillna(0).astype(float),
                real[cb].fillna(0).astype(float),
            )
            r_syn, _ = spearmanr(
                synthetic[ca].fillna(0).astype(float),
                synthetic[cb].fillna(0).astype(float),
            )
            delta = round(float(abs(r_real - r_syn)), 4)
            result[f"{ca} × {cb}"] = delta
    return result
