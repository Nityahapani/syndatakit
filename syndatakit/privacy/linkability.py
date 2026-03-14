"""
syndatakit.privacy.linkability
--------------------------------
Linkability attack: can an adversary link a synthetic record back to
a specific real individual by matching on shared attributes?

Methodology:
  Split real data into two halves (A and B).
  Train a nearest-neighbour matcher on half-A.
  For each synthetic record, find its nearest neighbour in half-A.
  Check whether that neighbour's counterpart in half-B is closer to
  the synthetic record than a random record from half-B.
  If yes, the synthetic record has "linked" the individual across datasets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _normalise(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = df[cols].fillna(0).values.astype(float)
    mu  = arr.mean(0)
    sd  = arr.std(0) + 1e-9
    return (arr - mu) / sd


def _nearest_neighbour_idx(query: np.ndarray, pool: np.ndarray) -> int:
    dists = np.sum((pool - query) ** 2, axis=1)
    return int(np.argmin(dists))


def linkability_risk(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    n_attacks: int = 300,
    seed: int = 42,
) -> dict:
    """
    Estimate linkability risk via nearest-neighbour attack.

    Returns
    -------
    dict with linkability_rate (0–1), baseline (0.5 expected by chance),
    risk_level, and lift over baseline.
    """
    rng = np.random.default_rng(seed)

    cols = numeric_cols or [
        c for c in real.columns
        if c in synthetic.columns
        and pd.api.types.is_numeric_dtype(real[c])
        and c != "syn_id"
    ]

    if len(cols) < 2 or len(real) < 20:
        return {"error": "Insufficient data for linkability test", "linkability_rate": 0.0}

    # Split real into A and B
    idx   = rng.permutation(len(real))
    half  = len(idx) // 2
    real_A = real.iloc[idx[:half]].reset_index(drop=True)
    real_B = real.iloc[idx[half:]].reset_index(drop=True)
    min_half = min(len(real_A), len(real_B))
    real_A = real_A.iloc[:min_half]
    real_B = real_B.iloc[:min_half]

    A_norm = _normalise(real_A, cols)
    B_norm = _normalise(real_B, cols)

    n_test  = min(n_attacks, len(synthetic))
    syn_sample = synthetic.sample(n=n_test, random_state=int(seed)).reset_index(drop=True)

    linked = 0
    for i in range(n_test):
        syn_vec = _normalise(
            syn_sample.iloc[[i]], cols
        )[0]

        # Find nearest in A
        nn_a_idx = _nearest_neighbour_idx(syn_vec, A_norm)

        # Distance from syn to B[nn_a_idx] (the "linked" record)
        d_linked = float(np.sum((B_norm[nn_a_idx] - syn_vec) ** 2))

        # Distance from syn to a random record in B
        rand_b_idx = int(rng.integers(0, len(B_norm)))
        d_random   = float(np.sum((B_norm[rand_b_idx] - syn_vec) ** 2))

        if d_linked < d_random:
            linked += 1

    rate     = round(linked / max(n_test, 1), 4)
    baseline = 0.5   # expected by chance
    lift     = round((rate - baseline) / baseline * 100, 1)

    return {
        "linkability_rate": rate,
        "baseline":         baseline,
        "lift_over_baseline_pct": lift,
        "risk_level":       _risk_level(rate),
        "n_attacks":        n_test,
        "n_linked":         linked,
        "numeric_cols_used": cols,
    }


def _risk_level(rate: float) -> str:
    if rate < 0.52: return "very_low"
    if rate < 0.60: return "low"
    if rate < 0.70: return "medium"
    if rate < 0.85: return "high"
    return "very_high"
