"""
syndatakit.privacy.singling_out
---------------------------------
Singling-out attack: can an adversary uniquely identify an individual
in the real dataset using only the synthetic data?

Methodology (simplified generalised singling-out):
  For each synthetic record, count how many real records match it
  on a random subset of quasi-identifier columns. If only 1 real record
  matches, the synthetic record has "singled out" that individual.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import itertools


def singling_out_risk(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    quasi_id_cols: list[str] | None = None,
    n_attacks: int = 500,
    seed: int = 42,
) -> dict:
    """
    Estimate singling-out risk via random quasi-identifier subset attacks.

    Returns
    -------
    dict with singling_out_rate (0–1), n_attacks, n_singled_out, risk_level
    """
    rng = np.random.default_rng(seed)

    shared = [c for c in real.columns if c in synthetic.columns and c != "syn_id"]
    qi_cols = quasi_id_cols or [
        c for c in shared if not pd.api.types.is_numeric_dtype(real[c])
    ][:8]

    if not qi_cols:
        return {"error": "No quasi-identifier columns found", "singling_out_rate": 0.0}

    n_singled = 0
    n_tested  = min(n_attacks, len(synthetic))
    syn_sample = synthetic.sample(n=n_tested, random_state=seed) if len(synthetic) > n_tested else synthetic

    for _, syn_row in syn_sample.iterrows():
        # Pick a random subset of 2–4 quasi-identifiers
        if len(qi_cols) < 2:
            n_singled += 0
            continue
        k = int(rng.integers(2, min(5, len(qi_cols) + 1)))
        cols = list(rng.choice(qi_cols, size=k, replace=False))

        mask = pd.Series([True] * len(real))
        for col in cols:
            mask = mask & (real[col].astype(str) == str(syn_row.get(col, "")))

        n_matching = int(mask.sum())
        if n_matching == 1:
            n_singled += 1

    rate = round(n_singled / max(n_tested, 1), 4)
    return {
        "singling_out_rate": rate,
        "n_attacks":         n_tested,
        "n_singled_out":     n_singled,
        "risk_level":        _risk_level(rate),
        "quasi_id_cols":     qi_cols,
    }


def _risk_level(rate: float) -> str:
    if rate < 0.001: return "very_low"
    if rate < 0.01:  return "low"
    if rate < 0.05:  return "medium"
    if rate < 0.15:  return "high"
    return "very_high"
