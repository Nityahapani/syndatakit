"""
syndatakit.privacy.disclosure
-------------------------------
Membership inference attack: can an adversary determine whether a
specific real record was used to train the generator?

Methodology (shadow model approach, simplified):
  A "member" is a real training record.
  A "non-member" is a held-out real record not seen during training.
  We score each by its distance to its nearest synthetic neighbour.
  Members are expected to be closer (memorisation signal).
  Attack advantage = AUC - 0.5  (0 = no advantage, 0.5 = perfect attack).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _min_dist_to_synthetic(
    records: np.ndarray,
    synthetic: np.ndarray,
    batch: int = 100,
) -> np.ndarray:
    """Return minimum L2 distance from each record to the synthetic set."""
    min_dists = np.full(len(records), np.inf)
    for i in range(0, len(synthetic), batch):
        block = synthetic[i: i + batch]
        dists = np.sqrt(((records[:, None, :] - block[None, :, :]) ** 2).sum(axis=2))
        min_dists = np.minimum(min_dists, dists.min(axis=1))
    return min_dists


def _auc_from_scores(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """Compute AUC: P(member score < non-member score)."""
    n_m  = len(member_scores)
    n_nm = len(nonmember_scores)
    if n_m == 0 or n_nm == 0:
        return 0.5
    # Members should have lower distance (closer to synthetic = more memorised)
    wins = sum(
        float((member_scores < nm).mean())
        for nm in nonmember_scores
    )
    return round(wins / n_nm, 4)


def membership_inference_risk(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    n_sample: int = 200,
    seed: int = 42,
) -> dict:
    """
    Estimate membership inference risk.

    Parameters
    ----------
    real_train   : records used to train the generator (members)
    real_holdout : records NOT used in training (non-members)
    synthetic    : generated synthetic data
    n_sample     : number of records to sample from each group

    Returns
    -------
    dict with attack_auc, advantage (AUC - 0.5), risk_level
    """
    rng = np.random.default_rng(seed)

    cols = numeric_cols or [
        c for c in real_train.columns
        if c in synthetic.columns
        and pd.api.types.is_numeric_dtype(real_train[c])
        and c != "syn_id"
    ]

    if len(cols) < 2:
        return {"error": "Insufficient numeric columns", "attack_auc": 0.5}

    def prep(df, n):
        sample = df.sample(n=min(n, len(df)), random_state=int(seed))
        arr = sample[cols].fillna(0).values.astype(float)
        mu  = arr.mean(0); sd = arr.std(0) + 1e-9
        return (arr - mu) / sd

    members     = prep(real_train,   n_sample)
    nonmembers  = prep(real_holdout, n_sample)
    syn_arr     = prep(synthetic,    min(n_sample * 5, len(synthetic)))

    # Score: distance to nearest synthetic neighbour
    m_dists  = _min_dist_to_synthetic(members,    syn_arr)
    nm_dists = _min_dist_to_synthetic(nonmembers, syn_arr)

    auc       = _auc_from_scores(m_dists, nm_dists)
    advantage = round(auc - 0.5, 4)

    return {
        "attack_auc":  auc,
        "advantage":   advantage,
        "risk_level":  _risk_level(auc),
        "n_members":   len(members),
        "n_nonmembers":len(nonmembers),
        "interpretation": (
            "No meaningful memorisation detected"   if auc < 0.52 else
            "Slight memorisation signal"             if auc < 0.60 else
            "Moderate memorisation — review generator" if auc < 0.70 else
            "High memorisation risk — consider DP noise"
        ),
    }


def _risk_level(auc: float) -> str:
    if auc < 0.52: return "very_low"
    if auc < 0.60: return "low"
    if auc < 0.70: return "medium"
    if auc < 0.80: return "high"
    return "very_high"
