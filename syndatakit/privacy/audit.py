"""
syndatakit.privacy.audit
--------------------------
Full privacy audit: runs all privacy tests and returns a structured report.

Tests run
---------
1. Exact copy check         — zero-tolerance
2. Membership inference     — AUC-based attack advantage
3. Singling-out risk        — quasi-identifier subset attack
4. Linkability risk         — nearest-neighbour cross-dataset attack

Each test returns a risk_level: very_low | low | medium | high | very_high
The overall verdict is the maximum risk level across all tests.
"""
from __future__ import annotations
import time
import pandas as pd
import numpy as np

from .singling_out import singling_out_risk
from .linkability  import linkability_risk
from .disclosure   import membership_inference_risk


_RISK_ORDER = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
_RISK_LABEL = {0: "very_low", 1: "low", 2: "medium", 3: "high", 4: "very_high"}


def privacy_audit(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    holdout_frac: float = 0.2,
    quasi_id_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    n_attacks: int = 300,
    seed: int = 42,
) -> dict:
    """
    Run all privacy tests against a synthetic dataset.

    Parameters
    ----------
    real          : real data used to train the generator
    synthetic     : generated synthetic data
    holdout_frac  : fraction of real data treated as non-members for MI test
    quasi_id_cols : columns used as quasi-identifiers for singling-out
    numeric_cols  : columns used for linkability / MI distance computation
    n_attacks     : number of attack attempts per test
    seed          : random seed

    Returns
    -------
    Nested dict with per-test results and an overall verdict.
    """
    t0  = time.time()
    rng = np.random.default_rng(seed)

    report: dict = {}

    # ── Exact copy check ──────────────────────────────────────────────────────
    shared = [c for c in real.columns if c in synthetic.columns and c != "syn_id"]
    real_hashes = set(real[shared].astype(str).apply("|".join, axis=1))
    syn_cols    = synthetic[[c for c in shared if c in synthetic.columns]]
    syn_hashes  = syn_cols.astype(str).apply("|".join, axis=1)
    n_exact     = int(syn_hashes.isin(real_hashes).sum())

    report["exact_copies"] = {
        "count":      n_exact,
        "rate":       round(n_exact / max(len(synthetic), 1), 6),
        "risk_level": "very_low" if n_exact == 0 else "very_high",
    }

    # ── Membership inference ──────────────────────────────────────────────────
    idx     = rng.permutation(len(real))
    split   = int(len(real) * (1 - holdout_frac))
    train   = real.iloc[idx[:split]].reset_index(drop=True)
    holdout = real.iloc[idx[split:]].reset_index(drop=True)

    report["membership_inference"] = membership_inference_risk(
        real_train=train,
        real_holdout=holdout,
        synthetic=synthetic,
        numeric_cols=numeric_cols,
        n_sample=n_attacks,
        seed=seed,
    )

    # ── Singling-out ─────────────────────────────────────────────────────────
    report["singling_out"] = singling_out_risk(
        real=real,
        synthetic=synthetic,
        quasi_id_cols=quasi_id_cols,
        n_attacks=n_attacks,
        seed=seed,
    )

    # ── Linkability ───────────────────────────────────────────────────────────
    report["linkability"] = linkability_risk(
        real=real,
        synthetic=synthetic,
        numeric_cols=numeric_cols,
        n_attacks=n_attacks,
        seed=seed,
    )

    # ── Overall verdict ───────────────────────────────────────────────────────
    risk_levels = [
        report["exact_copies"]["risk_level"],
        report["membership_inference"].get("risk_level", "very_low"),
        report["singling_out"].get("risk_level", "very_low"),
        report["linkability"].get("risk_level", "very_low"),
    ]
    max_risk = max(_RISK_ORDER.get(r, 0) for r in risk_levels)

    report["verdict"] = {
        "overall_risk":   _RISK_LABEL[max_risk],
        "exact_copies":   n_exact,
        "mi_auc":         report["membership_inference"].get("attack_auc", 0.5),
        "singling_out_rate": report["singling_out"].get("singling_out_rate", 0.0),
        "linkability_rate":  report["linkability"].get("linkability_rate", 0.5),
        "elapsed_seconds":   round(time.time() - t0, 3),
        "recommendation": _recommendation(max_risk, n_exact),
    }

    return report


def _recommendation(max_risk: int, exact_copies: int) -> str:
    if exact_copies > 0:
        return "FAIL: exact copies of real rows found. Check generation pipeline."
    if max_risk == 0:
        return "PASS: all privacy tests pass. Safe to release."
    if max_risk == 1:
        return "PASS with caution: low risk detected. Acceptable for most use cases."
    if max_risk == 2:
        return "REVIEW: medium risk detected. Consider applying DP noise or increasing dataset size."
    if max_risk == 3:
        return "FAIL: high risk detected. Apply differential privacy before release."
    return "FAIL: very high risk. Do not release without significant privacy hardening."


def format_audit(report: dict, width: int = 60) -> str:
    """Return a human-readable audit report string."""
    lines = ["=" * width, "  PRIVACY AUDIT REPORT", "=" * width]
    v = report.get("verdict", {})

    overall = v.get("overall_risk", "—").upper()
    icon = "✓" if overall in ("VERY_LOW", "LOW") else "✗"
    lines.append(f"  {icon} Overall risk: {overall}")
    lines.append("")

    ec = report.get("exact_copies", {})
    lines.append(f"  Exact copies      : {ec.get('count', '—')}  [{ec.get('risk_level','—')}]")

    mi = report.get("membership_inference", {})
    lines.append(f"  Membership inf.   : AUC={mi.get('attack_auc','—')}  [{mi.get('risk_level','—')}]")
    lines.append(f"    {mi.get('interpretation','')}")

    so = report.get("singling_out", {})
    lines.append(f"  Singling-out      : rate={so.get('singling_out_rate','—')}  [{so.get('risk_level','—')}]")

    lk = report.get("linkability", {})
    lines.append(f"  Linkability       : rate={lk.get('linkability_rate','—')}  [{lk.get('risk_level','—')}]")
    lines.append(f"    lift={lk.get('lift_over_baseline_pct','—')}% over baseline")

    lines.append("")
    lines.append(f"  Recommendation: {v.get('recommendation','—')}")
    lines.append(f"  Elapsed: {v.get('elapsed_seconds','—')}s")
    lines.append("=" * width)
    return "\n".join(lines)
