"""
syndatakit.fidelity.report
---------------------------
Assembles a complete fidelity report by running all available metrics.
Returns a structured dict suitable for JSON serialisation or CLI display.
"""
from __future__ import annotations
import time
import pandas as pd
import numpy as np

from .marginal import marginal_scores, mean_marginal_score
from .joint import correlation_distance_score, pairwise_correlation_report
from .stylized_facts import stylized_facts_score
from .causality import causality_score


def fidelity_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    dataset_type: str = "cross_sectional",   # cross_sectional | time_series | panel
    target_col: str | None = None,            # for TSTR downstream score
    include_temporal: bool | None = None,     # auto-detect from dataset_type
    include_downstream: bool = True,
    columns: list[str] | None = None,
) -> dict:
    """
    Run all applicable fidelity metrics and return a structured report.

    Parameters
    ----------
    real, synthetic  : DataFrames to compare
    dataset_type     : drives which temporal tests are included
    target_col       : if given, runs TSTR downstream evaluation
    include_temporal : override temporal test inclusion
    include_downstream : run TSTR if target_col is provided
    columns          : restrict to these columns (default: all shared)

    Returns
    -------
    Nested dict with sections: marginal, joint, temporal (optional),
    stylized_facts, downstream (optional), summary.
    """
    t0 = time.time()

    cols = columns or [c for c in real.columns if c in synthetic.columns
                       and c != "syn_id"]
    syn = synthetic.drop(columns=["syn_id"], errors="ignore")

    report: dict = {"dataset_type": dataset_type, "columns_evaluated": cols}

    # ── Marginal ──────────────────────────────────────────────────────────────
    m_scores = marginal_scores(real, syn, cols)
    report["marginal"] = {
        "column_scores": m_scores,
        "mean_score":    mean_marginal_score(m_scores),
    }

    # ── Joint ─────────────────────────────────────────────────────────────────
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(real[c])]
    corr_score = correlation_distance_score(real, syn, num_cols)
    report["joint"] = {
        "correlation_distance_score": corr_score,
        "pairwise_deltas": pairwise_correlation_report(real, syn, num_cols),
    }

    # ── Stylized facts ────────────────────────────────────────────────────────
    report["stylized_facts"] = stylized_facts_score(real, syn, num_cols)

    # ── Temporal (time series / panel only) ───────────────────────────────────
    do_temporal = include_temporal if include_temporal is not None \
                  else dataset_type in ("time_series", "panel")

    if do_temporal:
        from .temporal.stationarity  import stationarity_score
        from .temporal.cointegration import cointegration_score
        from .temporal.breaks        import breaks_score

        report["temporal"] = {
            "stationarity":  stationarity_score(real, syn, cols),
            "cointegration": cointegration_score(real, syn),
            "breaks":        breaks_score(real, syn, cols),
            "causality":     causality_score(real, syn),
        }

    # ── Downstream ────────────────────────────────────────────────────────────
    if include_downstream and target_col and target_col in real.columns:
        from .downstream import tstr_score
        report["downstream"] = tstr_score(real, syn, target_col=target_col)

    # ── Privacy (basic — full audit is in privacy/audit.py) ──────────────────
    real_hashes = set(real[cols].astype(str).apply("|".join, axis=1))
    syn_hashes  = syn[[c for c in cols if c in syn.columns]].astype(str).apply("|".join, axis=1)
    exact_copies = int(syn_hashes.isin(real_hashes).sum())
    report["privacy_basic"] = {
        "exact_copies":  exact_copies,
        "privacy_score": round((1 - exact_copies / max(len(syn), 1)) * 100, 2),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    all_scores = [report["marginal"]["mean_score"], corr_score]
    sf_mean = report["stylized_facts"].get("_summary", {}).get("mean_score", None)
    if sf_mean is not None:
        all_scores.append(sf_mean)

    report["summary"] = {
        "overall_fidelity":  round(float(np.mean(all_scores)), 2),
        "marginal_score":    report["marginal"]["mean_score"],
        "joint_score":       corr_score,
        "privacy_score":     report["privacy_basic"]["privacy_score"],
        "exact_copies":      exact_copies,
        "rows_real":         len(real),
        "rows_synthetic":    len(syn),
        "elapsed_seconds":   round(time.time() - t0, 3),
    }

    return report


def format_report(report: dict, width: int = 60) -> str:
    """Return a human-readable string summary of a fidelity report."""
    lines = []
    s = report.get("summary", {})

    lines.append("=" * width)
    lines.append("  SYNDATAKIT FIDELITY REPORT")
    lines.append("=" * width)
    lines.append(f"  Dataset type    : {report.get('dataset_type','—')}")
    lines.append(f"  Rows (real/syn) : {s.get('rows_real','?')} / {s.get('rows_synthetic','?')}")
    lines.append("")
    lines.append(f"  Overall fidelity: {s.get('overall_fidelity','—')}%")
    lines.append(f"  Marginal score  : {s.get('marginal_score','—')}%")
    lines.append(f"  Joint score     : {s.get('joint_score','—')}%")
    lines.append(f"  Privacy score   : {s.get('privacy_score','—')}%")
    lines.append(f"  Exact copies    : {s.get('exact_copies','—')}")
    lines.append("")

    m_cols = report.get("marginal", {}).get("column_scores", {})
    if m_cols:
        lines.append("  Per-column marginal scores:")
        for col, sc in m_cols.items():
            bar = "█" * int(sc / 5) + "░" * (20 - int(sc / 5))
            lines.append(f"    {col:<26} {bar}  {sc}%")

    if "downstream" in report:
        d = report["downstream"]
        lines.append("")
        lines.append(f"  Downstream (TSTR):")
        lines.append(f"    Target  : {d.get('target_col')}")
        lines.append(f"    Metric  : {d.get('metric')} | TSTR {d.get('tstr_score')} | TRR {d.get('trr_score')}")
        lines.append(f"    Ratio   : {d.get('ratio')}  — {d.get('interpretation','')[:50]}")

    lines.append("")
    lines.append(f"  Computed in {s.get('elapsed_seconds','?')}s")
    lines.append("=" * width)
    return "\n".join(lines)
