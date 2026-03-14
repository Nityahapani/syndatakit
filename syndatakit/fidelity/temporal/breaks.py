"""
syndatakit.fidelity.temporal.breaks
-------------------------------------
Structural break detection using a Chow-test inspired scan.

Checks whether synthetic series exhibit regime changes at roughly the same
points as the real series — important for macro data with recessions/crises.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _chow_statistic(arr: np.ndarray, bp: int) -> float:
    """F-statistic for a break at position bp."""
    n = len(arr)
    if bp < 3 or bp > n - 3:
        return 0.0

    def ssr(x):
        return float(np.sum((x - x.mean()) ** 2))

    s_full  = ssr(arr)
    s_split = ssr(arr[:bp]) + ssr(arr[bp:])
    k = 2  # intercept-only model
    f = ((s_full - s_split) / k) / max(s_split / max(n - 2 * k, 1), 1e-12)
    return float(f)


def detect_breaks(series: np.ndarray, n_candidates: int = 5) -> list[int]:
    """Return indices of top structural break candidates."""
    n = len(series)
    scores = [_chow_statistic(series, i) for i in range(3, n - 3)]
    if not scores:
        return []
    # Return top-n break points by F-statistic
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    breaks = []
    for idx in ranked:
        bp = idx + 3
        if not any(abs(bp - b) < 5 for b in breaks):
            breaks.append(bp)
        if len(breaks) >= n_candidates:
            break
    return sorted(breaks)


def breaks_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    tolerance: int = 5,
) -> dict:
    """
    Compare break point locations between real and synthetic series.
    A break is 'matched' if synthetic has one within ±tolerance steps.
    Score = fraction of real breaks that are matched.
    """
    cols = columns or [
        c for c in real.columns
        if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])
    ]
    results = {}
    total_real = total_matched = 0

    for col in cols:
        r_arr = real[col].dropna().astype(float).values
        s_arr = synthetic[col].dropna().astype(float).values
        if len(r_arr) < 20 or len(s_arr) < 20:
            continue

        r_breaks = detect_breaks(r_arr)
        s_breaks = detect_breaks(s_arr)

        matched = sum(
            1 for rb in r_breaks
            if any(abs(rb - sb) <= tolerance for sb in s_breaks)
        )
        total_real    += len(r_breaks)
        total_matched += matched

        results[col] = {
            "real_breaks":      r_breaks,
            "synthetic_breaks": s_breaks,
            "matched":          matched,
            "total_real":       len(r_breaks),
        }

    match_rate = round(total_matched / max(total_real, 1) * 100, 1)
    results["_summary"] = {
        "break_match_rate": match_rate,
        "total_real_breaks":    total_real,
        "total_matched_breaks": total_matched,
    }
    return results
