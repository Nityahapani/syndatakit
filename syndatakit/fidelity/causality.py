"""
syndatakit.fidelity.causality
------------------------------
Granger causality preservation test.

Checks whether synthetic series preserve the directional predictive
relationships present in real data — e.g. whether yield_curve_spread
still Granger-causes unemployment_rate in the synthetic data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _granger_fstat(x: np.ndarray, y: np.ndarray, lags: int = 2) -> float:
    """
    Bivariate Granger causality F-statistic: does x Granger-cause y?
    Restricted model: y ~ y_lags
    Unrestricted model: y ~ y_lags + x_lags
    """
    n = len(y)
    if n < lags * 2 + 10:
        return 0.0

    def build_X(cols: list[np.ndarray]) -> np.ndarray:
        rows = []
        for t in range(lags, n):
            row = [1.0]
            for col in cols:
                row.extend(col[t - l] for l in range(1, lags + 1))
            rows.append(row)
        return np.array(rows)

    Y_t = y[lags:]

    # Restricted: only y lags
    Xr = build_X([y])
    try:
        br   = np.linalg.lstsq(Xr, Y_t, rcond=None)[0]
        ssr_r = float(np.sum((Y_t - Xr @ br) ** 2))
    except Exception:
        return 0.0

    # Unrestricted: y lags + x lags
    Xu = build_X([y, x])
    try:
        bu   = np.linalg.lstsq(Xu, Y_t, rcond=None)[0]
        ssr_u = float(np.sum((Y_t - Xu @ bu) ** 2))
    except Exception:
        return 0.0

    df1 = lags
    df2 = max(n - 2 * lags - 1, 1)
    f   = ((ssr_r - ssr_u) / df1) / max(ssr_u / df2, 1e-12)
    return float(f)


def causality_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    lags: int = 2,
    f_threshold: float = 3.84,   # ~5% critical value for F(2, ∞)
) -> dict:
    """
    For each (x→y) pair, test Granger causality on real and synthetic.
    Score = fraction of pairs where both agree on whether causality exists.
    """
    num_cols = [c for c in real.columns
                if c in synthetic.columns and pd.api.types.is_numeric_dtype(real[c])]

    if pairs is None:
        pairs = [(ca, cb) for i, ca in enumerate(num_cols[:5])
                           for cb in num_cols[i + 1:5]]

    results = {}
    agreements = 0

    for (ca, cb) in pairs:
        if ca not in real.columns or cb not in real.columns:
            continue
        rx  = real[ca].dropna().astype(float).values
        ry  = real[cb].dropna().astype(float).values
        sx  = synthetic[ca].dropna().astype(float).values
        sy  = synthetic[cb].dropna().astype(float).values
        n   = min(len(rx), len(ry), len(sx), len(sy))
        if n < 20:
            continue

        rf = _granger_fstat(rx[:n], ry[:n], lags)
        sf = _granger_fstat(sx[:n], sy[:n], lags)

        r_causes = rf > f_threshold
        s_causes = sf > f_threshold
        agree = r_causes == s_causes
        if agree:
            agreements += 1

        results[f"{ca} → {cb}"] = {
            "real_f":        round(rf, 3),
            "synthetic_f":   round(sf, 3),
            "real_causes":   r_causes,
            "synthetic_causes": s_causes,
            "agreement":     agree,
        }

    n_pairs = max(len(results), 1)
    results["_summary"] = {
        "agreement_rate": round(agreements / n_pairs * 100, 1),
        "pairs_tested":   len(results) - 1,
        "f_threshold":    f_threshold,
    }
    return results
