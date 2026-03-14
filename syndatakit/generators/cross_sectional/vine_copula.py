"""
syndatakit.generators.cross_sectional.vine_copula
--------------------------------------------------
Vine (Pair) Copula generator for cross-sectional data.

Improves on Gaussian Copula in two key ways:
1. Each column pair gets its own copula family (Clayton, Gumbel, Frank, Joe, Gaussian)
   chosen by AIC — capturing asymmetric dependence structures
2. Tail dependence: Clayton captures lower tail dependence (joint crashes),
   Gumbel captures upper tail dependence (joint booms)

This matters for financial data where correlations are higher in downturns
than in normal periods — the Gaussian Copula misses this completely.

Requirements
-----------
    pip install pyvinecopulib

Usage
-----
    from syndatakit.generators.cross_sectional import VineCopulaGenerator

    gen = VineCopulaGenerator()
    gen.fit(real_df)
    syn = gen.sample(10_000)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from ..base import BaseGenerator

warnings.filterwarnings("ignore")


def _require_pyvine():
    try:
        import pyvinecopulib  # noqa
    except ImportError:
        raise ImportError(
            "pyvinecopulib is required for VineCopulaGenerator.\n"
            "Install it with: pip install syndatakit[vine]\n"
            "  or: pip install pyvinecopulib"
        )


class VineCopulaGenerator(BaseGenerator):
    """
    Vine (pair) copula generator.

    Fits a C-vine or R-vine structure where each pair of variables
    gets its own copula family, capturing asymmetric tail dependence.

    Usage
    -----
        gen = VineCopulaGenerator(family_set="all")
        gen.fit(real_df)
        syn = gen.sample(10_000)

    Parameters
    ----------
    family_set : copula families to consider
        "parametric"   — Gaussian, Student-t, Clayton, Gumbel, Frank, Joe
        "all"          — all parametric + non-parametric
        "tll"          — transformation kernel (most flexible, slowest)
        list           — e.g. ["gaussian", "clayton", "gumbel"]
    trunc_lvl : int
        Truncation level for the vine (0 = full vine, 1 = fast approximation)
    """

    supported_types = ["cross_sectional"]

    def _init(self, family_set: str = "parametric", trunc_lvl: int = 0, **kwargs):
        self._family_set   = family_set
        self._trunc_lvl    = trunc_lvl
        self._vine         = None
        self._marginals    = {}
        self._numeric_cols: list[str] = []
        self._cat_cols:     list[str] = []
        self._cat_marginals = {}

    def fit(self, data: pd.DataFrame) -> "VineCopulaGenerator":
        _require_pyvine()
        import pyvinecopulib as pv

        self._record_schema(data)

        self._numeric_cols = [c for c in self._columns
                               if pd.api.types.is_numeric_dtype(data[c])]
        self._cat_cols     = [c for c in self._columns
                               if not pd.api.types.is_numeric_dtype(data[c])]

        # Fit marginals for each numeric column (empirical CDF)
        for col in self._numeric_cols:
            arr = data[col].dropna().astype(float).values
            self._marginals[col] = {
                "sorted": np.sort(arr),
                "n":      len(arr),
                "min":    arr.min(),
                "max":    arr.max(),
            }

        # Fit categorical distributions
        for col in self._cat_cols:
            vc = data[col].value_counts(normalize=True)
            self._cat_marginals[col] = {"cats": list(vc.index), "probs": vc.values}

        # Transform to uniform via empirical CDF
        n = len(data)
        U = np.zeros((n, len(self._numeric_cols)))
        for i, col in enumerate(self._numeric_cols):
            arr = data[col].fillna(data[col].median()).astype(float).values
            ranks = stats.rankdata(arr)
            U[:, i] = ranks / (n + 1)

        U = np.clip(U, 1e-4, 1 - 1e-4)

        # Fit vine copula
        controls = pv.FitControlsVinecop(
            family_set=pv.BicopFamily.__members__.get(
                self._family_set, list(pv.BicopFamily.__members__.values())[:8]
            ) if isinstance(self._family_set, str) and self._family_set != "parametric"
            else [pv.BicopFamily.gaussian, pv.BicopFamily.student,
                  pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
                  pv.BicopFamily.frank,   pv.BicopFamily.joe],
            trunc_lvl=self._trunc_lvl,
            num_threads=1,
        )
        self._vine = pv.Vinecop(data=U, controls=controls)
        self._fitted = True
        return self

    def sample(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()
        _require_pyvine()

        n_gen = n * (6 if filters else 1)

        # Simulate from vine
        if seed is not None:
            np.random.seed(seed)
        U_syn = self._vine.simulate(n_gen)
        U_syn = np.clip(U_syn, 1e-4, 1 - 1e-4)

        # Invert through empirical marginals (quantile function)
        records = {}
        for i, col in enumerate(self._numeric_cols):
            m = self._marginals[col]
            # Empirical quantile: interpolate in sorted array
            quantile_idx = U_syn[:, i] * (m["n"] - 1)
            lower = np.floor(quantile_idx).astype(int)
            upper = np.minimum(lower + 1, m["n"] - 1)
            frac   = quantile_idx - lower
            values = m["sorted"][lower] * (1 - frac) + m["sorted"][upper] * frac
            records[col] = np.clip(values, m["min"], m["max"])

        # Sample categorical columns independently (vine doesn't model categoricals)
        for col in self._cat_cols:
            m = self._cat_marginals[col]
            records[col] = np.random.choice(m["cats"], size=n_gen, p=m["probs"])

        df = pd.DataFrame(records)[self._columns]
        df = self._cast_types(df)

        if filters:
            df = self._apply_filters(df, filters)

        return self._add_syn_id(df.head(n))

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        for key, val in filters.items():
            if key.endswith("_min"):
                col = key[:-4]
                if col in df.columns: df = df[df[col] >= val]
            elif key.endswith("_max"):
                col = key[:-4]
                if col in df.columns: df = df[df[col] <= val]
            elif key in df.columns:
                if isinstance(val, list): df = df[df[key].isin([str(v) for v in val])]
                else: df = df[df[key] == val]
        return df

    def tail_dependence_report(self) -> dict:
        """
        Return upper and lower tail dependence coefficients for each pair.
        High values (> 0.1) indicate the variables tend to move together in extremes.
        """
        if self._vine is None:
            return {}
        report = {}
        for i, ci in enumerate(self._numeric_cols):
            for j, cj in enumerate(self._numeric_cols[i+1:], i+1):
                # Get the bicop for this pair from the first tree
                try:
                    bicop = self._vine.get_pair_copula(0, i)
                    report[f"{ci} × {cj}"] = {
                        "family":     str(bicop.family),
                        "parameters": bicop.parameters.tolist(),
                        "tau":        round(float(bicop.tau), 3),
                        "lower_tail": round(float(bicop.ltdc()), 3),
                        "upper_tail": round(float(bicop.utdc()), 3),
                    }
                except Exception:
                    pass
        return report
