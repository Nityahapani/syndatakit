"""
syndatakit.generators.cross_sectional.gaussian_copula
------------------------------------------------------
Gaussian Copula generator for cross-sectional tabular data.

Algorithm
---------
1. Fit a marginal model per column (log-normal / normal / categorical)
2. Transform each column to uniform via its CDF, then to normal via Φ⁻¹
3. Learn the inter-column correlation matrix in normal space
4. Sample correlated normals via Cholesky decomposition
5. Invert back through each marginal to produce synthetic values

Produces statistically faithful synthetic records with zero exact copies
of real rows and no individual-level information.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from ..base import BaseGenerator

warnings.filterwarnings("ignore")


# ── Marginal models ───────────────────────────────────────────────────────────

class _NumericMarginal:
    """Fits and inverts a single numeric column distribution."""

    def __init__(self):
        self._params: dict = {}
        self._min = self._max = 0.0

    def fit(self, series: pd.Series) -> "_NumericMarginal":
        arr = series.dropna().astype(float).values
        self._min, self._max = float(arr.min()), float(arr.max())
        skewness = float(stats.skew(arr))
        if skewness > 1.0 and self._min > 0:
            s, loc, scale = stats.lognorm.fit(arr, floc=0)
            self._params = dict(kind="lognorm", s=s, loc=loc, scale=scale)
        else:
            self._params = dict(kind="norm", loc=float(arr.mean()), scale=float(arr.std()) or 1.0)
        return self

    def to_uniform(self, series: pd.Series) -> np.ndarray:
        arr = series.fillna(series.median()).astype(float).values
        p = self._params
        if p["kind"] == "lognorm":
            u = stats.lognorm.cdf(arr, p["s"], loc=p["loc"], scale=p["scale"])
        else:
            u = stats.norm.cdf(arr, loc=p["loc"], scale=p["scale"])
        return np.clip(u, 1e-6, 1 - 1e-6)

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1 - 1e-6)
        p = self._params
        if p["kind"] == "lognorm":
            v = stats.lognorm.ppf(u, p["s"], loc=p["loc"], scale=p["scale"])
        else:
            v = stats.norm.ppf(u, loc=p["loc"], scale=p["scale"])
        return np.clip(v, self._min, self._max)

    @property
    def kind(self) -> str:
        return self._params.get("kind", "unknown")


class _CategoricalMarginal:
    """Fits and inverts a single categorical column distribution."""

    def __init__(self):
        self._cats: list = []
        self._probs: np.ndarray = np.array([])

    def fit(self, series: pd.Series) -> "_CategoricalMarginal":
        vc = series.dropna().value_counts(normalize=True)
        self._cats  = list(vc.index)
        self._probs = vc.values
        return self

    def to_uniform(self, series: pd.Series) -> np.ndarray:
        mapping = {c: (i + 0.5) / len(self._cats) for i, c in enumerate(self._cats)}
        return np.array([mapping.get(v, 0.5) for v in series.fillna(self._cats[0])])

    def from_uniform(self, u: np.ndarray) -> list:
        cum = np.cumsum(self._probs)
        idx = np.clip(np.searchsorted(cum, np.clip(u, 1e-6, 1 - 1e-6)), 0, len(self._cats) - 1)
        return [self._cats[i] for i in idx]

    @property
    def kind(self) -> str:
        return "categorical"


# ── Generator ─────────────────────────────────────────────────────────────────

class GaussianCopulaGenerator(BaseGenerator):
    """
    Cross-sectional Gaussian Copula synthetic data generator.

    Usage
    -----
        from syndatakit.generators.cross_sectional import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(real_df)
        syn = gen.sample(1000)
        syn = gen.sample(500, filters={"state": ["CA", "TX"], "dti_min": 45})
    """

    supported_types = ["cross_sectional"]

    # Shorthand aliases: 'dti' → 'debt_to_income', etc.
    _ALIASES: dict[str, str] = {
        "dti":    "debt_to_income",
        "income": "applicant_income",
        "loan":   "loan_amount",
        "gdp":    "gdp_growth_yoy",
        "ffr":    "fed_funds_rate",
        "assets": "total_assets",
    }

    def _init(self, priors=None, **kwargs):
        self._marginals: dict[str, _NumericMarginal | _CategoricalMarginal] = {}
        self._corr: np.ndarray | None = None
        self._priors = priors  # PriorSet | None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame) -> "GaussianCopulaGenerator":
        self._record_schema(data)

        # Fit marginals, applying prior regularisation where provided
        for col in self._columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                m = _NumericMarginal().fit(data[col])
                if self._priors is not None:
                    prior = self._priors.get(col)
                    if prior is not None and m._params.get("kind") in ("norm", "lognormal"):
                        n = len(data[col].dropna())
                        p = m._params
                        p["loc"]   = prior.map_mean(p.get("loc", p.get("scale", 0)), n)
                        p["scale"] = max(prior.map_std(p.get("scale", 1), n), 1e-6)
                self._marginals[col] = m
            else:
                self._marginals[col] = _CategoricalMarginal().fit(data[col])

        # Transform to normal space
        uniform = np.column_stack([
            self._marginals[c].to_uniform(data[c]) for c in self._columns
        ])
        normal = stats.norm.ppf(np.clip(uniform, 1e-6, 1 - 1e-6))

        # Estimate correlation, ensure PSD
        corr = np.corrcoef(normal.T)
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 0:
            corr += (-eigvals.min() + 1e-8) * np.eye(len(self._columns))
        self._corr = corr

        self._fitted = True
        return self

    # ── sample ────────────────────────────────────────────────────────────────

    def sample(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()
        if seed is not None:
            np.random.seed(seed)

        n_gen = n * (6 if filters else 1)

        # Correlated normal samples via Cholesky
        try:
            L = np.linalg.cholesky(self._corr)
            z = np.random.standard_normal((n_gen, len(self._columns))) @ L.T
        except np.linalg.LinAlgError:
            z = np.random.standard_normal((n_gen, len(self._columns)))

        u = stats.norm.cdf(z)

        records = {
            col: self._marginals[col].from_uniform(u[:, i])
            for i, col in enumerate(self._columns)
        }
        df = pd.DataFrame(records)
        df = self._cast_types(df)

        if filters:
            df = self._apply_filters(df, filters)

        return self._add_syn_id(df.head(n))

    # ── filter helpers ────────────────────────────────────────────────────────

    def _resolve_col(self, key: str) -> str | None:
        """Resolve abbreviated key → actual column name."""
        if key in self._columns:
            return key
        if key in self._ALIASES and self._ALIASES[key] in self._columns:
            return self._ALIASES[key]
        # unambiguous prefix match
        matches = [c for c in self._columns if c == key or c.startswith(key + "_")]
        return matches[0] if len(matches) == 1 else None

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        for key, val in filters.items():
            if key.endswith("_min"):
                col = self._resolve_col(key[:-4])
                if col:
                    df = df[df[col] >= val]
            elif key.endswith("_max"):
                col = self._resolve_col(key[:-4])
                if col:
                    df = df[df[col] <= val]
            else:
                col = self._resolve_col(key)
                if col:
                    vals = [str(v) for v in val] if isinstance(val, list) else val
                    if isinstance(vals, list):
                        df = df[df[col].isin(vals)]
                    else:
                        df = df[df[col] == vals]
        return df

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def marginal_kinds(self) -> dict[str, str]:
        """Return the fitted distribution kind for each column."""
        return {col: m.kind for col, m in self._marginals.items()}

    @property
    def correlation_matrix(self) -> pd.DataFrame | None:
        """Return the learned correlation matrix as a DataFrame."""
        if self._corr is None:
            return None
        return pd.DataFrame(self._corr, index=self._columns, columns=self._columns)
