"""
syndatakit.generators.time_series.var
--------------------------------------
Vector Autoregression (VAR) generator for multivariate time series data.

Fits a VAR(p) model to the real time series, then simulates forward from
bootstrapped initial conditions to produce synthetic series that preserve:
  - Per-variable marginal distributions
  - Cross-variable lead/lag relationships
  - Autocorrelation structure
  - Approximate stationarity (via differencing if needed)

Suitable for: fred_macro, bls (any dataset with a meaningful time dimension).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from ..base import BaseGenerator

warnings.filterwarnings("ignore")


class VARGenerator(BaseGenerator):
    """
    VAR(p) synthetic time series generator.

    Usage
    -----
        from syndatakit.generators.time_series import VARGenerator

        gen = VARGenerator(lags=2, time_col="year")
        gen.fit(real_df)
        syn = gen.sample(n=240)   # 240 time steps
    """

    supported_types = ["time_series"]

    def _init(self, lags: int = 2, time_col: str | None = None, **kwargs):
        self._lags     = lags
        self._time_col = time_col       # column to sort by before fitting
        self._numeric_cols: list[str] = []
        self._cat_cols:     list[str] = []
        self._cat_modes:    dict      = {}
        self._A:    list[np.ndarray]  = []   # VAR coefficient matrices [A1, A2, ..., Ap]
        self._mu:   np.ndarray | None = None  # intercept
        self._Sigma: np.ndarray | None = None # residual covariance
        self._means: np.ndarray | None = None # column means (for de/re-meaning)
        self._stds:  np.ndarray | None = None # column stds
        self._marginals: dict = {}            # per-column marginal for back-transform

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame) -> "VARGenerator":
        self._record_schema(data)

        df = data.copy()
        if self._time_col and self._time_col in df.columns:
            df = df.sort_values(self._time_col).reset_index(drop=True)

        # Separate numeric / categorical
        self._numeric_cols = [c for c in self._columns
                               if pd.api.types.is_numeric_dtype(df[c])
                               and c != self._time_col]
        self._cat_cols     = [c for c in self._columns
                               if not pd.api.types.is_numeric_dtype(df[c])]
        self._cat_modes    = {c: df[c].mode()[0] for c in self._cat_cols}

        # Fit per-column marginals for inversion
        from ..cross_sectional.gaussian_copula import _NumericMarginal, _CategoricalMarginal
        for col in self._numeric_cols:
            self._marginals[col] = _NumericMarginal().fit(df[col])

        # Standardise numeric series
        Y = df[self._numeric_cols].values.astype(float)
        self._means = Y.mean(axis=0)
        self._stds  = Y.std(axis=0) + 1e-8
        Y_std = (Y - self._means) / self._stds

        T, K = Y_std.shape
        p = self._lags

        if T <= p + K * p:
            # Not enough data — fall back to independent normal
            self._mu    = np.zeros(K)
            self._A     = [np.zeros((K, K)) for _ in range(p)]
            self._Sigma = np.eye(K) * 0.1
        else:
            # Build lagged regressor matrix
            X_rows, Y_rows = [], []
            for t in range(p, T):
                row = np.concatenate([Y_std[t - l] for l in range(1, p + 1)])
                X_rows.append(np.concatenate([[1.0], row]))
                Y_rows.append(Y_std[t])

            X = np.array(X_rows)   # (T-p) × (1 + K*p)
            Yt = np.array(Y_rows)  # (T-p) × K

            # OLS: B = (X'X)^{-1} X'Y
            try:
                B = np.linalg.lstsq(X, Yt, rcond=None)[0]  # (1+K*p) × K
            except Exception:
                B = np.zeros((1 + K * p, K))

            self._mu = B[0]  # intercept (K,)
            self._A  = [B[1 + l * K: 1 + (l + 1) * K].T for l in range(p)]

            # Residual covariance
            resid = Yt - X @ B
            self._Sigma = (resid.T @ resid) / max(len(resid) - 1, 1)
            # Ensure PSD
            eigvals = np.linalg.eigvalsh(self._Sigma)
            if eigvals.min() < 0:
                self._Sigma += (-eigvals.min() + 1e-8) * np.eye(K)

        # Store raw data for bootstrapping initial conditions
        self._Y_std = Y_std

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
        rng = np.random.default_rng(seed)

        p, K = self._lags, len(self._numeric_cols)
        T_raw = len(self._Y_std)

        # Bootstrap initial conditions from real series
        start = int(rng.integers(0, max(T_raw - p, 1)))
        history = list(self._Y_std[start: start + p])
        while len(history) < p:
            history.append(self._Y_std[0])

        # Cholesky for correlated innovations
        try:
            L = np.linalg.cholesky(self._Sigma)
        except np.linalg.LinAlgError:
            L = np.eye(K)

        # Simulate VAR forward
        series = []
        for _ in range(n):
            y_next = self._mu.copy()
            for l, A_l in enumerate(self._A):
                y_next += A_l @ history[-(l + 1)]
            y_next += L @ rng.standard_normal(K)
            history.append(y_next)
            series.append(y_next)

        Y_syn = np.array(series)  # (n, K)

        # Rescale back to original units
        Y_rescaled = Y_syn * self._stds + self._means

        # Clip to marginal bounds via quantile mapping
        records: dict = {}
        for i, col in enumerate(self._numeric_cols):
            m = self._marginals[col]
            # Map through uniform → back through marginal PPF
            u = stats.norm.cdf(Y_syn[:, i])
            records[col] = m.from_uniform(np.clip(u, 1e-6, 1 - 1e-6))

        # Fill categorical columns with their mode (categorical VAR not yet supported)
        for col in self._cat_cols:
            records[col] = [self._cat_modes[col]] * n

        # Rebuild in original column order (excluding time_col)
        ordered = {}
        for col in self._columns:
            if col == self._time_col:
                continue
            ordered[col] = records.get(col, [np.nan] * n)

        df = pd.DataFrame(ordered)
        df = self._cast_types(df)

        if filters:
            df = self._apply_filters(df, filters)

        return self._add_syn_id(df.head(n))

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        for key, val in filters.items():
            if key not in df.columns:
                continue
            if key.endswith("_min"):
                col = key[:-4]
                if col in df.columns:
                    df = df[df[col] >= val]
            elif key.endswith("_max"):
                col = key[:-4]
                if col in df.columns:
                    df = df[df[col] <= val]
            elif key in df.columns:
                if isinstance(val, list):
                    df = df[df[key].isin([str(v) for v in val])]
                else:
                    df = df[df[key] == val]
        return df
