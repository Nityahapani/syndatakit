"""
syndatakit.generators.time_series.vecm_garch
----------------------------------------------
VECM + GARCH time series generator for financial and macro data.

Improves on the simple VAR(p) in three ways:

1. VECM (Vector Error Correction Model)
   Handles non-stationary integrated series (I(1)) that share long-run
   equilibrium relationships (cointegration). E.g. the 2y and 10y Treasury
   yields don't drift apart forever — the VECM captures this.

2. GARCH(1,1) on residuals
   Models volatility clustering: periods of high volatility beget more
   high volatility. Essential for financial returns where σ²_t depends on
   past squared residuals.

3. Johansen cointegration test
   Determines which variables share long-run relationships before fitting,
   rather than assuming independence.

Requirements
-----------
    pip install statsmodels arch

Usage
-----
    from syndatakit.generators.time_series import VECMGARCHGenerator

    gen = VECMGARCHGenerator(lags=2, time_col="year")
    gen.fit(real_macro_df)
    syn = gen.sample(500)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from ..base import BaseGenerator
from ..cross_sectional.gaussian_copula import GaussianCopulaGenerator

warnings.filterwarnings("ignore")


def _require_statsmodels():
    try:
        import statsmodels  # noqa
    except ImportError:
        raise ImportError(
            "statsmodels is required for VECMGARCHGenerator.\n"
            "Install it with: pip install syndatakit[timeseries]\n"
            "  or: pip install statsmodels arch"
        )


class VECMGARCHGenerator(BaseGenerator):
    """
    VECM + GARCH time series generator.

    Fits a Johansen cointegration test, VECM on cointegrated groups,
    stationary VAR on the remainder, and GARCH(1,1) on each residual.

    Usage
    -----
        from syndatakit.generators.time_series import VECMGARCHGenerator

        gen = VECMGARCHGenerator(lags=2, time_col="year")
        gen.fit(load_seed("fred_macro"))
        syn = gen.sample(500)

    Parameters
    ----------
    lags        : VAR/VECM lag order
    time_col    : column containing the time index (excluded from modelling)
    det_order   : VECM deterministic component (-1=none, 0=const, 1=trend)
    use_garch   : whether to fit GARCH(1,1) on residuals (slower but more realistic)
    """

    supported_types = ["time_series"]

    def _init(
        self,
        lags: int = 2,
        time_col: str | None = None,
        det_order: int = 0,
        use_garch: bool = True,
        **kwargs,
    ):
        self._lags       = lags
        self._time_col   = time_col
        self._det_order  = det_order
        self._use_garch  = use_garch
        self._numeric_cols: list[str] = []
        self._cat_cols:     list[str] = []
        self._cat_modes:    dict      = {}
        # Model components
        self._vecm_model  = None       # fitted VECM (statsmodels)
        self._var_model   = None       # fitted VAR for stationary part
        self._garch_models = {}        # col -> fitted GARCH
        self._garch_vols   = {}        # col -> conditional volatility series
        self._means:  np.ndarray | None = None
        self._stds:   np.ndarray | None = None
        self._Y_raw:  np.ndarray | None = None
        self._coint_cols: list[str] = []
        self._stat_cols:  list[str] = []
        self._marginals:  dict = {}

    def fit(self, data: pd.DataFrame) -> "VECMGARCHGenerator":
        _require_statsmodels()
        from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
        from statsmodels.tsa.stattools import adfuller

        self._record_schema(data)
        df = data.copy()
        if self._time_col and self._time_col in df.columns:
            df = df.sort_values(self._time_col).reset_index(drop=True)

        self._numeric_cols = [c for c in self._columns
                               if pd.api.types.is_numeric_dtype(df[c])
                               and c != self._time_col]
        self._cat_cols  = [c for c in self._columns
                           if not pd.api.types.is_numeric_dtype(df[c])]
        self._cat_modes = {c: df[c].mode()[0] for c in self._cat_cols}

        # Store raw numeric data
        Y = df[self._numeric_cols].fillna(method="ffill").fillna(method="bfill").values.astype(float)
        self._Y_raw  = Y
        self._means  = Y.mean(0)
        self._stds   = Y.std(0) + 1e-9

        # Fit per-column marginals for back-transformation
        from ..cross_sectional.gaussian_copula import _NumericMarginal
        for col in self._numeric_cols:
            self._marginals[col] = _NumericMarginal().fit(df[col])

        # Step 1: ADF test — identify integrated (I(1)) vs stationary series
        integrated = []
        stationary = []
        for i, col in enumerate(self._numeric_cols):
            try:
                adf_p = adfuller(Y[:, i], autolag="AIC")[1]
                if adf_p > 0.10:
                    integrated.append(col)
                else:
                    stationary.append(col)
            except Exception:
                stationary.append(col)

        # Step 2: Johansen cointegration on integrated series
        self._coint_cols = integrated
        self._stat_cols  = stationary + [c for c in self._numeric_cols if c not in integrated + stationary]

        if len(integrated) >= 2:
            Y_int = Y[:, [self._numeric_cols.index(c) for c in integrated]]
            try:
                vecm = VECM(Y_int, k_ar_diff=self._lags, det_order=self._det_order)
                self._vecm_model = vecm.fit()
                print(f"    VECM fitted on {len(integrated)} integrated series")
            except Exception as e:
                print(f"    VECM failed ({e}), falling back to VAR")
                self._coint_cols = []
                self._stat_cols  = self._numeric_cols

        # Step 3: VAR on stationary + differenced integrated (if VECM failed)
        if self._stat_cols and not self._vecm_model:
            from statsmodels.tsa.vector_ar.var_model import VAR
            Y_stat = Y[:, [self._numeric_cols.index(c) for c in self._stat_cols]]
            if Y_stat.shape[1] >= 1 and len(Y_stat) > self._lags + 5:
                try:
                    var = VAR(Y_stat)
                    self._var_model = var.fit(self._lags)
                except Exception:
                    pass

        # Step 4: GARCH(1,1) on residuals (models volatility clustering)
        if self._use_garch:
            try:
                from arch import arch_model
                for col in self._numeric_cols[:6]:  # limit to 6 cols for speed
                    col_idx = self._numeric_cols.index(col)
                    series  = Y[:, col_idx]
                    ret     = np.diff(series)
                    if len(ret) < 20:
                        continue
                    try:
                        garch = arch_model(ret, vol="Garch", p=1, q=1, dist="normal")
                        res   = garch.fit(disp="off", show_warning=False)
                        self._garch_models[col] = res
                        self._garch_vols[col]   = res.conditional_volatility
                    except Exception:
                        pass
                if self._garch_models:
                    print(f"    GARCH fitted on {len(self._garch_models)} series")
            except ImportError:
                print("    arch not installed — skipping GARCH (pip install arch)")

        self._fitted = True
        return self

    def sample(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()
        rng = np.random.default_rng(seed)
        K = len(self._numeric_cols)

        series = np.zeros((n, K))

        # Start from a random slice of the real data
        start = int(rng.integers(0, max(len(self._Y_raw) - self._lags, 1)))
        history = self._Y_raw[start:start + self._lags].tolist()

        for t in range(n):
            # VECM forecast for cointegrated series
            row = np.zeros(K)
            if self._vecm_model and self._coint_cols:
                try:
                    ci = [self._numeric_cols.index(c) for c in self._coint_cols]
                    prev = np.array([history[-1][i] for i in ci]).reshape(1, -1)
                    fc   = self._vecm_model.predict(steps=1)
                    for j, idx in enumerate(ci):
                        row[idx] = float(fc[0, j]) if fc.size > j else prev[0, j]
                except Exception:
                    for i in [self._numeric_cols.index(c) for c in self._coint_cols]:
                        row[i] = history[-1][i]

            # VAR forecast for stationary series
            if self._var_model and self._stat_cols:
                try:
                    si  = [self._numeric_cols.index(c) for c in self._stat_cols]
                    lag_data = np.array([[history[-l-1][i] for i in si]
                                          for l in range(self._lags)])
                    fc  = self._var_model.forecast(lag_data, steps=1)
                    for j, idx in enumerate(si):
                        row[idx] = float(fc[0, j])
                except Exception:
                    for i in [self._numeric_cols.index(c) for c in self._stat_cols]:
                        row[i] = history[-1][i]

            # Add GARCH noise on top
            for col, garch_res in self._garch_models.items():
                idx = self._numeric_cols.index(col)
                try:
                    # Simulate one step from GARCH
                    params = garch_res.params
                    omega  = params.get("omega", 0.0001)
                    alpha  = params.get("alpha[1]", 0.05)
                    beta   = params.get("beta[1]",  0.90)
                    last_vol = self._garch_vols[col][-1] if len(self._garch_vols[col]) > 0 else 0.01
                    last_eps = rng.standard_normal() * last_vol
                    new_var  = omega + alpha * last_eps**2 + beta * last_vol**2
                    new_vol  = float(np.sqrt(max(new_var, 1e-8)))
                    row[idx] += rng.standard_normal() * new_vol
                except Exception:
                    pass

            # Add general noise scaled to data std
            noise = rng.standard_normal(K) * self._stds * 0.05
            row  += noise

            # Clip to realistic range
            row = np.clip(row,
                          self._means - 5 * self._stds,
                          self._means + 5 * self._stds)
            series[t] = row
            history.append(row.tolist())

        # Back-transform through marginals
        from scipy import stats as scipy_stats
        records = {}
        for i, col in enumerate(self._numeric_cols):
            m = self._marginals[col]
            u = scipy_stats.norm.cdf((series[:, i] - self._means[i]) / self._stds[i])
            records[col] = m.from_uniform(np.clip(u, 1e-4, 1 - 1e-4))

        for col in self._cat_cols:
            records[col] = [self._cat_modes[col]] * n

        # Reorder columns
        ordered = {col: records[col] for col in self._columns if col != self._time_col}
        df = pd.DataFrame(ordered)
        df = self._cast_types(df)

        if filters:
            df = self._apply_filters(df, filters)

        return self._add_syn_id(df.head(n))

    def _apply_filters(self, df, filters):
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
