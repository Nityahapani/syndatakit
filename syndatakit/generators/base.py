"""
syndatakit.generators.base
--------------------------
Abstract base class that every generator must implement.
Enforces a consistent interface across cross-sectional, time-series,
panel and deep generators.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseGenerator(ABC):
    """
    Abstract generator. Subclasses implement fit() and sample().

    Lifecycle
    ---------
        gen = MyGenerator(**kwargs)
        gen.fit(real_df)               # learn from data
        syn = gen.sample(n=1000)       # draw synthetic rows
        syn = gen.sample(n=500, filters={"state": ["CA"]})

    Or shorthand (fit + sample in one call):
        syn = gen.fit_sample(real_df, n=1000)
    """

    # Subclasses declare which dataset types they support
    # e.g. ["cross_sectional", "panel", "time_series"]
    supported_types: list[str] = []

    def __init__(self, **kwargs: Any):
        self._fitted   = False
        self._n_fit    = 0        # rows seen during fit
        self._columns  = []       # column order from fit
        self._dtypes   = {}       # original dtypes
        self._meta: dict = {}     # arbitrary metadata subclasses may store
        self._init(**kwargs)

    def _init(self, **kwargs: Any) -> None:
        """Optional hook for subclass __init__ logic without overriding __init__."""
        pass

    # ── Core interface ────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseGenerator":
        """
        Learn the statistical structure of real data.
        Must set self._fitted = True and self._columns before returning.
        Returns self (fluent interface).
        """

    @abstractmethod
    def sample(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Draw n synthetic rows.

        Parameters
        ----------
        n       : number of rows to return
        filters : optional column constraints (see filter spec in README)
        seed    : random seed for reproducibility

        Returns a DataFrame with a 'syn_id' column prepended.
        """

    # ── Convenience ───────────────────────────────────────────────────────────

    def fit_sample(
        self,
        data: pd.DataFrame,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Fit on data then immediately sample n rows."""
        return self.fit(data).sample(n, filters=filters, seed=seed)

    # ── Guard ─────────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call .fit(df) first."
            )

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _record_schema(self, df: pd.DataFrame) -> None:
        """Store column order and dtypes from the training DataFrame."""
        self._columns = list(df.columns)
        self._dtypes  = dict(df.dtypes)
        self._n_fit   = len(df)

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast generated columns back to original dtypes where safe."""
        import pandas as _pd
        for col, dtype in self._dtypes.items():
            if col not in df.columns:
                continue
            try:
                if _pd.api.types.is_integer_dtype(dtype):
                    df[col] = df[col].round(0).astype(int)
                elif _pd.api.types.is_float_dtype(dtype):
                    df[col] = df[col].round(4)
            except Exception:
                pass
        return df

    def _add_syn_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepend a unique synthetic row identifier."""
        df = df.reset_index(drop=True)
        df.insert(0, "syn_id", [f"SYN-{100_000 + i}" for i in range(len(df))])
        return df

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = f"fitted on {self._n_fit:,} rows" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({status})"
