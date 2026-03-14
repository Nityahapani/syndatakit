"""
syndatakit.generators.deep.ctgan
---------------------------------
Stub for CTGAN (Conditional Tabular GAN) deep generator.

CTGAN significantly outperforms Gaussian Copula on:
  - Multi-modal numeric distributions
  - Imbalanced categorical columns
  - Complex non-linear inter-column relationships

Installation of the deep extra required:
    pip install syndatakit[deep]   # installs ctgan, torch

This stub raises an informative error until the deep extra is installed,
and documents the interface so the upgrade path is clear.
"""
from __future__ import annotations
import pandas as pd
from ..base import BaseGenerator


class CTGANGenerator(BaseGenerator):
    """
    CTGAN deep tabular generator.

    Requires: pip install syndatakit[deep]

    Usage (same interface as GaussianCopulaGenerator):
        gen = CTGANGenerator(epochs=300, batch_size=500)
        gen.fit(real_df)
        syn = gen.sample(1000)
    """

    supported_types = ["cross_sectional"]

    def _init(self, epochs: int = 300, batch_size: int = 500, **kwargs):
        self._epochs     = epochs
        self._batch_size = batch_size
        self._model      = None

    def _require_ctgan(self):
        try:
            import ctgan  # noqa: F401
        except ImportError:
            raise ImportError(
                "CTGAN is not installed.\n"
                "Run: pip install syndatakit[deep]\n\n"
                "This installs ctgan and its PyTorch dependency (~2 GB).\n"
                "If you don't need deep learning quality, GaussianCopulaGenerator\n"
                "achieves 96–98% fidelity with no additional dependencies."
            )

    def fit(self, data: pd.DataFrame) -> "CTGANGenerator":
        self._require_ctgan()
        from ctgan import CTGAN

        self._record_schema(data)
        discrete_cols = [
            c for c in self._columns
            if not pd.api.types.is_numeric_dtype(data[c])
        ]
        self._model = CTGAN(epochs=self._epochs, batch_size=self._batch_size, verbose=False)
        self._model.fit(data, discrete_columns=discrete_cols)
        self._fitted = True
        return self

    def sample(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()
        self._require_ctgan()
        df = self._model.sample(n * (4 if filters else 1))
        df = self._cast_types(df)
        if filters:
            df = self._apply_basic_filters(df, filters)
        return self._add_syn_id(df.head(n))

    def _apply_basic_filters(self, df, filters):
        for key, val in filters.items():
            if key in df.columns:
                if isinstance(val, list):
                    df = df[df[key].isin([str(v) for v in val])]
                else:
                    df = df[df[key] == val]
        return df
