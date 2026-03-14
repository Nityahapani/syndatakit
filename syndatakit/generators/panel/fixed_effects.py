"""
syndatakit.generators.panel.fixed_effects
------------------------------------------
Panel data generator using a fixed-effects decomposition.

Models each observation as:
    y_it = α_i  (entity fixed effect)
           + β_t (time fixed effect)
           + ε_it (idiosyncratic shock, drawn from Gaussian Copula)

Suitable for: world_bank (country × year), fdic (bank × quarter),
              bls (industry × state × quarter).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from ..base import BaseGenerator
from ..cross_sectional.gaussian_copula import GaussianCopulaGenerator

warnings.filterwarnings("ignore")


class FixedEffectsGenerator(BaseGenerator):
    """
    Panel data generator with entity and time fixed effects.

    Usage
    -----
        from syndatakit.generators.panel import FixedEffectsGenerator

        gen = FixedEffectsGenerator(entity_col="country_code", time_col="year")
        gen.fit(real_df)
        syn = gen.sample(n=1000)
    """

    supported_types = ["panel"]

    def _init(
        self,
        entity_col: str = "entity",
        time_col: str   = "year",
        **kwargs,
    ):
        self._entity_col = entity_col
        self._time_col   = time_col
        self._entity_effects: dict = {}   # α_i per entity
        self._time_effects:   dict = {}   # β_t per time period
        self._entity_dist: dict   = {}    # distribution of entity-level means
        self._residual_gen: GaussianCopulaGenerator | None = None
        self._numeric_cols: list[str] = []
        self._cat_cols:     list[str] = []
        self._cat_freqs:    dict      = {}
        self._n_entities: int = 0
        self._n_periods:  int = 0

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame) -> "FixedEffectsGenerator":
        self._record_schema(data)
        df = data.copy()

        self._numeric_cols = [
            c for c in self._columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in (self._entity_col, self._time_col)
        ]
        self._cat_cols = [
            c for c in self._columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and c not in (self._entity_col, self._time_col)
        ]
        for col in self._cat_cols:
            self._cat_freqs[col] = df[col].value_counts(normalize=True).to_dict()

        # Entity fixed effects: mean of each numeric col per entity
        if self._entity_col in df.columns:
            entity_means = df.groupby(self._entity_col)[self._numeric_cols].mean()
            self._entity_effects = entity_means.to_dict(orient="index")
            self._n_entities = len(entity_means)
            # Distribution of entity effects
            for col in self._numeric_cols:
                vals = entity_means[col].dropna().values
                self._entity_dist[col] = dict(mean=float(vals.mean()), std=float(vals.std() or 1))
        else:
            self._n_entities = 100

        # Time fixed effects: mean of each numeric col per period
        if self._time_col in df.columns:
            time_means = df.groupby(self._time_col)[self._numeric_cols].mean()
            self._time_effects = time_means.to_dict(orient="index")
            self._n_periods = len(time_means)
        else:
            self._n_periods = 10

        # Fit a Gaussian Copula on the within-entity residuals
        residuals = df[self._numeric_cols].copy()
        if self._entity_col in df.columns:
            entity_means_full = df.groupby(self._entity_col)[self._numeric_cols].transform("mean")
            residuals = residuals - entity_means_full
        if self._time_col in df.columns:
            time_means_full = df.groupby(self._time_col)[self._numeric_cols].transform("mean")
            residuals = residuals - time_means_full

        self._residual_gen = GaussianCopulaGenerator()
        self._residual_gen.fit(residuals.dropna())

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

        # Generate synthetic entity IDs
        n_entities = max(self._n_entities, n // max(self._n_periods, 1))
        entity_ids = [f"ENT-{i:05d}" for i in range(n_entities)]

        # Sample entity-level means from their distribution
        entity_means: dict[str, dict] = {}
        for eid in entity_ids:
            if eid in self._entity_effects:
                entity_means[eid] = self._entity_effects[eid]
            else:
                entity_means[eid] = {
                    col: float(rng.normal(
                        self._entity_dist[col]["mean"],
                        self._entity_dist[col]["std"] * 0.5,
                    ))
                    for col in self._numeric_cols
                }

        # Draw residuals from the copula
        resid_df = self._residual_gen.sample(n, seed=seed).drop(columns=["syn_id"])

        # Assign entity + time, add fixed effects back
        rows = []
        for i in range(n):
            eid = entity_ids[i % len(entity_ids)]
            period = list(self._time_effects.keys())[i % max(self._n_periods, 1)] \
                     if self._time_effects else i % max(self._n_periods, 1)

            row = {self._entity_col: eid, self._time_col: period}

            for col in self._numeric_cols:
                alpha_i = entity_means[eid].get(col, 0)
                beta_t  = self._time_effects.get(period, {}).get(col, 0)
                eps     = float(resid_df.iloc[i % len(resid_df)][col]) if col in resid_df.columns else 0.0
                row[col] = alpha_i + beta_t * 0.3 + eps

            for col in self._cat_cols:
                cats  = list(self._cat_freqs[col].keys())
                probs = list(self._cat_freqs[col].values())
                row[col] = rng.choice(cats, p=np.array(probs) / sum(probs))

            rows.append(row)

        df = pd.DataFrame(rows)
        df = self._cast_types(df)
        return self._add_syn_id(df)
