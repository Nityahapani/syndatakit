"""
syndatakit.calibration.priors
------------------------------
Bayesian parameter priors for generator fitting.

Purpose
-------
When fitting a generator on small datasets (< 500 rows), maximum-likelihood
estimates of distribution parameters are noisy. Priors encode domain knowledge
to regularise these estimates toward plausible values.

A Prior is a named distribution over a single parameter. A PriorSet bundles
priors for all columns in a dataset and is passed into a generator's fit()
method via the `priors` argument.

Built-in prior sets are provided for all 10 catalog datasets, derived from
published statistics (CFPB reports, Fed H.8, BLS bulletins, etc.).

Usage
-----
    from syndatakit.calibration.priors import PriorSet, DATASET_PRIORS
    from syndatakit.generators import GaussianCopulaGenerator
    from syndatakit.catalog import load_seed

    # Use a built-in prior set
    priors = DATASET_PRIORS["hmda"]
    gen = GaussianCopulaGenerator(priors=priors)
    gen.fit(load_seed("hmda").sample(100))   # only 100 rows — priors stabilise estimates

    # Define a custom prior set
    from syndatakit.calibration.priors import Prior, PriorSet
    priors = PriorSet({
        "loan_amount":      Prior("lognormal", mu=12.1, sigma=0.7),
        "applicant_income": Prior("lognormal", mu=11.2, sigma=0.6),
        "debt_to_income":   Prior("normal",    mu=38.0, sigma=10.0),
    })
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


# ── Prior distribution ────────────────────────────────────────────────────────

@dataclass
class Prior:
    """
    A prior distribution over a single parameter.

    Supported distributions
    -----------------------
    normal    : params mu, sigma
    lognormal : params mu (log-scale mean), sigma (log-scale std)
    beta      : params alpha, beta  (for proportions / rates in [0,1])
    gamma     : params alpha (shape), beta (rate)
    fixed     : params value  — pins the parameter to an exact value

    Parameters
    ----------
    distribution : str
        Name of the prior distribution.
    strength : float
        How strongly the prior pulls the estimate. 1.0 = equivalent to
        adding `strength * n_pseudo_obs` pseudo-observations. Higher =
        stronger regularisation. Typical range: 0.1 (weak) to 10.0 (strong).
    **params : float
        Distribution-specific parameters (see above).

    Examples
    --------
        Prior("lognormal", mu=12.1, sigma=0.7)          # loan amounts
        Prior("normal",    mu=38.0, sigma=10.0)         # DTI ratios
        Prior("beta",      alpha=2.0, beta=5.0)         # default rates
        Prior("fixed",     value=0.0)                   # pin a parameter
    """
    distribution: str
    strength: float = 1.0
    params: dict[str, float] = field(default_factory=dict)

    def __init__(self, distribution: str, strength: float = 1.0, **params):
        self.distribution = distribution.lower()
        self.strength      = strength
        self.params        = params
        self._validate()

    def _validate(self):
        valid = {"normal", "lognormal", "beta", "gamma", "fixed"}
        if self.distribution not in valid:
            raise ValueError(f"Unknown distribution '{self.distribution}'. Valid: {valid}")
        required = {
            "normal":    {"mu", "sigma"},
            "lognormal": {"mu", "sigma"},
            "beta":      {"alpha", "beta"},
            "gamma":     {"alpha", "beta"},
            "fixed":     {"value"},
        }
        missing = required[self.distribution] - set(self.params)
        if missing:
            raise ValueError(f"Prior '{self.distribution}' missing params: {missing}")

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Draw n samples from this prior."""
        rng = np.random.default_rng(seed)
        d, p = self.distribution, self.params
        if d == "normal":
            return rng.normal(p["mu"], p["sigma"], n)
        if d == "lognormal":
            return rng.lognormal(p["mu"], p["sigma"], n)
        if d == "beta":
            return rng.beta(p["alpha"], p["beta"], n)
        if d == "gamma":
            return rng.gamma(p["alpha"], 1.0 / p["beta"], n)
        if d == "fixed":
            return np.full(n, p["value"])
        raise ValueError(f"Unknown distribution: {d}")

    # ── MAP estimate (prior + data) ───────────────────────────────────────────

    def map_mean(self, data_mean: float, n_obs: int) -> float:
        """
        Compute the MAP (Maximum A Posteriori) estimate of the mean,
        blending the prior mean with the observed data mean.

        MAP = (prior_weight * prior_mean + n_obs * data_mean)
              / (prior_weight + n_obs)
        """
        prior_weight = self.strength * max(n_obs, 1) ** 0.5
        prior_mean   = self._prior_mean()
        if prior_mean is None:
            return data_mean
        return (prior_weight * prior_mean + n_obs * data_mean) / (prior_weight + n_obs)

    def map_std(self, data_std: float, n_obs: int) -> float:
        """
        MAP estimate of the standard deviation.
        Blends prior std with observed std weighted by prior strength.
        """
        prior_weight = self.strength * max(n_obs, 1) ** 0.5
        prior_std    = self._prior_std()
        if prior_std is None:
            return data_std
        return (prior_weight * prior_std + n_obs * data_std) / (prior_weight + n_obs)

    def _prior_mean(self) -> float | None:
        d, p = self.distribution, self.params
        if d == "normal":    return p["mu"]
        if d == "lognormal": return float(np.exp(p["mu"] + p["sigma"]**2 / 2))
        if d == "beta":      return p["alpha"] / (p["alpha"] + p["beta"])
        if d == "gamma":     return p["alpha"] / p["beta"]
        if d == "fixed":     return p["value"]
        return None

    def _prior_std(self) -> float | None:
        d, p = self.distribution, self.params
        if d == "normal":    return p["sigma"]
        if d == "lognormal": return float(np.sqrt((np.exp(p["sigma"]**2) - 1) * np.exp(2*p["mu"] + p["sigma"]**2)))
        if d == "beta":
            a, b = p["alpha"], p["beta"]
            return float(np.sqrt(a*b / ((a+b)**2 * (a+b+1))))
        if d == "gamma":     return float(np.sqrt(p["alpha"])) / p["beta"]
        return None

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Prior({self.distribution}, strength={self.strength}, {param_str})"


# ── Prior set ─────────────────────────────────────────────────────────────────

class PriorSet:
    """
    A collection of column-level priors for a dataset.

    Usage
    -----
        ps = PriorSet({
            "loan_amount":    Prior("lognormal", mu=12.1, sigma=0.7),
            "debt_to_income": Prior("normal", mu=38.0, sigma=10.0, strength=2.0),
        })
        # Apply to a fitted marginal model
        blended_mean = ps.map_mean("loan_amount", data_mean=250000, n_obs=50)
    """

    def __init__(self, priors: dict[str, Prior] | None = None):
        self._priors: dict[str, Prior] = priors or {}

    def add(self, column: str, prior: Prior) -> "PriorSet":
        self._priors[column] = prior
        return self

    def get(self, column: str) -> Prior | None:
        return self._priors.get(column)

    def columns(self) -> list[str]:
        return list(self._priors.keys())

    def map_mean(self, column: str, data_mean: float, n_obs: int) -> float:
        prior = self.get(column)
        if prior is None:
            return data_mean
        return prior.map_mean(data_mean, n_obs)

    def map_std(self, column: str, data_std: float, n_obs: int) -> float:
        prior = self.get(column)
        if prior is None:
            return data_std
        return prior.map_std(data_std, n_obs)

    def apply_to_params(self, column: str, params: dict, n_obs: int) -> dict:
        """
        Apply prior to a marginal model's fitted params dict.
        Blends 'loc' (mean) and 'scale' (std) toward prior values.
        """
        prior = self.get(column)
        if prior is None:
            return params
        out = dict(params)
        if "loc" in out:
            out["loc"] = prior.map_mean(out["loc"], n_obs)
        if "scale" in out:
            out["scale"] = max(prior.map_std(out["scale"], n_obs), 1e-6)
        return out

    def sample_prior_data(self, n: int = 100, seed: int = 42) -> "dict[str, np.ndarray]":
        """
        Generate pseudo-observations from all priors.
        Useful for inspecting what the prior encodes before fitting.
        """
        rng = np.random.default_rng(seed)
        return {
            col: prior.sample(n, seed=int(rng.integers(0, 10000)))
            for col, prior in self._priors.items()
        }

    def summary(self) -> list[dict]:
        """Return a summary list of all priors."""
        rows = []
        for col, prior in self._priors.items():
            rows.append({
                "column":       col,
                "distribution": prior.distribution,
                "strength":     prior.strength,
                "prior_mean":   round(prior._prior_mean() or 0, 4),
                "prior_std":    round(prior._prior_std() or 0, 4) if prior._prior_std() else None,
                **{f"param_{k}": v for k, v in prior.params.items()},
            })
        return rows

    def __repr__(self) -> str:
        return f"PriorSet({len(self._priors)} priors: {list(self._priors.keys())})"


# ── Built-in prior sets ───────────────────────────────────────────────────────
# Derived from published statistics for each data source.

DATASET_PRIORS: dict[str, PriorSet] = {

    "hmda": PriorSet({
        # CFPB HMDA 2022: median loan $280K, 5th–95th pct $80K–$650K
        "loan_amount":      Prior("lognormal", mu=12.1, sigma=0.72, strength=2.0),
        # CFPB HMDA 2022: median income $95K
        "applicant_income": Prior("lognormal", mu=11.2, sigma=0.62, strength=2.0),
        # Industry standard DTI distribution
        "debt_to_income":   Prior("normal",    mu=38.0, sigma=11.0, strength=1.5),
    }),

    "fdic": PriorSet({
        # FDIC 2023 Q4: industry avg tier1=14.7%, NIM=3.3%, ROA=1.1%
        "tier1_capital_ratio":  Prior("normal",    mu=14.7, sigma=2.5,  strength=2.0),
        "net_interest_margin":  Prior("normal",    mu=3.30, sigma=0.55, strength=2.0),
        "roa":                  Prior("normal",    mu=1.10, sigma=0.40, strength=1.5),
        "roe":                  Prior("normal",    mu=10.5, sigma=3.5,  strength=1.5),
        "npl_ratio":            Prior("lognormal", mu=-2.4, sigma=0.7,  strength=1.5),
        "loan_to_deposit":      Prior("normal",    mu=71.0, sigma=11.0, strength=1.0),
    }),

    "credit_risk": PriorSet({
        # Federal Reserve SCB 2023: base default rate ~3%, stressed ~12%
        "default_12m":       Prior("beta",      alpha=1.5, beta=30.0, strength=3.0),
        "credit_utilisation":Prior("beta",      alpha=2.0, beta=5.5,  strength=1.5),
        "debt_to_income":    Prior("normal",    mu=38.0, sigma=12.0,  strength=1.5),
        "employment_years":  Prior("gamma",     alpha=2.5, beta=0.5,  strength=1.0),
    }),

    "edgar": PriorSet({
        # S&P 500 median EBITDA margin ~18%, median net debt/EBITDA ~2.0x
        "ebitda_margin":    Prior("normal",    mu=18.0, sigma=9.0,  strength=1.5),
        "net_debt_ebitda":  Prior("normal",    mu=2.05, sigma=1.8,  strength=1.5),
        "roa":              Prior("normal",    mu=6.0,  sigma=4.0,  strength=1.0),
        "roe":              Prior("normal",    mu=14.0, sigma=9.0,  strength=1.0),
        "current_ratio":    Prior("lognormal", mu=0.45, sigma=0.38, strength=1.0),
    }),

    "cftc": PriorSet({
        # CFTC COT historical: open interest distribution is right-skewed
        "open_interest":     Prior("lognormal", mu=11.8, sigma=1.3, strength=1.5),
        "net_commercial":    Prior("normal",    mu=-18000, sigma=42000, strength=1.0),
        "net_noncommercial": Prior("normal",    mu=18000,  sigma=38000, strength=1.0),
    }),

    "fred_macro": PriorSet({
        # Fed long-run projections: GDP 2%, CPI 2%, unemployment 4%
        "gdp_growth_yoy":    Prior("normal",    mu=2.3,  sigma=1.5, strength=1.5),
        "cpi_yoy":           Prior("normal",    mu=2.8,  sigma=1.8, strength=1.5),
        "core_cpi_yoy":      Prior("normal",    mu=2.5,  sigma=1.4, strength=1.5),
        "unemployment_rate": Prior("normal",    mu=5.5,  sigma=2.0, strength=1.5),
        "fed_funds_rate":    Prior("lognormal", mu=0.85, sigma=1.0, strength=1.0),
        "vix":               Prior("lognormal", mu=3.10, sigma=0.40, strength=1.5),
        "yield_curve_spread":Prior("normal",    mu=0.90, sigma=0.90, strength=1.0),
    }),

    "bls": PriorSet({
        # BLS Q3 2023: avg weekly wage $1,168, YoY wage growth 4.3%
        "avg_weekly_wage":      Prior("lognormal", mu=6.95, sigma=0.38, strength=2.0),
        "yoy_wage_change":      Prior("normal",    mu=3.8,  sigma=2.2,  strength=1.5),
        "yoy_employment_change":Prior("normal",    mu=1.5,  sigma=4.0,  strength=1.0),
    }),

    "world_bank": PriorSet({
        # World Bank WDI 2022 global medians
        "gdp_per_capita":       Prior("lognormal", mu=8.5,  sigma=1.6, strength=1.5),
        "gdp_growth":           Prior("normal",    mu=3.0,  sigma=3.5, strength=1.5),
        "inflation":            Prior("lognormal", mu=1.6,  sigma=0.9, strength=1.5),
        "govt_debt_pct_gdp":    Prior("normal",    mu=56.0, sigma=30.0,strength=1.0),
        "gini":                 Prior("normal",    mu=37.5, sigma=8.0, strength=1.5),
    }),

    "irs_soi": PriorSet({
        # IRS SOI 2021: avg effective rate 13.3%
        "effective_rate": Prior("normal",    mu=13.3, sigma=6.0, strength=2.0),
        "total_agi":      Prior("lognormal", mu=10.9, sigma=1.0, strength=1.5),
    }),

    "census_acs": PriorSet({
        # Census ACS 2022: median HH income $74K, median rent $1,062/mo
        "household_income": Prior("lognormal", mu=10.9, sigma=0.72, strength=2.0),
        "housing_cost":     Prior("lognormal", mu=7.45, sigma=0.52, strength=2.0),
        "cost_burden_pct":  Prior("beta",      alpha=2.5, beta=5.0, strength=1.5),
    }),
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_priors(dataset_id: str) -> PriorSet:
    """
    Return the built-in PriorSet for a dataset.

    Raises ValueError if no priors are defined for the dataset.
    """
    if dataset_id not in DATASET_PRIORS:
        available = ", ".join(DATASET_PRIORS)
        raise ValueError(
            f"No built-in priors for '{dataset_id}'. "
            f"Available: {available}\n"
            f"Define custom priors with PriorSet({{col: Prior(...)}})"
        )
    return DATASET_PRIORS[dataset_id]


def blend_with_prior(
    data_value: float,
    prior: Prior,
    n_obs: int,
    mode: str = "mean",
) -> float:
    """
    Utility: blend a single data estimate with a prior.

    Parameters
    ----------
    data_value : MLE estimate from data
    prior      : Prior to blend toward
    n_obs      : number of real observations
    mode       : 'mean' or 'std'
    """
    if mode == "mean":
        return prior.map_mean(data_value, n_obs)
    elif mode == "std":
        return prior.map_std(data_value, n_obs)
    raise ValueError(f"Unknown mode '{mode}'. Use 'mean' or 'std'.")
