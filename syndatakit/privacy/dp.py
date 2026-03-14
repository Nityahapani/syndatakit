"""
syndatakit.privacy.dp
----------------------
Differential Privacy (ε-DP) noise addition and budget tracking.

Provides Laplace and Gaussian mechanisms for adding calibrated noise
to aggregate statistics or model parameters, with ε budget accounting.

Note: This applies DP to the *fitting process* (noise on statistics),
not post-hoc to synthetic rows. For a rigorous DP guarantee the entire
generation pipeline must use these mechanisms throughout.
"""
from __future__ import annotations
import numpy as np


class PrivacyBudget:
    """Tracks consumed ε (epsilon) and δ (delta) across mechanisms."""

    def __init__(self, epsilon: float, delta: float = 0.0):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.delta   = delta
        self._used_epsilon = 0.0
        self._used_delta   = 0.0
        self._log: list[dict] = []

    def consume(self, eps: float, delta: float = 0.0, label: str = "") -> None:
        self._used_epsilon += eps
        self._used_delta   += delta
        self._log.append({"label": label, "epsilon": eps, "delta": delta})
        if self._used_epsilon > self.epsilon:
            raise RuntimeError(
                f"Privacy budget exhausted: used {self._used_epsilon:.4f} "
                f"of {self.epsilon:.4f} ε"
            )

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.epsilon - self._used_epsilon)

    @property
    def log(self) -> list[dict]:
        return list(self._log)

    def __repr__(self) -> str:
        return (
            f"PrivacyBudget(ε={self.epsilon}, used={self._used_epsilon:.4f}, "
            f"remaining={self.remaining_epsilon:.4f})"
        )


def laplace_mechanism(
    value: float | np.ndarray,
    sensitivity: float,
    epsilon: float,
    budget: PrivacyBudget | None = None,
    label: str = "laplace",
    seed: int | None = None,
) -> float | np.ndarray:
    """
    Add Laplace noise calibrated to (sensitivity / epsilon).

    Parameters
    ----------
    value       : true statistic(s) to protect
    sensitivity : L1 sensitivity of the query
    epsilon     : privacy parameter for this release
    budget      : if provided, consume epsilon from budget
    """
    rng = np.random.default_rng(seed)
    scale = sensitivity / epsilon
    noise = rng.laplace(0, scale, size=np.asarray(value).shape or None)
    if budget is not None:
        budget.consume(epsilon, label=label)
    return value + noise


def gaussian_mechanism(
    value: float | np.ndarray,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
    budget: PrivacyBudget | None = None,
    label: str = "gaussian",
    seed: int | None = None,
) -> float | np.ndarray:
    """
    Add Gaussian noise for (ε, δ)-DP.
    σ = sensitivity * sqrt(2 ln(1.25/δ)) / ε
    """
    rng = np.random.default_rng(seed)
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = rng.normal(0, sigma, size=np.asarray(value).shape or None)
    if budget is not None:
        budget.consume(epsilon, delta=delta, label=label)
    return value + noise


def privatise_histogram(
    counts: dict,
    epsilon: float,
    budget: PrivacyBudget | None = None,
    seed: int | None = None,
) -> dict:
    """
    Add Laplace noise to a frequency histogram (sensitivity = 1 for count queries).
    Negative counts are clipped to 0 and re-normalised.
    """
    rng = np.random.default_rng(seed)
    keys = list(counts.keys())
    vals = np.array([counts[k] for k in keys], dtype=float)
    scale = 1.0 / epsilon
    noisy = vals + rng.laplace(0, scale, size=len(vals))
    noisy = np.maximum(noisy, 0)
    total = noisy.sum()
    if total > 0:
        noisy /= total
    if budget is not None:
        budget.consume(epsilon, label="histogram")
    return dict(zip(keys, noisy.tolist()))
