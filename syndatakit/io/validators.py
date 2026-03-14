"""
syndatakit.io.validators
--------------------------
Schema validation for real data before it's passed to a generator.

Catches common problems early:
  - Missing required columns
  - Wrong dtypes
  - Nulls above threshold
  - Constant columns (zero variance)
  - Extreme cardinality in categorical columns
  - Duplicate rows above threshold
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    passed: bool
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats:    dict      = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["ValidationResult:"]
        lines.append(f"  Passed  : {self.passed}")
        if self.errors:
            lines.append(f"  Errors  : {len(self.errors)}")
            for e in self.errors:
                lines.append(f"    ✗ {e}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        return "\n".join(lines)


def validate(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    expected_dtypes: dict[str, str] | None = None,
    null_threshold: float = 0.3,
    duplicate_threshold: float = 0.05,
    max_cardinality: int = 500,
    min_rows: int = 50,
) -> ValidationResult:
    """
    Validate a real DataFrame before generator fitting.

    Parameters
    ----------
    df                  : DataFrame to validate
    required_columns    : columns that must be present
    expected_dtypes     : {col: 'numeric' | 'categorical'} type expectations
    null_threshold      : maximum allowed null fraction per column (default 0.3)
    duplicate_threshold : maximum allowed duplicate row fraction (default 0.05)
    max_cardinality     : max unique values in a categorical column (default 500)
    min_rows            : minimum required rows (default 50)

    Returns
    -------
    ValidationResult with passed flag, errors, warnings, and summary stats.
    """
    errors:   list[str] = []
    warnings: list[str] = []
    stats:    dict      = {}

    # ── Basic shape ───────────────────────────────────────────────────────────
    n_rows, n_cols = df.shape
    stats["rows"] = n_rows
    stats["cols"] = n_cols

    if n_rows < min_rows:
        errors.append(f"Too few rows: {n_rows} (minimum {min_rows})")

    if n_cols == 0:
        errors.append("DataFrame has no columns")
        return ValidationResult(passed=False, errors=errors, stats=stats)

    # ── Required columns ──────────────────────────────────────────────────────
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

    # ── Dtype expectations ────────────────────────────────────────────────────
    if expected_dtypes:
        for col, expected in expected_dtypes.items():
            if col not in df.columns:
                continue
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if expected == "numeric" and not is_numeric:
                errors.append(f"Column '{col}' expected numeric, got {df[col].dtype}")
            elif expected == "categorical" and is_numeric:
                warnings.append(f"Column '{col}' expected categorical but appears numeric")

    # ── Per-column checks ─────────────────────────────────────────────────────
    col_stats: dict[str, dict] = {}

    for col in df.columns:
        s = df[col]
        null_frac = float(s.isna().mean())
        n_unique  = int(s.nunique())
        is_const  = n_unique <= 1

        col_stats[col] = {
            "null_frac": round(null_frac, 4),
            "n_unique":  n_unique,
            "dtype":     str(s.dtype),
        }

        if null_frac > null_threshold:
            errors.append(
                f"Column '{col}' has {null_frac:.1%} nulls (threshold: {null_threshold:.1%})"
            )

        if is_const:
            warnings.append(f"Column '{col}' is constant (all values = {s.dropna().iloc[0] if len(s.dropna()) else 'NA'})")

        if not pd.api.types.is_numeric_dtype(s) and n_unique > max_cardinality:
            warnings.append(
                f"Column '{col}' has very high cardinality ({n_unique} unique values). "
                f"Consider encoding or dropping."
            )

    stats["columns"] = col_stats

    # ── Duplicate rows ────────────────────────────────────────────────────────
    n_dupes = int(df.duplicated().sum())
    dupe_frac = n_dupes / max(n_rows, 1)
    stats["duplicate_rows"] = n_dupes
    stats["duplicate_frac"] = round(dupe_frac, 4)

    if dupe_frac > duplicate_threshold:
        warnings.append(
            f"{n_dupes} duplicate rows ({dupe_frac:.1%}) — "
            f"this may inflate generator memorisation."
        )

    # ── Numeric sanity ────────────────────────────────────────────────────────
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols:
        s = df[col].dropna().astype(float)
        if len(s) < 4:
            continue
        if not np.all(np.isfinite(s)):
            errors.append(f"Column '{col}' contains Inf or NaN values after dropna")

    passed = len(errors) == 0
    return ValidationResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
