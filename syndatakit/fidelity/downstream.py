"""
syndatakit.fidelity.downstream
--------------------------------
Train-on-Synthetic, Test-on-Real (TSTR) utility evaluation.

The gold standard for synthetic data quality: train a model on synthetic
data, evaluate on held-out real data. Compare to Train-on-Real (TRR).

TSTR score = TSTR_metric / TRR_metric  (ratio, ideally close to 1.0)

Supports: classification (default_12m, action_taken) and regression tasks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient from sorted predictions (proxy for AUC)."""
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    n = len(y_sorted)
    cumulative = np.cumsum(y_sorted)
    lorenz = cumulative / (cumulative[-1] + 1e-9)
    gini = (lorenz.sum() / n) - 0.5
    return float(2 * gini)


def _simple_logreg(X_train, y_train, X_test) -> np.ndarray:
    """Minimal logistic regression via gradient descent (no sklearn needed)."""
    X = np.column_stack([np.ones(len(X_train)), X_train])
    Xt = np.column_stack([np.ones(len(X_test)), X_test])
    w = np.zeros(X.shape[1])
    lr = 0.01
    for _ in range(200):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -10, 10)))
        grad = X.T @ (p - y_train) / len(y_train)
        w -= lr * grad
    return 1 / (1 + np.exp(-np.clip(Xt @ w, -10, 10)))


def _simple_linreg_r2(X_train, y_train, X_test, y_test) -> float:
    """R² from OLS."""
    X = np.column_stack([np.ones(len(X_train)), X_train])
    Xt = np.column_stack([np.ones(len(X_test)), X_test])
    try:
        b = np.linalg.lstsq(X, y_train, rcond=None)[0]
        pred = Xt @ b
        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        return float(1 - ss_res / max(ss_tot, 1e-9))
    except Exception:
        return 0.0


def tstr_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task: str = "auto",
    test_frac: float = 0.3,
    seed: int = 42,
) -> dict:
    """
    Compute TSTR score: train on synthetic, evaluate on real.

    Parameters
    ----------
    real, synthetic : DataFrames (must share columns)
    target_col      : column to predict
    feature_cols    : predictor columns (default: all shared numeric cols except target)
    task            : 'classification', 'regression', or 'auto'
    test_frac       : fraction of real data held out for evaluation

    Returns
    -------
    dict with tstr_score, trr_score, ratio (TSTR/TRR), and task type
    """
    all_cols = [c for c in real.columns if c in synthetic.columns]
    if feature_cols is None:
        feature_cols = [
            c for c in all_cols
            if c != target_col and pd.api.types.is_numeric_dtype(real[c])
        ]

    if not feature_cols or target_col not in real.columns:
        return {"error": f"target '{target_col}' not found or no numeric features"}

    # Infer task
    if task == "auto":
        n_unique = real[target_col].nunique()
        task = "classification" if n_unique <= 10 else "regression"

    # Prepare data
    rng = np.random.default_rng(seed)
    real_clean = real[feature_cols + [target_col]].dropna()
    syn_clean  = synthetic[feature_cols + [target_col]].dropna()

    # Split real into train/test
    idx = rng.permutation(len(real_clean))
    split = int(len(real_clean) * (1 - test_frac))
    real_train = real_clean.iloc[idx[:split]]
    real_test  = real_clean.iloc[idx[split:]]

    def to_arrays(df):
        X = df[feature_cols].values.astype(float)
        y = df[target_col].values.astype(float)
        # Normalise features
        mu, sd = X.mean(0), X.std(0) + 1e-9
        return (X - mu) / sd, y, mu, sd

    X_real_tr, y_real_tr, mu, sd = to_arrays(real_train)
    X_test_raw = real_test[feature_cols].values.astype(float)
    X_test = (X_test_raw - mu) / sd
    y_test = real_test[target_col].values.astype(float)

    X_syn = (syn_clean[feature_cols].values.astype(float) - mu) / sd
    y_syn = syn_clean[target_col].values.astype(float)

    if task == "classification":
        y_real_tr_b = (y_real_tr > y_real_tr.mean()).astype(float)
        y_syn_b     = (y_syn     > y_real_tr.mean()).astype(float)
        y_test_b    = (y_test    > y_real_tr.mean()).astype(float)

        p_tstr = _simple_logreg(X_syn,     y_syn_b,     X_test)
        p_trr  = _simple_logreg(X_real_tr, y_real_tr_b, X_test)

        tstr = _gini(y_test_b, p_tstr)
        trr  = _gini(y_test_b, p_trr)
        metric = "gini"
    else:
        tstr = _simple_linreg_r2(X_syn,     y_syn,     X_test, y_test)
        trr  = _simple_linreg_r2(X_real_tr, y_real_tr, X_test, y_test)
        metric = "r2"

    ratio = round(tstr / max(abs(trr), 1e-6), 4)

    return {
        "task":       task,
        "metric":     metric,
        "tstr_score": round(tstr, 4),
        "trr_score":  round(trr, 4),
        "ratio":      ratio,
        "target_col": target_col,
        "n_features": len(feature_cols),
        "n_synthetic_train": len(syn_clean),
        "n_real_test":       len(real_test),
        "interpretation": (
            "TSTR ≈ TRR: synthetic data is a good substitute for real data"
            if 0.8 <= ratio <= 1.2 else
            "TSTR < TRR: synthetic data loses predictive signal — check generator quality"
            if ratio < 0.8 else
            "TSTR > TRR: synthetic data may be overfit to training distribution"
        ),
    }
