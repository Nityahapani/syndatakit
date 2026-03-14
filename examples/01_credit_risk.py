"""
Example 1: Credit Risk Model Training
--------------------------------------
Generate synthetic consumer credit data, train a PD (probability of default)
model on it, evaluate on real-world-like holdout, and audit privacy.

Run: python examples/01_credit_risk.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from syndatakit.generators import GaussianCopulaGenerator
from syndatakit.catalog import load_seed
from syndatakit.fidelity import fidelity_report, format_report
from syndatakit.privacy import privacy_audit, format_audit
from syndatakit.calibration.priors import get_priors


def main():
    print("=" * 60)
    print("  Credit Risk — Synthetic PD Model Training")
    print("=" * 60)

    # ── 1. Load seed and fit generator ────────────────────────────────────────
    print("\n[1/5] Fitting generator on credit_risk seed data...")
    seed = load_seed("credit_risk")
    priors = get_priors("credit_risk")

    gen = GaussianCopulaGenerator(priors=priors)
    gen.fit(seed)
    print(f"      {gen}")
    print(f"      Marginal kinds: {gen.marginal_kinds}")

    # ── 2. Generate training data ─────────────────────────────────────────────
    print("\n[2/5] Generating 10,000 synthetic training records...")
    train = gen.sample(10_000, seed=42)
    print(f"      Shape: {train.shape}")
    print(f"      Default rate: {train['default_12m'].mean():.2%}")

    # ── 3. Generate stressed data (credit crisis scenario) ────────────────────
    print("\n[3/5] Generating 2,000 credit-crisis-scenario records...")
    from syndatakit.calibration import apply_scenario
    stressed = gen.sample(2_000, seed=99)
    stressed = apply_scenario(stressed, "credit_crisis", intensity=1.0)
    print(f"      Stressed default rate: {stressed['default_12m'].mean():.2%}")

    # ── 4. Full fidelity report ───────────────────────────────────────────────
    print("\n[4/5] Running fidelity report...")
    syn_body = train.drop(columns=["syn_id"])
    report = fidelity_report(seed, syn_body, target_col="default_12m", include_downstream=True)
    s = report["summary"]
    print(f"      Overall fidelity : {s['overall_fidelity']}%")
    print(f"      Marginal score   : {s['marginal_score']}%")
    print(f"      Joint score      : {s['joint_score']}%")
    if "downstream" in report:
        d = report["downstream"]
        print(f"      TSTR Gini        : {d['tstr_score']}  (TRR: {d['trr_score']})  ratio: {d['ratio']}")

    # ── 5. Privacy audit ──────────────────────────────────────────────────────
    print("\n[5/5] Running privacy audit (200 attacks)...")
    audit = privacy_audit(seed, syn_body, n_attacks=200, seed=42)
    v = audit["verdict"]
    print(f"      Overall risk     : {v['overall_risk']}")
    print(f"      Exact copies     : {v['exact_copies']}")
    print(f"      MI AUC           : {v['mi_auc']}")
    print(f"      Recommendation   : {v['recommendation']}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    from syndatakit.io import write
    write(train,   "examples/output_credit_train.csv")
    write(stressed, "examples/output_credit_stressed.csv")
    print(f"\n  Saved: examples/output_credit_train.csv   ({len(train):,} rows)")
    print(f"         examples/output_credit_stressed.csv ({len(stressed):,} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
