"""
Example 3: Privacy Audit Walkthrough
--------------------------------------
Demonstrates the full privacy audit workflow:
- Membership inference attack
- Singling-out risk
- Linkability risk
- Differential privacy noise addition
- Interpreting risk levels and recommendations

Run: python examples/03_privacy_audit.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from syndatakit.generators import GaussianCopulaGenerator
from syndatakit.catalog import load_seed
from syndatakit.privacy import privacy_audit, format_audit
from syndatakit.privacy.dp import PrivacyBudget, laplace_mechanism, gaussian_mechanism
from syndatakit.privacy.singling_out import singling_out_risk
from syndatakit.privacy.linkability import linkability_risk
from syndatakit.privacy.disclosure import membership_inference_risk


def main():
    print("=" * 60)
    print("  Privacy Audit — HMDA Mortgage Data")
    print("=" * 60)

    # ── 1. Generate synthetic data ────────────────────────────────────────────
    print("\n[1/4] Generating synthetic HMDA data...")
    seed = load_seed("hmda")
    gen = GaussianCopulaGenerator()
    gen.fit(seed)
    syn = gen.sample(1000, seed=42)
    syn_body = syn.drop(columns=["syn_id"])
    print(f"      Real rows: {len(seed):,}  |  Synthetic rows: {len(syn_body):,}")

    # ── 2. Full audit ─────────────────────────────────────────────────────────
    print("\n[2/4] Running full privacy audit (300 attacks per test)...")
    audit = privacy_audit(seed, syn_body, n_attacks=300, seed=42)
    print(format_audit(audit))

    # ── 3. Individual test deep-dives ─────────────────────────────────────────
    print("\n[3/4] Individual test results:")

    # Membership inference
    mi = audit["membership_inference"]
    print(f"\n  Membership Inference:")
    print(f"    Attack AUC     : {mi['attack_auc']}  (0.5 = random, 1.0 = perfect attack)")
    print(f"    Advantage      : {mi['advantage']}  (AUC - 0.5)")
    print(f"    Risk level     : {mi['risk_level']}")
    print(f"    Interpretation : {mi['interpretation']}")

    # Singling-out
    so = audit["singling_out"]
    print(f"\n  Singling-Out:")
    print(f"    Rate           : {so['singling_out_rate']}  (fraction of attacks that uniquely identify)")
    print(f"    Risk level     : {so['risk_level']}")
    print(f"    QI columns used: {so.get('quasi_id_cols', [])}")

    # Linkability
    lk = audit["linkability"]
    print(f"\n  Linkability:")
    print(f"    Rate           : {lk['linkability_rate']}  (0.5 = random baseline)")
    print(f"    Lift           : {lk['lift_over_baseline_pct']}% over baseline")
    print(f"    Risk level     : {lk['risk_level']}")

    # ── 4. Differential privacy demo ─────────────────────────────────────────
    print("\n[4/4] Differential privacy — protecting aggregate statistics:")
    budget = PrivacyBudget(epsilon=1.0)
    print(f"      Budget: {budget}")

    true_mean_loan  = float(seed["loan_amount"].mean())
    true_mean_income = float(seed["applicant_income"].mean())

    noisy_loan = laplace_mechanism(
        true_mean_loan, sensitivity=2_000_000, epsilon=0.3,
        budget=budget, label="mean_loan_amount",
    )
    noisy_income = laplace_mechanism(
        true_mean_income, sensitivity=1_000_000, epsilon=0.3,
        budget=budget, label="mean_applicant_income",
    )

    print(f"\n      Loan amount:   true={true_mean_loan:>12,.0f}  noisy={noisy_loan:>12,.0f}")
    print(f"      App. income:   true={true_mean_income:>12,.0f}  noisy={noisy_income:>12,.0f}")
    print(f"\n      Budget log:")
    for entry in budget.log:
        print(f"        ε={entry['epsilon']}  [{entry['label']}]")
    print(f"      Remaining ε: {budget.remaining_epsilon:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
