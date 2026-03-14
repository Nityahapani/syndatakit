"""
Example 4: REST API Usage
--------------------------
Demonstrates every API endpoint using Python requests.
Start the server first: syndatakit serve --port 8080

Run: python examples/04_api_usage.py
"""
import sys
import json
import time

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8080"


def check_server():
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        j = r.json()
        print(f"  Server OK — v{j['version']}  {j['datasets']} datasets  {j['generators_cached']} cached")
        return True
    except Exception:
        print(f"  Server not running. Start it with: syndatakit serve --port 8080")
        return False


def main():
    print("=" * 60)
    print("  syndatakit REST API — Usage Examples")
    print("=" * 60)

    print("\n[0] Checking server...")
    if not check_server():
        return

    # ── List datasets ─────────────────────────────────────────────────────────
    print("\n[1] GET /datasets — all datasets")
    r = requests.get(f"{BASE}/datasets")
    datasets = r.json()["data"]
    for d in datasets:
        print(f"    {d['id']:<15} {d['vertical']:<25} fidelity={d['fidelity']}")

    # ── Filter by vertical ────────────────────────────────────────────────────
    print("\n[2] GET /datasets?vertical=Capital+Markets")
    r = requests.get(f"{BASE}/datasets", params={"vertical": "Capital Markets"})
    j = r.json()
    print(f"    {j['meta']['count']} datasets in Capital Markets: {[d['id'] for d in j['data']]}")

    # ── Sample ────────────────────────────────────────────────────────────────
    print("\n[3] GET /datasets/edgar/sample?rows=3")
    r = requests.get(f"{BASE}/datasets/edgar/sample", params={"rows": 3})
    rows = r.json()["data"]
    print(f"    {len(rows)} rows, columns: {list(rows[0].keys())[:5]}...")

    # ── Generate basic ────────────────────────────────────────────────────────
    print("\n[4] POST /generate — 500 HMDA rows with filter")
    t0 = time.time()
    r = requests.post(f"{BASE}/generate", json={
        "dataset": "hmda",
        "rows":    500,
        "filters": {"state": ["CA", "TX"], "dti_min": 40},
        "seed":    42,
    })
    j = r.json()
    m = j["meta"]
    print(f"    Generated: {m['rows_generated']} rows in {m['generation_ms']}ms")
    print(f"    States:    {set(row['state'] for row in j['data'][:10])}")

    # ── Generate with scenario ────────────────────────────────────────────────
    print("\n[5] POST /generate — fred_macro with recession scenario")
    r = requests.post(f"{BASE}/generate", json={
        "dataset":   "fred_macro",
        "rows":      200,
        "scenario":  "recession",
        "intensity": 0.9,
        "seed":      1,
    })
    j = r.json()
    rows = j["data"]
    gdp_mean = sum(row["gdp_growth_yoy"] for row in rows) / len(rows)
    print(f"    Rows: {j['meta']['rows_generated']}  scenario={j['meta']['scenario']}")
    print(f"    Mean GDP growth: {gdp_mean:.2f}%  (recession = negative expected)")

    # ── Generate CSV ──────────────────────────────────────────────────────────
    print("\n[6] POST /generate — CSV format download")
    r = requests.post(f"{BASE}/generate", json={
        "dataset": "credit_risk",
        "rows":    100,
        "format":  "csv",
        "seed":    7,
    })
    lines = r.text.strip().split("\n")
    print(f"    CSV: {len(lines)} lines  header: {lines[0][:70]}...")

    # ── Scenarios ─────────────────────────────────────────────────────────────
    print("\n[7] GET /scenarios")
    r = requests.get(f"{BASE}/scenarios")
    for s in r.json()["data"]:
        print(f"    {s['name']:<20} {s['description']}")

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\n[8] POST /validate — check real data before fitting")
    import pandas as pd
    sys.path.insert(0, ".")
    from syndatakit.catalog import load_seed
    seed = load_seed("hmda").head(200)
    r = requests.post(f"{BASE}/validate", json={"data": seed.to_dict(orient="records")})
    result = r.json()["data"]
    print(f"    passed={result['passed']}  errors={len(result['errors'])}  warnings={len(result['warnings'])}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[9] POST /evaluate — fidelity report")
    from syndatakit.generators import GaussianCopulaGenerator
    full_seed = load_seed("hmda")
    gen = GaussianCopulaGenerator(); gen.fit(full_seed)
    syn = gen.sample(300, seed=42).drop(columns=["syn_id"])

    r = requests.post(f"{BASE}/evaluate", json={
        "dataset":   "hmda",
        "type":      "cross_sectional",
        "real":      full_seed.head(300).to_dict(orient="records"),
        "synthetic": syn.to_dict(orient="records"),
    })
    s = r.json()["data"]["summary"]
    print(f"    Overall fidelity: {s['overall_fidelity']}%")
    print(f"    Marginal score:   {s['marginal_score']}%")
    print(f"    Privacy score:    {s['privacy_score']}%")

    # ── Audit ─────────────────────────────────────────────────────────────────
    print("\n[10] POST /audit — privacy audit")
    r = requests.post(f"{BASE}/audit", json={
        "real":      full_seed.head(300).to_dict(orient="records"),
        "synthetic": syn.to_dict(orient="records"),
        "attacks":   150,
    })
    v = r.json()["data"]["verdict"]
    print(f"    Overall risk: {v['overall_risk']}")
    print(f"    Exact copies: {v['exact_copies']}")
    print(f"    Recommendation: {v['recommendation'][:60]}...")

    print("\nAll API examples complete.")


if __name__ == "__main__":
    main()
