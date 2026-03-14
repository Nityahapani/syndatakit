"""
syndatakit v2 CLI
-----------------
Commands:
    syndatakit list                         List datasets (with vertical filter)
    syndatakit info <dataset>              Full dataset metadata
    syndatakit generate <dataset>          Generate synthetic data
    syndatakit evaluate <real> <syn>       Full fidelity report
    syndatakit audit <real> <syn>          Full privacy audit
    syndatakit scenario list               List built-in scenarios
    syndatakit scenario apply <scenario>   Apply scenario to a CSV
    syndatakit validate <file>             Validate a real data file
    syndatakit serve                       Start REST API
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path


# ── Terminal colours ──────────────────────────────────────────────────────────

class C:
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GRAY   = "\033[90m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

def _c(t, c):  return f"{c}{t}{C.RESET}"
def ok(m):     print(_c("  ✓ ", C.GREEN)  + m)
def info(m):   print(_c("  → ", C.CYAN)   + m)
def warn(m):   print(_c("  ! ", C.YELLOW) + m)
def err(m):    print(_c("  ✗ ", C.RED)    + m, file=sys.stderr)
def dim(m):    print(_c(m, C.GRAY))
def bold(m):   print(_c(m, C.BOLD))

def header(title, sub=""):
    print()
    print(_c("  " + title, C.BOLD))
    if sub:
        print(_c("  " + sub, C.GRAY))
    print(_c("  " + "─" * max(len(title), len(sub)), C.GRAY))

def section(title):
    print()
    print(_c("  ┌─ " + title, C.CYAN))

def bar(score, width=22):
    score = float(score)
    filled = int(score / 100 * width)
    col = C.GREEN if score >= 90 else C.YELLOW if score >= 75 else C.RED
    return _c("█" * filled, col) + _c("░" * (width - filled), C.GRAY)

def risk_colour(level):
    return {
        "very_low":  _c(level, C.GREEN),
        "low":       _c(level, C.GREEN),
        "medium":    _c(level, C.YELLOW),
        "high":      _c(level, C.RED),
        "very_high": _c(level, C.RED),
    }.get(level, level)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_filters(filter_args):
    filters = {}
    if not filter_args:
        return filters
    for f in filter_args:
        if ":" not in f:
            warn(f"Skipping malformed filter '{f}' — use key:value")
            continue
        key, val = f.split(":", 1)
        key = key.strip().lower().replace("-", "_")
        if key.endswith("_min") or key.endswith("_max"):
            try:
                filters[key] = float(val)
            except ValueError:
                warn(f"Filter '{key}' expects a number, got '{val}'")
        else:
            filters[key] = val.split(",") if "," in val else val
    return filters


def _load_generator(dataset_id, generator_type="auto"):
    """Load and fit a generator for the given dataset."""
    from syndatakit.catalog import load_seed, get_dataset_info
    from syndatakit.generators import GaussianCopulaGenerator
    from syndatakit.generators.time_series import VARGenerator
    from syndatakit.generators.panel import FixedEffectsGenerator

    meta = get_dataset_info(dataset_id)
    seed_df = load_seed(dataset_id)

    # Auto-detect generator type from dataset metadata
    if generator_type == "auto":
        time_series_datasets = {"fred_macro", "bls"}
        panel_datasets        = {"world_bank", "fdic"}
        if dataset_id in time_series_datasets:
            generator_type = "var"
        elif dataset_id in panel_datasets:
            generator_type = "panel"
        else:
            generator_type = "copula"

    if generator_type == "var":
        time_col = "year" if "year" in seed_df.columns else None
        gen = VARGenerator(lags=2, time_col=time_col)
    elif generator_type == "panel":
        entity_col = next((c for c in ["country_code", "bank_id", "entity"] if c in seed_df.columns), seed_df.columns[0])
        time_col   = "year" if "year" in seed_df.columns else "quarter" if "quarter" in seed_df.columns else None
        gen = FixedEffectsGenerator(entity_col=entity_col, time_col=time_col)
    else:
        gen = GaussianCopulaGenerator()

    gen.fit(seed_df)
    return gen, seed_df, generator_type


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(args):
    from syndatakit.catalog import list_datasets

    header("Available datasets", "finance & econometrics · 10 datasets · 4 verticals")
    df = list_datasets(vertical=args.vertical if hasattr(args, "vertical") else None)

    for vertical in df["vertical"].unique():
        sub = df[df["vertical"] == vertical]
        print(f"\n  {_c(vertical.upper(), C.CYAN)}")
        for _, row in sub.iterrows():
            fid = _c(row["fidelity"], C.GREEN)
            print(f"    {_c(row['id'], C.BOLD):<28} {row['name']:<40} {fid}")

    print()
    dim("  syndatakit info <id>           full metadata + columns")
    dim("  syndatakit generate <id>       generate synthetic data")
    dim("  syndatakit serve               start REST API at :5000")
    print()


def cmd_info(args):
    from syndatakit.catalog import get_dataset_info

    try:
        meta = get_dataset_info(args.dataset)
    except ValueError as e:
        err(str(e)); sys.exit(1)

    header(f"Dataset: {args.dataset}", meta["name"])

    pairs = [
        ("Vertical",  meta["vertical"]),
        ("Source",    meta["source"]),
        ("Columns",   str(meta["col_count"])),
        ("Fidelity",  f"{meta['fidelity']}%"),
        ("Status",    meta["status"]),
        ("Tags",      ", ".join(meta["tags"])),
    ]
    for label, value in pairs:
        print(f"    {_c(label+':', C.GRAY):<22}{value}")

    print(f"\n    {_c('Columns:', C.GRAY)}")
    cols = meta["columns"]
    for i in range(0, len(cols), 4):
        print("    " + "  ".join(_c(c, C.CYAN) for c in cols[i:i+4]))

    print(f"\n    {_c('Use cases:', C.GRAY)}")
    for uc in meta["use_cases"]:
        print(f"    · {uc}")

    print(f"\n    {_c('Description:', C.GRAY)}")
    print(f"    {meta['description']}")
    print()


def cmd_generate(args):
    header(f"Generating: {args.dataset}", f"generator={args.generator}  rows={args.rows:,}")

    filters = _parse_filters(getattr(args, "filter", None))
    if filters:
        info(f"Filters: {filters}")

    # Load + fit generator
    t0 = time.time()
    print()
    info("Fitting generator...")
    try:
        gen, seed_df, gen_type = _load_generator(args.dataset, args.generator)
    except ValueError as e:
        err(str(e)); sys.exit(1)
    ok(f"{gen}  [{gen_type}]  ({time.time()-t0:.1f}s)")

    # Generate
    info(f"Sampling {args.rows:,} rows...")
    t1 = time.time()
    syn = gen.sample(args.rows, filters=filters or None, seed=args.seed)
    ok(f"{len(syn):,} rows generated  ({time.time()-t1:.1f}s)")

    # Apply scenario if requested
    if getattr(args, "scenario", None):
        from syndatakit.calibration import apply_scenario
        info(f"Applying scenario: {args.scenario}  (intensity={args.intensity})")
        syn = apply_scenario(syn, args.scenario, intensity=args.intensity)
        ok(f"Scenario applied")

    # Moment calibration
    if getattr(args, "calibrate", False):
        from syndatakit.calibration import match_moments
        info("Running moment calibration...")
        syn_body = syn.drop(columns=["syn_id"], errors="ignore")
        syn_body = match_moments(seed_df, syn_body)
        syn_body.insert(0, "syn_id", syn["syn_id"])
        syn = syn_body
        ok("Moments calibrated")

    # Fidelity evaluation
    if not getattr(args, "no_eval", False):
        from syndatakit.fidelity import fidelity_report

        info("Running fidelity report...")
        dataset_type = {"var": "time_series", "panel": "panel"}.get(gen_type, "cross_sectional")
        syn_body = syn.drop(columns=["syn_id"], errors="ignore")
        report = fidelity_report(seed_df, syn_body, dataset_type=dataset_type)
        s = report["summary"]

        section("Fidelity report")
        for col, score in report["marginal"]["column_scores"].items():
            print(f"    {col:<26}{bar(score)}  {_c(str(score)+'%', C.GREEN if score>=90 else C.YELLOW)}")
        print()
        print(f"    {_c('Marginal score:',  C.GRAY):<34}{_c(str(s['marginal_score'])+'%',  C.GREEN)}")
        print(f"    {_c('Joint score:',     C.GRAY):<34}{_c(str(s['joint_score'])+'%',     C.GREEN)}")
        print(f"    {_c('Overall fidelity:',C.GRAY):<34}{_c(str(s['overall_fidelity'])+'%',C.GREEN)}")
        print(f"    {_c('Privacy score:',   C.GRAY):<34}{_c(str(s['privacy_score'])+'%',   C.GREEN)}")
        print(f"    {_c('Exact copies:',    C.GRAY):<34}{s['exact_copies']}")

        if "temporal" in report:
            t = report["temporal"]
            section("Temporal fidelity")
            print(f"    Stationarity agreement  {t['stationarity']['_summary']['agreement_rate']}%")
            print(f"    Cointegration agreement {t['cointegration']['_summary']['agreement_rate']}%")
            print(f"    Break match rate        {t['breaks']['_summary']['break_match_rate']}%")
            print(f"    Causality agreement     {t['causality']['_summary']['agreement_rate']}%")

    # Save output
    from syndatakit.io import write
    output = Path(args.output)
    write(syn, output)
    print()
    ok(f"Saved → {_c(str(output), C.CYAN)}  ({output.stat().st_size // 1024} KB)")
    print()


def cmd_evaluate(args):
    import pandas as pd
    from syndatakit.fidelity import fidelity_report, format_report
    from syndatakit.catalog import DATASETS

    header("Fidelity evaluation", f"{args.real}  vs  {args.synthetic}")

    for p in [Path(args.real), Path(args.synthetic)]:
        if not p.exists():
            err(f"File not found: {p}"); sys.exit(1)

    from syndatakit.io import read
    info(f"Loading real:      {args.real}")
    real = read(args.real)
    info(f"Loading synthetic: {args.synthetic}")
    syn  = read(args.synthetic)
    info(f"Rows — real: {len(real):,}  synthetic: {len(syn):,}")

    dataset_type = getattr(args, "type", "cross_sectional") or "cross_sectional"
    target_col   = getattr(args, "target", None)

    info("Running full fidelity report...")
    report = fidelity_report(
        real, syn,
        dataset_type=dataset_type,
        target_col=target_col,
        include_downstream=bool(target_col),
    )

    print(format_report(report))

    if getattr(args, "json", False):
        import json as _json
        # Make report JSON-serialisable
        def _clean(obj):
            if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):  return [_clean(v) for v in obj]
            if isinstance(obj, float): return round(obj, 6)
            return obj
        print(_json.dumps(_clean(report), indent=2))

    if getattr(args, "output", None):
        import json as _json
        Path(args.output).write_text(_json.dumps(report, indent=2, default=str))
        ok(f"Report saved → {args.output}")


def cmd_audit(args):
    header("Privacy audit", f"{args.real}  vs  {args.synthetic}")

    for p in [Path(args.real), Path(args.synthetic)]:
        if not p.exists():
            err(f"File not found: {p}"); sys.exit(1)

    from syndatakit.io import read
    from syndatakit.privacy import privacy_audit, format_audit

    info(f"Loading real:      {args.real}")
    real = read(args.real)
    info(f"Loading synthetic: {args.synthetic}")
    syn  = read(args.synthetic)
    info(f"Rows — real: {len(real):,}  synthetic: {len(syn):,}")
    info(f"Running {args.attacks} attacks per test...")
    print()

    report = privacy_audit(
        real, syn,
        n_attacks=args.attacks,
        seed=42,
    )

    print(format_audit(report))

    v = report["verdict"]
    section("Detailed results")
    ec = report["exact_copies"]
    mi = report["membership_inference"]
    so = report["singling_out"]
    lk = report["linkability"]

    print(f"    {'Exact copies':<28}{ec['count']}  [{risk_colour(ec['risk_level'])}]")
    print(f"    {'Membership inference AUC':<28}{mi.get('attack_auc','—')}  [{risk_colour(mi.get('risk_level','—'))}]")
    print(f"      {_c(mi.get('interpretation',''), C.GRAY)}")
    print(f"    {'Singling-out rate':<28}{so.get('singling_out_rate','—')}  [{risk_colour(so.get('risk_level','—'))}]")
    print(f"    {'Linkability rate':<28}{lk.get('linkability_rate','—')}  [{risk_colour(lk.get('risk_level','—'))}]")
    print(f"      lift {lk.get('lift_over_baseline_pct','—')}% over baseline")
    print()
    print(f"    {_c('Recommendation:', C.BOLD)}")
    print(f"    {v['recommendation']}")

    if getattr(args, "json", False):
        import json as _json
        print(_json.dumps(report, indent=2, default=str))

    if getattr(args, "output", None):
        import json as _json
        Path(args.output).write_text(_json.dumps(report, indent=2, default=str))
        ok(f"Audit saved → {args.output}")
    print()


def cmd_scenario_list(args):
    from syndatakit.calibration import list_scenarios
    header("Built-in scenarios")
    df = list_scenarios()
    for _, row in df.iterrows():
        print(f"\n    {_c(row['name'], C.BOLD)}")
        print(f"    {_c(row['description'], C.GRAY)}")
        print(f"    {row['columns_affected']} columns affected")
    print()
    dim("  Usage: syndatakit scenario apply <name> --input data.csv --output stressed.csv")
    print()


def cmd_scenario_apply(args):
    from syndatakit.calibration import apply_scenario
    from syndatakit.io import read, write

    header(f"Applying scenario: {args.scenario}", f"intensity={args.intensity}")

    if not Path(args.input).exists():
        err(f"File not found: {args.input}"); sys.exit(1)

    info(f"Loading: {args.input}")
    df = read(args.input)
    info(f"Rows: {len(df):,}  columns: {len(df.columns)}")

    try:
        result = apply_scenario(df, args.scenario, intensity=args.intensity)
    except ValueError as e:
        err(str(e)); sys.exit(1)

    output = Path(args.output)
    write(result, output)
    ok(f"Saved → {_c(str(output), C.CYAN)}")

    # Show which columns shifted
    num_cols = [c for c in df.columns if df[c].dtype.kind in "if" and c in result.columns]
    if num_cols:
        section("Column shifts")
        for col in num_cols[:8]:
            before = df[col].mean()
            after  = result[col].mean()
            delta  = after - before
            sign   = "+" if delta >= 0 else ""
            print(f"    {col:<28}  {before:>10.2f} → {after:>10.2f}  ({sign}{delta:.2f})")
    print()


def cmd_validate(args):
    from syndatakit.io import read, validate

    header(f"Validating: {args.file}")

    if not Path(args.file).exists():
        err(f"File not found: {args.file}"); sys.exit(1)

    info(f"Loading: {args.file}")
    df = read(args.file)
    info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()

    result = validate(
        df,
        null_threshold=args.null_threshold,
        duplicate_threshold=args.dup_threshold,
        max_cardinality=args.max_cardinality,
        min_rows=args.min_rows,
    )

    if result.passed:
        ok("Validation PASSED")
    else:
        err("Validation FAILED")

    if result.errors:
        section(f"Errors ({len(result.errors)})")
        for e in result.errors:
            print(f"    {_c('✗', C.RED)} {e}")

    if result.warnings:
        section(f"Warnings ({len(result.warnings)})")
        for w in result.warnings:
            print(f"    {_c('!', C.YELLOW)} {w}")

    section("Column summary")
    col_stats = result.stats.get("columns", {})
    for col, cs in list(col_stats.items())[:20]:
        null_pct = f"{cs['null_frac']*100:.1f}%"
        print(f"    {col:<28}  dtype={cs['dtype']:<12}  nulls={null_pct:<8}  unique={cs['n_unique']}")

    print()
    print(f"    Duplicate rows: {result.stats.get('duplicate_rows', 0)}  ({result.stats.get('duplicate_frac', 0)*100:.1f}%)")
    print()


def cmd_serve(args):
    header("Starting API server", f"http://localhost:{args.port}/docs")

    try:
        import flask  # noqa
    except ImportError:
        err("Flask required. Run: pip install syndatakit[api]")
        sys.exit(1)

    from syndatakit.api import app, _warm_generators
    from syndatakit.catalog import DATASETS

    print()
    info(f"Pre-warming {len(DATASETS)} generators...")
    _warm_generators()
    ok("All generators ready")
    print()
    info(f"Docs    → {_c(f'http://localhost:{args.port}/docs', C.CYAN)}")
    info(f"Datasets→ {_c(f'http://localhost:{args.port}/datasets', C.CYAN)}")
    info(f"Generate→ {_c(f'POST http://localhost:{args.port}/generate', C.CYAN)}")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


# ── Arg parser ────────────────────────────────────────────────────────────────

def main():
    print()
    print(_c("  syndatakit", C.BOLD) + _c(" v2.0.0", C.GRAY))
    print(_c("  research-grade synthetic data · finance & econometrics", C.GRAY))

    parser = argparse.ArgumentParser(prog="syndatakit", add_help=False)
    parser.add_argument("--help", "-h", action="help")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # list
    p = sub.add_parser("list", help="List available datasets.")
    p.add_argument("--vertical", type=str, default=None, metavar="V",
                   help="Filter by vertical e.g. 'Capital Markets'")
    p.set_defaults(func=cmd_list)

    # info
    p = sub.add_parser("info", help="Show full dataset metadata.")
    p.add_argument("dataset")
    p.set_defaults(func=cmd_info)

    # generate
    p = sub.add_parser("generate", help="Generate synthetic data.")
    p.add_argument("dataset")
    p.add_argument("--rows",      type=int,   default=1000,         metavar="N")
    p.add_argument("--output",    type=str,   default="output.csv", metavar="FILE",
                   help="Output path. Extension determines format: .csv .json .dta .xlsx")
    p.add_argument("--generator", type=str,   default="auto",       metavar="TYPE",
                   help="auto | copula | var | panel")
    p.add_argument("--filter",    type=str,   action="append",      metavar="EXPR",
                   help="e.g. --filter state:CA,TX  --filter dti_min:45")
    p.add_argument("--scenario",  type=str,   default=None,         metavar="NAME",
                   help="Apply built-in scenario: recession | rate_shock | credit_crisis ...")
    p.add_argument("--intensity", type=float, default=1.0,          metavar="F",
                   help="Scenario intensity 0.0–1.0 (default: 1.0)")
    p.add_argument("--calibrate", action="store_true",
                   help="Run moment calibration after generation")
    p.add_argument("--seed",      type=int,   default=None,         metavar="INT")
    p.add_argument("--no-eval",   action="store_true",
                   help="Skip fidelity evaluation")
    p.set_defaults(func=cmd_generate)

    # evaluate
    p = sub.add_parser("evaluate", help="Full fidelity report.")
    p.add_argument("real")
    p.add_argument("synthetic")
    p.add_argument("--type",   type=str, default="cross_sectional", metavar="TYPE",
                   help="cross_sectional | time_series | panel")
    p.add_argument("--target", type=str, default=None, metavar="COL",
                   help="Target column for TSTR downstream evaluation")
    p.add_argument("--json",   action="store_true", help="Also print JSON output")
    p.add_argument("--output", type=str, default=None, metavar="FILE",
                   help="Save JSON report to file")
    p.set_defaults(func=cmd_evaluate)

    # audit
    p = sub.add_parser("audit", help="Full privacy audit.")
    p.add_argument("real")
    p.add_argument("synthetic")
    p.add_argument("--attacks", type=int, default=300, metavar="N",
                   help="Number of attack attempts per test (default: 300)")
    p.add_argument("--json",    action="store_true")
    p.add_argument("--output",  type=str, default=None, metavar="FILE")
    p.set_defaults(func=cmd_audit)

    # scenario
    p_sc = sub.add_parser("scenario", help="Scenario management.")
    sc_sub = p_sc.add_subparsers(dest="scenario_cmd", metavar="<subcommand>")

    p_scl = sc_sub.add_parser("list", help="List built-in scenarios.")
    p_scl.set_defaults(func=cmd_scenario_list)

    p_sca = sc_sub.add_parser("apply", help="Apply a scenario to a CSV.")
    p_sca.add_argument("scenario")
    p_sca.add_argument("--input",     type=str, required=True,  metavar="FILE")
    p_sca.add_argument("--output",    type=str, required=True,  metavar="FILE")
    p_sca.add_argument("--intensity", type=float, default=1.0,  metavar="F")
    p_sca.set_defaults(func=cmd_scenario_apply)

    p_sc.set_defaults(func=lambda a: (sc_sub.print_help(), print()))

    # validate
    p = sub.add_parser("validate", help="Validate a real data file before fitting.")
    p.add_argument("file")
    p.add_argument("--null-threshold", type=float, default=0.3,  metavar="F")
    p.add_argument("--dup-threshold",  type=float, default=0.05, metavar="F")
    p.add_argument("--max-cardinality",type=int,   default=500,  metavar="N")
    p.add_argument("--min-rows",       type=int,   default=50,   metavar="N")
    p.set_defaults(func=cmd_validate)

    # serve
    p = sub.add_parser("serve", help="Start REST API server.")
    p.add_argument("--host",  type=str,  default="0.0.0.0", metavar="HOST")
    p.add_argument("--port",  type=int,  default=5000,      metavar="PORT")
    p.add_argument("--debug", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print()
        dim("  Examples:")
        dim("    syndatakit list")
        dim("    syndatakit generate fred_macro --rows 5000 --scenario recession")
        dim("    syndatakit generate credit_risk --filter fico_band:300-579 --filter default_12m:1")
        dim("    syndatakit evaluate real.csv synthetic.csv --type time_series --target gdp_growth_yoy")
        dim("    syndatakit audit real.csv synthetic.csv --attacks 500")
        dim("    syndatakit scenario list")
        dim("    syndatakit scenario apply recession --input syn.csv --output stressed.csv")
        dim("    syndatakit validate my_data.csv")
        dim("    syndatakit serve --port 8080")
        print()
        sys.exit(0)

    # Handle scenario sub-subcommand routing
    if args.command == "scenario" and not getattr(args, "scenario_cmd", None):
        cmd_scenario_list(args)
        return

    args.func(args)


if __name__ == "__main__":
    main()
