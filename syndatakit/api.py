"""
syndatakit v2 REST API
-----------------------
Endpoints:
    GET  /health
    GET  /datasets[?vertical=]
    GET  /datasets/<id>
    GET  /datasets/<id>/sample[?rows=&format=]
    POST /generate
    POST /evaluate
    POST /audit
    POST /scenario/apply
    GET  /scenarios
    POST /validate
    GET  /docs
"""
from __future__ import annotations
import io
import json
import time
import traceback
from functools import wraps

from flask import Flask, request, jsonify, Response

from syndatakit.catalog import list_datasets, get_dataset_info, load_seed, DATASETS
from syndatakit.calibration import apply_scenario, list_scenarios

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Generator cache ───────────────────────────────────────────────────────────

_generators: dict[str, object] = {}
_seed_cache:  dict[str, object] = {}

_TIME_SERIES_DATASETS = {"fred_macro", "bls"}
_PANEL_DATASETS       = {"world_bank", "fdic"}


def _make_generator(dataset_id: str):
    from syndatakit.generators import GaussianCopulaGenerator
    from syndatakit.generators.time_series import VARGenerator
    from syndatakit.generators.panel import FixedEffectsGenerator

    seed = _get_seed(dataset_id)

    if dataset_id in _TIME_SERIES_DATASETS:
        time_col = "year" if "year" in seed.columns else None
        gen = VARGenerator(lags=2, time_col=time_col)
    elif dataset_id in _PANEL_DATASETS:
        entity_col = next((c for c in ["country_code","bank_id"] if c in seed.columns), seed.columns[0])
        time_col   = "year" if "year" in seed.columns else "quarter" if "quarter" in seed.columns else None
        gen = FixedEffectsGenerator(entity_col=entity_col, time_col=time_col)
    else:
        gen = GaussianCopulaGenerator()

    gen.fit(seed)
    return gen


def _get_gen(dataset_id: str):
    if dataset_id not in _generators:
        _generators[dataset_id] = _make_generator(dataset_id)
    return _generators[dataset_id]


def _get_seed(dataset_id: str):
    if dataset_id not in _seed_cache:
        _seed_cache[dataset_id] = load_seed(dataset_id)
    return _seed_cache[dataset_id]


def _warm_generators():
    """Pre-fit all generators. Called on startup."""
    for did in DATASETS:
        try:
            _get_gen(did)
            print(f"    ✓ {did}")
        except Exception as e:
            print(f"    ✗ {did}: {e}")


# ── Response helpers ──────────────────────────────────────────────────────────

def _ok(data, meta=None, status=200):
    payload = {"status": status, "data": data}
    if meta:
        payload["meta"] = meta
    return jsonify(payload), status


def _err(msg, code=400):
    return jsonify({"status": code, "error": msg}), code


def _timed(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        resp = result[0] if isinstance(result, tuple) else result
        try:
            resp.headers["X-Response-Time-Ms"] = str(round((time.time()-t0)*1000, 1))
        except Exception:
            pass
        return result
    return wrapper


def _clean_for_json(obj):
    """Recursively make a dict JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return round(float(obj), 6)
        if isinstance(obj, np.bool_):       return bool(obj)
    except ImportError:
        pass
    return obj


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":            "ok",
        "version":           "2.0.0",
        "datasets":          len(DATASETS),
        "generators_cached": len(_generators),
    })


@app.route("/datasets")
@_timed
def get_datasets():
    """GET /datasets[?vertical=Capital+Markets]"""
    vertical = request.args.get("vertical")
    df = list_datasets(vertical=vertical)
    return _ok(
        df.to_dict(orient="records"),
        meta={"count": len(df), "vertical_filter": vertical},
    )


@app.route("/datasets/<dataset_id>")
@_timed
def get_dataset(dataset_id):
    """GET /datasets/<id>"""
    try:
        info = get_dataset_info(dataset_id)
    except ValueError as e:
        return _err(str(e), 404)
    return _ok(info)


@app.route("/datasets/<dataset_id>/sample")
@_timed
def get_sample(dataset_id):
    """GET /datasets/<id>/sample?rows=10&format=json"""
    try:
        n   = min(int(request.args.get("rows", 10)), 100)
        fmt = request.args.get("format", "json").lower()
        df  = _get_seed(dataset_id).head(n)
        if fmt == "csv":
            return Response(
                df.to_csv(index=False), mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename={dataset_id}_sample.csv"},
            )
        return _ok(df.to_dict(orient="records"), meta={"rows": len(df), "dataset": dataset_id})
    except ValueError as e:
        return _err(str(e), 404)
    except Exception as e:
        return _err(str(e), 500)


@app.route("/generate", methods=["POST"])
@_timed
def generate():
    """
    POST /generate
    {
        "dataset":   "fred_macro",
        "rows":      1000,
        "format":    "json",          // json | csv
        "seed":      42,
        "generator": "auto",          // auto | copula | var | panel
        "filters":   { "state": ["CA","TX"], "dti_min": 45 },
        "scenario":  "recession",     // optional built-in scenario
        "intensity": 0.8,             // scenario intensity 0–1
        "calibrate": false            // moment calibration
    }
    """
    body = request.get_json(silent=True) or {}

    dataset_id = body.get("dataset")
    if not dataset_id:
        return _err("'dataset' is required. See GET /datasets.")
    if dataset_id not in DATASETS:
        return _err(f"Unknown dataset '{dataset_id}'. Available: {', '.join(DATASETS)}", 404)

    rows      = min(int(body.get("rows", 1000)), 50_000)
    fmt       = body.get("format", "json").lower()
    seed      = body.get("seed")
    filters   = body.get("filters") or {}
    scenario  = body.get("scenario")
    intensity = float(body.get("intensity", 1.0))
    calibrate = bool(body.get("calibrate", False))

    try:
        t0  = time.time()
        gen = _get_gen(dataset_id)
        df  = gen.sample(rows, filters=filters or None, seed=seed)

        # Apply scenario if requested
        if scenario:
            df = apply_scenario(df, scenario, intensity=intensity)

        # Moment calibration
        if calibrate:
            from syndatakit.calibration import match_moments
            seed_df  = _get_seed(dataset_id)
            df_body  = df.drop(columns=["syn_id"], errors="ignore")
            df_body  = match_moments(seed_df, df_body)
            df_body.insert(0, "syn_id", df["syn_id"])
            df = df_body

        elapsed = round((time.time()-t0)*1000, 1)

    except ValueError as e:
        return _err(str(e), 400)
    except Exception as e:
        traceback.print_exc()
        return _err(f"Generation failed: {e}", 500)

    meta = {
        "dataset":        dataset_id,
        "rows_generated": len(df),
        "rows_requested": rows,
        "filters":        filters,
        "scenario":       scenario,
        "calibrated":     calibrate,
        "generation_ms":  elapsed,
    }

    if fmt == "csv":
        return Response(
            df.to_csv(index=False), mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={dataset_id}_synthetic.csv",
                "X-Rows-Generated":    str(len(df)),
                "X-Generation-Ms":     str(elapsed),
            },
        )

    return _ok(df.to_dict(orient="records"), meta=meta)


@app.route("/evaluate", methods=["POST"])
@_timed
def evaluate():
    """
    POST /evaluate  (multipart/form-data or JSON)

    Multipart:
        dataset      : dataset ID
        type         : cross_sectional | time_series | panel
        target_col   : optional, for TSTR downstream score
        real         : CSV file
        synthetic    : CSV file

    JSON:
        { "dataset": "hmda", "type": "cross_sectional",
          "real": [...], "synthetic": [...], "target_col": "loan_amount" }
    """
    import pandas as pd
    from syndatakit.fidelity import fidelity_report

    if "real" in request.files:
        dataset_id   = request.form.get("dataset", "hmda")
        dataset_type = request.form.get("type", "cross_sectional")
        target_col   = request.form.get("target_col") or None
        try:
            real = pd.read_csv(request.files["real"])
            syn  = pd.read_csv(request.files["synthetic"])
        except Exception as e:
            return _err(f"Could not parse uploaded files: {e}")
    else:
        body = request.get_json(silent=True) or {}
        dataset_id   = body.get("dataset", "hmda")
        dataset_type = body.get("type", "cross_sectional")
        target_col   = body.get("target_col")
        try:
            real = pd.DataFrame(body.get("real", []))
            syn  = pd.DataFrame(body.get("synthetic", []))
        except Exception as e:
            return _err(f"Could not parse data arrays: {e}")

    if real.empty or syn.empty:
        return _err("Both 'real' and 'synthetic' must be non-empty.")

    try:
        report = fidelity_report(
            real, syn,
            dataset_type=dataset_type,
            target_col=target_col,
            include_downstream=bool(target_col),
        )
    except Exception as e:
        traceback.print_exc()
        return _err(f"Evaluation failed: {e}", 500)

    return _ok(
        _clean_for_json(report),
        meta={"dataset": dataset_id, "real_rows": len(real), "synthetic_rows": len(syn)},
    )


@app.route("/audit", methods=["POST"])
@_timed
def audit():
    """
    POST /audit  (multipart/form-data or JSON)
    Runs full privacy audit: exact copies, membership inference,
    singling-out, linkability.

    Multipart: real=<csv>, synthetic=<csv>, [attacks=300]
    JSON: { "real": [...], "synthetic": [...], "attacks": 300 }
    """
    import pandas as pd
    from syndatakit.privacy import privacy_audit

    if "real" in request.files:
        n_attacks = int(request.form.get("attacks", 300))
        try:
            real = pd.read_csv(request.files["real"])
            syn  = pd.read_csv(request.files["synthetic"])
        except Exception as e:
            return _err(f"Could not parse uploaded files: {e}")
    else:
        body = request.get_json(silent=True) or {}
        n_attacks = int(body.get("attacks", 300))
        try:
            real = pd.DataFrame(body.get("real", []))
            syn  = pd.DataFrame(body.get("synthetic", []))
        except Exception as e:
            return _err(f"Could not parse data arrays: {e}")

    if real.empty or syn.empty:
        return _err("Both 'real' and 'synthetic' must be non-empty.")

    try:
        report = privacy_audit(real, syn, n_attacks=n_attacks, seed=42)
    except Exception as e:
        traceback.print_exc()
        return _err(f"Audit failed: {e}", 500)

    return _ok(
        _clean_for_json(report),
        meta={"real_rows": len(real), "synthetic_rows": len(syn), "n_attacks": n_attacks},
    )


@app.route("/scenarios")
@_timed
def get_scenarios():
    """GET /scenarios — list all built-in scenarios."""
    from syndatakit.calibration import SCENARIOS
    out = [
        {
            "name":              k,
            "description":       v["description"],
            "columns_affected":  len(v["shifts"]),
            "columns":           list(v["shifts"].keys()),
        }
        for k, v in SCENARIOS.items()
    ]
    return _ok(out, meta={"count": len(out)})


@app.route("/scenario/apply", methods=["POST"])
@_timed
def scenario_apply():
    """
    POST /scenario/apply
    {
        "scenario":  "recession",
        "intensity": 1.0,
        "data":      [{ ... }]     // array of records
    }
    Or multipart with data=<csv>, scenario=<name>, intensity=<f>
    """
    import pandas as pd

    if "data" in request.files:
        scenario  = request.form.get("scenario", "recession")
        intensity = float(request.form.get("intensity", 1.0))
        try:
            df = pd.read_csv(request.files["data"])
        except Exception as e:
            return _err(f"Could not parse uploaded file: {e}")
    else:
        body      = request.get_json(silent=True) or {}
        scenario  = body.get("scenario", "recession")
        intensity = float(body.get("intensity", 1.0))
        try:
            df = pd.DataFrame(body.get("data", []))
        except Exception as e:
            return _err(f"Could not parse data: {e}")

    if df.empty:
        return _err("'data' must be a non-empty array of records.")

    try:
        result = apply_scenario(df, scenario, intensity=intensity)
    except ValueError as e:
        return _err(str(e), 400)

    fmt = request.args.get("format", "json").lower()
    if fmt == "csv":
        return Response(
            result.to_csv(index=False), mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={scenario}.csv"},
        )

    return _ok(
        result.to_dict(orient="records"),
        meta={"scenario": scenario, "intensity": intensity, "rows": len(result)},
    )


@app.route("/validate", methods=["POST"])
@_timed
def validate_data():
    """
    POST /validate  (multipart: file=<csv>)
    Validates a real data file for generator compatibility.
    """
    import pandas as pd
    from syndatakit.io import validate

    if "file" in request.files:
        try:
            df = pd.read_csv(request.files["file"])
        except Exception as e:
            return _err(f"Could not parse file: {e}")
    else:
        body = request.get_json(silent=True) or {}
        try:
            df = pd.DataFrame(body.get("data", []))
        except Exception as e:
            return _err(f"Could not parse data: {e}")

    if df.empty:
        return _err("No data provided.")

    result = validate(df)

    return _ok({
        "passed":   result.passed,
        "errors":   result.errors,
        "warnings": result.warnings,
        "stats":    _clean_for_json(result.stats),
    })


# ── Docs page ─────────────────────────────────────────────────────────────────

DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>syndatakit v2 API</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#080b0f;--s:#0f1318;--s2:#161c23;--b:#1e262f;--b2:#263040;--t:#dce8f0;--m:#5a7a90;--a:#00d4aa;--info:#4fa3e8;--warn:#f0b840;--red:#e85050;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--t);font-family:'Sora',sans-serif;font-size:13.5px;line-height:1.7;}
nav{padding:16px 40px;border-bottom:1px solid var(--b);display:flex;align-items:center;gap:20px;}
.logo{font-family:'DM Mono',monospace;color:var(--a);font-size:16px;font-weight:500;}
.version{font-family:'DM Mono',monospace;font-size:11px;background:rgba(0,212,170,.08);color:var(--a);border:1px solid rgba(0,212,170,.2);border-radius:4px;padding:2px 8px;}
.tag{font-size:11px;background:var(--s2);color:var(--m);border:1px solid var(--b);border-radius:4px;padding:2px 8px;}
.container{max-width:960px;margin:0 auto;padding:40px;}
h1{font-size:28px;font-weight:300;margin-bottom:6px;}
h1 strong{color:var(--a);font-weight:600;}
.sub{color:var(--m);margin-bottom:36px;font-size:14px;}
.section-title{font-size:10px;letter-spacing:2.5px;text-transform:uppercase;color:var(--m);margin:36px 0 14px;border-bottom:1px solid var(--b);padding-bottom:8px;}
.endpoint{background:var(--s);border:1px solid var(--b);border-radius:10px;margin-bottom:10px;overflow:hidden;transition:border-color .2s;}
.endpoint:hover{border-color:var(--b2);}
.ep-header{display:flex;align-items:center;gap:12px;padding:13px 18px;cursor:pointer;user-select:none;}
.method{font-family:'DM Mono',monospace;font-size:11px;font-weight:500;padding:3px 10px;border-radius:4px;flex-shrink:0;min-width:48px;text-align:center;}
.get {background:rgba(79,163,232,.1);color:var(--info);border:1px solid rgba(79,163,232,.25);}
.post{background:rgba(0,212,170,.1);color:var(--a);border:1px solid rgba(0,212,170,.25);}
.path{font-family:'DM Mono',monospace;font-size:13px;flex:1;}
.ep-desc{color:var(--m);font-size:12px;}
.ep-body{padding:0 18px 18px;border-top:1px solid var(--b);display:none;}
.ep-body.open{display:block;}
.ep-body h4{font-size:10px;color:var(--m);letter-spacing:1.5px;text-transform:uppercase;margin:14px 0 7px;}
pre{background:var(--s2);border:1px solid var(--b);border-radius:6px;padding:13px;font-family:'DM Mono',monospace;font-size:12px;line-height:1.75;overflow-x:auto;white-space:pre-wrap;}
.try-btn{display:inline-flex;align-items:center;gap:6px;font-family:'Sora',sans-serif;font-size:12px;background:rgba(0,212,170,.08);color:var(--a);border:1px solid rgba(0,212,170,.2);border-radius:6px;padding:6px 14px;cursor:pointer;margin-top:10px;transition:all .15s;}
.try-btn:hover{background:rgba(0,212,170,.16);}
.response-box{margin-top:10px;display:none;}
.response-box.show{display:block;}
.response-box pre{border-color:rgba(0,212,170,.3);max-height:320px;overflow-y:auto;}
.ds-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;}
.ds-table th{background:var(--s2);padding:8px 12px;text-align:left;color:var(--m);font-weight:500;border-bottom:1px solid var(--b);}
.ds-table td{padding:8px 12px;border-bottom:1px solid var(--b);font-family:'DM Mono',monospace;}
.ds-table tr:last-child td{border:none;}
.live{color:var(--a);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px;}
.stat-box{background:var(--s2);border:1px solid var(--b);border-radius:8px;padding:14px;}
.stat-val{font-size:22px;font-weight:600;color:var(--a);font-family:'DM Mono',monospace;}
.stat-lbl{font-size:11px;color:var(--m);margin-top:2px;}
</style>
</head>
<body>
<nav>
  <div class="logo">syndatakit</div>
  <div class="version">v2.0.0</div>
  <div class="tag">finance &amp; econometrics</div>
  <div class="tag">10 datasets</div>
  <div class="tag">running on :5000</div>
</nav>

<div class="container">
  <h1>Synthetic data <strong>API</strong></h1>
  <p class="sub">Gaussian Copula · VAR · Panel · Full fidelity reporting · Privacy audit · Scenario calibration</p>

  <div class="grid">
    <div class="stat-box"><div class="stat-val" id="s-datasets">—</div><div class="stat-lbl">live datasets</div></div>
    <div class="stat-box"><div class="stat-val" id="s-cached">—</div><div class="stat-lbl">generators cached</div></div>
  </div>

  <div class="section-title">Core</div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method get">GET</span><span class="path">/datasets</span>
      <span class="ep-desc">List all datasets · optional ?vertical=</span>
    </div>
    <div class="ep-body">
      <h4>Available datasets</h4>
      <table class="ds-table"><thead><tr><th>ID</th><th>Name</th><th>Vertical</th><th>Cols</th><th>Fidelity</th></tr></thead>
      <tbody id="ds-tbody"><tr><td colspan="5" style="color:var(--m)">Loading...</td></tr></tbody></table>
      <button class="try-btn" onclick="tryGet(this,'/datasets')">Run →</button>
      <div class="response-box"><pre></pre></div>
    </div>
  </div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method get">GET</span><span class="path">/datasets/{id}/sample</span>
      <span class="ep-desc">Preview sample rows</span>
    </div>
    <div class="ep-body">
      <button class="try-btn" onclick="tryGet(this,'/datasets/fred_macro/sample?rows=5')">Sample fred_macro →</button>
      <div class="response-box"><pre></pre></div>
    </div>
  </div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method post">POST</span><span class="path">/generate</span>
      <span class="ep-desc">Generate synthetic rows</span>
    </div>
    <div class="ep-body">
      <h4>Request body</h4>
      <pre id="gen-body">{
  "dataset":   "fred_macro",
  "rows":      500,
  "generator": "auto",
  "seed":      42,
  "scenario":  "recession",
  "intensity": 0.8,
  "filters":   { "gdp_growth_yoy_min": -5 }
}</pre>
      <h4>Generator types</h4>
      <pre>auto    — inferred from dataset (recommended)
copula  — Gaussian Copula (cross-sectional)
var     — VAR(p) (time series: fred_macro, bls)
panel   — Fixed effects (panel: world_bank, fdic)</pre>
      <h4>Filter syntax</h4>
      <pre>{ "state": ["CA","TX"] }          // exact match / list
{ "dti_min": 45 }                 // numeric lower bound
{ "loan_max": 500000 }            // numeric upper bound
{ "sector": "Technology" }        // single categorical value</pre>
      <button class="try-btn" onclick="tryPost(this,'/generate',document.getElementById('gen-body').textContent)">Run →</button>
      <div class="response-box"><pre></pre></div>
    </div>
  </div>

  <div class="section-title">Evaluation</div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method post">POST</span><span class="path">/evaluate</span>
      <span class="ep-desc">Full fidelity report</span>
    </div>
    <div class="ep-body">
      <h4>Multipart upload</h4>
      <pre>curl -X POST http://localhost:5000/evaluate \\
  -F dataset=fred_macro \\
  -F type=time_series \\
  -F target_col=gdp_growth_yoy \\
  -F real=@real.csv \\
  -F synthetic=@synthetic.csv</pre>
      <h4>Response includes</h4>
      <pre>marginal      — per-column KS / TVD scores
joint         — Spearman correlation distance
stylized_facts— fat tails, skewness, ARCH, autocorr
temporal      — stationarity, cointegration, breaks, causality
downstream    — TSTR (train-on-synthetic, test-on-real)
privacy_basic — exact copy count
summary       — overall_fidelity, marginal, joint, privacy</pre>
    </div>
  </div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method post">POST</span><span class="path">/audit</span>
      <span class="ep-desc">Full privacy audit</span>
    </div>
    <div class="ep-body">
      <h4>Multipart upload</h4>
      <pre>curl -X POST http://localhost:5000/audit \\
  -F real=@real.csv \\
  -F synthetic=@synthetic.csv \\
  -F attacks=500</pre>
      <h4>Tests run</h4>
      <pre>exact_copies          — zero-tolerance check
membership_inference  — AUC-based attack (shadow model)
singling_out          — quasi-identifier subset attack
linkability           — nearest-neighbour cross-dataset attack
verdict               — overall risk + recommendation</pre>
    </div>
  </div>

  <div class="section-title">Calibration</div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method get">GET</span><span class="path">/scenarios</span>
      <span class="ep-desc">List built-in scenarios</span>
    </div>
    <div class="ep-body">
      <button class="try-btn" onclick="tryGet(this,'/scenarios')">Run →</button>
      <div class="response-box"><pre></pre></div>
    </div>
  </div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method post">POST</span><span class="path">/scenario/apply</span>
      <span class="ep-desc">Apply recession, rate_shock, etc. to any dataset</span>
    </div>
    <div class="ep-body">
      <h4>Request body</h4>
      <pre id="sc-body">{
  "scenario":  "recession",
  "intensity": 1.0,
  "data":      []
}</pre>
      <h4>Available scenarios</h4>
      <pre>recession         mild contraction (-2.5% GDP, rising unemployment)
severe_recession  GFC-style shock (-5% GDP, VIX=55)
rate_shock        rapid tightening (FFR=5%, CPI=7.5%)
credit_crisis     NPL surge, bank stress
expansion         above-trend growth, tight labour market</pre>
    </div>
  </div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method post">POST</span><span class="path">/validate</span>
      <span class="ep-desc">Validate a real data file</span>
    </div>
    <div class="ep-body">
      <h4>Multipart upload</h4>
      <pre>curl -X POST http://localhost:5000/validate -F file=@my_data.csv</pre>
      <h4>Checks</h4>
      <pre>null fractions · constant columns · cardinality · duplicate rows · dtype expectations</pre>
    </div>
  </div>

  <div class="section-title">Code examples</div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method get">PY</span><span class="path">Python</span>
    </div>
    <div class="ep-body"><pre>import requests, pandas as pd

# Generate 2,000 recession-scenario macro rows
r = requests.post("http://localhost:5000/generate", json={
    "dataset":   "fred_macro",
    "rows":      2000,
    "scenario":  "recession",
    "intensity": 0.9,
})
df = pd.DataFrame(r.json()["data"])

# Full fidelity evaluation
real = pd.read_csv("real_macro.csv")
r2 = requests.post("http://localhost:5000/evaluate",
    files={"real": open("real_macro.csv"), "synthetic": open("synthetic.csv")},
    data={"type": "time_series", "target_col": "gdp_growth_yoy"},
)
print(r2.json()["data"]["summary"])</pre></div>
  </div>

  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method get">SH</span><span class="path">curl</span>
    </div>
    <div class="ep-body"><pre># Generate + scenario
curl -X POST http://localhost:5000/generate \\
  -H "Content-Type: application/json" \\
  -d '{"dataset":"credit_risk","rows":5000,"scenario":"credit_crisis"}' \\
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['data']),'rows')"

# Audit (CSV upload)
curl -X POST http://localhost:5000/audit \\
  -F real=@real.csv -F synthetic=@syn.csv -F attacks=500 \\
  | python3 -m json.tool</pre></div>
  </div>
</div>

<script>
function toggle(el){ el.nextElementSibling.classList.toggle('open'); }

async function tryGet(btn, url) {
  const box = btn.nextElementSibling;
  box.classList.add('show');
  box.querySelector('pre').textContent = 'Loading...';
  try {
    const r = await fetch(url);
    const j = await r.json();
    box.querySelector('pre').textContent = JSON.stringify(j, null, 2).slice(0, 3000);
  } catch(e) { box.querySelector('pre').textContent = 'Error: ' + e.message; }
}

async function tryPost(btn, url, body) {
  const box = btn.nextElementSibling;
  box.classList.add('show');
  box.querySelector('pre').textContent = 'Generating...';
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: body.trim()
    });
    const j = await r.json();
    box.querySelector('pre').textContent = JSON.stringify(j, null, 2).slice(0, 4000);
  } catch(e) { box.querySelector('pre').textContent = 'Error: ' + e.message; }
}

// Load stats + dataset table on mount
fetch('/health').then(r=>r.json()).then(j=>{
  document.getElementById('s-datasets').textContent = j.datasets;
  document.getElementById('s-cached').textContent   = j.generators_cached;
});
fetch('/datasets').then(r=>r.json()).then(j=>{
  document.getElementById('ds-tbody').innerHTML = j.data.map(d =>
    `<tr>
      <td class="live">${d.id}</td>
      <td>${d.name}</td>
      <td style="color:var(--m)">${d.vertical}</td>
      <td>${d.columns}</td>
      <td class="live">${d.fidelity}</td>
    </tr>`
  ).join('');
});
</script>
</body>
</html>"""


@app.route("/")
@app.route("/docs")
def docs():
    return DOCS_HTML
