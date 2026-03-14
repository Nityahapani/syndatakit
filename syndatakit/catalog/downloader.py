"""
syndatakit.catalog.downloader
------------------------------
Downloads real bulk data from public government sources and caches it locally.
Once downloaded, load_seed() uses the real data instead of hand-coded approximations.

Usage
-----
    # Download one dataset
    from syndatakit.catalog.downloader import download, status
    download("hmda")

    # Download all
    download("all")

    # Check what's cached
    status()

CLI
---
    syndatakit download hmda
    syndatakit download all
    syndatakit download status

Sources
-------
All sources are free, public, and require no authentication except FRED
(which needs a free API key from fred.stlouisfed.org/docs/api/api_key.html).
"""
from __future__ import annotations
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ── Cache location ────────────────────────────────────────────────────────────
# Default: ~/.syndatakit/cache/
# Override: set SYNDATAKIT_CACHE env var

def _cache_dir() -> Path:
    base = Path(os.environ.get("SYNDATAKIT_CACHE", Path.home() / ".syndatakit" / "cache"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path(dataset_id: str) -> Path:
    return _cache_dir() / f"{dataset_id}.parquet"


def is_cached(dataset_id: str) -> bool:
    return cache_path(dataset_id).exists()


# ── Download registry ─────────────────────────────────────────────────────────

DOWNLOADERS: dict[str, dict] = {

    "hmda": {
        "name":        "HMDA Mortgage Applications",
        "source":      "CFPB HMDA 2022",
        "url":         "https://ffiec.cfpb.gov/data-download",
        "method":      "cfpb_api",
        "size_hint":   "~3 GB (14M rows) — may take 5–10 minutes",
        "columns_map": {
            "loan_amount":           "loan_amount",
            "income":                "applicant_income",
            "action_taken":          "action_taken",
            "loan_purpose":          "loan_purpose",
            "derived_dwelling_category": "property_type",
            "debt_to_income_ratio":  "debt_to_income",
            "state_code":            "state",
        },
    },

    "fdic": {
        "name":      "FDIC Bank Call Reports",
        "source":    "FDIC Statistics on Depository Institutions",
        "url":       "https://banks.data.fdic.gov/api/financials",
        "method":    "fdic_api",
        "size_hint": "~50 MB (5,000 banks × 20 quarters)",
    },

    "fred_macro": {
        "name":      "FRED Macroeconomic Indicators",
        "source":    "Federal Reserve FRED",
        "url":       "https://api.stlouisfed.org/fred/series/observations",
        "method":    "fred_api",
        "size_hint": "~1 MB — fast",
        "requires":  "FRED_API_KEY environment variable",
        "series": {
            "GDP":       "gdp_growth_yoy",
            "CPIAUCSL":  "cpi_yoy",
            "CPILFESL":  "core_cpi_yoy",
            "UNRATE":    "unemployment_rate",
            "FEDFUNDS":  "fed_funds_rate",
            "GS10":      "t10y_rate",
            "GS2":       "t2y_rate",
            "M2SL":      "m2_growth",
            "HOUST":     "housing_starts",
            "INDPRO":    "industrial_production",
            "UMCSENT":   "consumer_sentiment",
            "VIXCLS":    "vix",
        },
    },

    "bls": {
        "name":      "BLS Employment & Wages",
        "source":    "Bureau of Labor Statistics QCEW",
        "url":       "https://www.bls.gov/cew/downloadable-data.htm",
        "method":    "bls_api",
        "size_hint": "~200 MB",
    },

    "world_bank": {
        "name":      "World Bank Development Indicators",
        "source":    "World Bank WDI API",
        "url":       "https://api.worldbank.org/v2/country/all/indicator",
        "method":    "worldbank_api",
        "size_hint": "~5 MB — fast",
        "indicators": {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "NY.GDP.PCAP.KD":    "gdp_per_capita",
            "FP.CPI.TOTL.ZG":    "inflation",
            "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
            "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",
            "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",
            "SP.POP.GROW":       "population_growth",
            "SI.POV.GINI":       "gini",
        },
    },

    "census_acs": {
        "name":      "Census ACS Income & Housing",
        "source":    "US Census ACS 5-Year API",
        "url":       "https://api.census.gov/data/2022/acs/acs5",
        "method":    "census_api",
        "size_hint": "~10 MB",
    },
}


# ── Download implementations ──────────────────────────────────────────────────

def _download_world_bank(dataset_id: str, n_sample: int = 10000) -> pd.DataFrame:
    """Download World Bank WDI via free REST API. No key needed."""
    try:
        import urllib.request
    except ImportError:
        raise ImportError("urllib required (should be in stdlib)")

    indicators = DOWNLOADERS["world_bank"]["indicators"]
    base_url   = DOWNLOADERS["world_bank"]["url"]
    frames     = {}

    for code, col_name in indicators.items():
        url = (f"{base_url}/{code}?format=json&per_page=20000"
               f"&mrv=25&date=2000:2023")
        print(f"    Fetching {col_name} ({code})...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            records = data[1] if len(data) > 1 else []
            rows = []
            for rec in records:
                if rec.get("value") is not None:
                    rows.append({
                        "country_code": rec["country"]["id"],
                        "year":         int(rec["date"]),
                        col_name:       float(rec["value"]),
                    })
            frames[col_name] = pd.DataFrame(rows)
            print(f"{len(rows)} rows")
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        time.sleep(0.3)  # be polite to the API

    if not frames:
        raise RuntimeError("No World Bank data downloaded")

    # Merge all indicators on country_code + year
    df = None
    for col_name, frame in frames.items():
        if df is None:
            df = frame
        else:
            df = df.merge(frame, on=["country_code","year"], how="outer")

    # Add metadata columns
    meta_url = "https://api.worldbank.org/v2/country?format=json&per_page=300"
    try:
        with urllib.request.urlopen(meta_url, timeout=30) as r:
            meta_data = json.loads(r.read())
        meta_rows = []
        for c in meta_data[1]:
            meta_rows.append({
                "country_code":  c["id"],
                "income_group":  c.get("incomeLevel",{}).get("value","Unknown"),
                "region":        c.get("region",{}).get("value","Unknown"),
            })
        meta_df = pd.DataFrame(meta_rows)
        df = df.merge(meta_df, on="country_code", how="left")
    except Exception:
        df["income_group"] = "Unknown"
        df["region"] = "Unknown"

    df = df.dropna(subset=["gdp_growth"]).reset_index(drop=True)
    return df.sample(min(n_sample, len(df)), random_state=42)


def _download_fred(dataset_id: str, n_sample: int = 10000) -> pd.DataFrame:
    """Download FRED series via API. Requires FRED_API_KEY env var."""
    import urllib.request, urllib.parse

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY environment variable not set.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then: export FRED_API_KEY=your_key_here"
        )

    series_map = DOWNLOADERS["fred_macro"]["series"]
    base_url   = "https://api.stlouisfed.org/fred/series/observations"
    frames     = {}

    for series_id, col_name in series_map.items():
        params = urllib.parse.urlencode({
            "series_id":      series_id,
            "api_key":        api_key,
            "file_type":      "json",
            "observation_start": "2000-01-01",
            "frequency":      "m",
        })
        url = f"{base_url}?{params}"
        print(f"    Fetching {col_name} ({series_id})...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            rows = [
                {"date": obs["date"], col_name: float(obs["value"])}
                for obs in data["observations"]
                if obs["value"] != "."
            ]
            frames[col_name] = pd.DataFrame(rows)
            print(f"{len(rows)} observations")
        except Exception as e:
            print(f"FAILED ({e})")
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("No FRED data downloaded")

    df = None
    for col_name, frame in frames.items():
        if df is None:
            df = frame
        else:
            df = df.merge(frame, on="date", how="outer")

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.dropna(subset=["gdp_growth_yoy"] if "gdp_growth_yoy" in df.columns
                   else [list(df.columns)[1]])

    # Add VIX if available (it's in a different series)
    if "vix" not in df.columns:
        df["vix"] = np.nan

    return df.sample(min(n_sample, len(df)), random_state=42)


def _download_census(dataset_id: str, n_sample: int = 10000) -> pd.DataFrame:
    """Download Census ACS via free API. No key needed for basic variables."""
    import urllib.request, urllib.parse

    # ACS 5-year 2022 — household income and housing cost variables
    base = "https://api.census.gov/data/2022/acs/acs5"
    variables = "B19013_001E,B25070_001E,B25070_010E,B23025_002E,B23025_005E,B25003_001E,B25003_002E"
    # B19013: median HH income, B25070: rent burden, B23025: employment, B25003: tenure

    # Pull by state (need to paginate)
    states = list(range(1, 57))
    all_rows = []

    for state in states[:10]:  # sample 10 states for speed; real download would do all
        params = urllib.parse.urlencode({
            "get":   variables,
            "for":   "tract:*",
            "in":    f"state:{state:02d}",
        })
        url = f"{base}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            headers = data[0]
            for row in data[1:]:
                rec = dict(zip(headers, row))
                all_rows.append(rec)
        except Exception:
            continue
        time.sleep(0.1)

    if not all_rows:
        raise RuntimeError("No Census data downloaded")

    df_raw = pd.DataFrame(all_rows)

    # Clean and rename
    rename = {
        "B19013_001E": "household_income",
        "B25070_001E": "total_renter_units",
        "B25070_010E": "severe_burden_units",
        "B23025_002E": "in_labor_force",
        "B23025_005E": "unemployed",
        "B25003_001E": "total_housing_units",
        "B25003_002E": "owner_occupied",
        "state":       "state_fips",
        "tract":       "tract_id",
    }
    df = df_raw.rename(columns={k: v for k, v in rename.items() if k in df_raw.columns})

    # Convert to numerics
    for col in ["household_income","total_renter_units","severe_burden_units",
                "in_labor_force","unemployed","total_housing_units","owner_occupied"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["household_income"] > 0].dropna(subset=["household_income"])
    return df.sample(min(n_sample, len(df)), random_state=42)


# ── Main download function ────────────────────────────────────────────────────

def download(
    dataset_id: str,
    force: bool = False,
    n_sample: int = 50_000,
) -> pd.DataFrame:
    """
    Download real bulk data for a dataset and cache it locally.

    Parameters
    ----------
    dataset_id : dataset ID, or "all" to download everything
    force      : re-download even if cached
    n_sample   : max rows to cache (default 50,000 — enough for high-fidelity seeds)

    Returns
    -------
    DataFrame of downloaded data (also saved to cache)

    Example
    -------
        from syndatakit.catalog.downloader import download
        df = download("world_bank")
        df = download("fred_macro")   # needs FRED_API_KEY env var
    """
    if dataset_id == "all":
        results = {}
        for did in DOWNLOADERS:
            try:
                results[did] = download(did, force=force, n_sample=n_sample)
            except Exception as e:
                print(f"  ✗ {did}: {e}")
        return results

    if dataset_id not in DOWNLOADERS:
        available = ", ".join(DOWNLOADERS)
        raise ValueError(
            f"No downloader for '{dataset_id}'. Available: {available}\n"
            f"For other datasets, pass your own CSV to gen.fit(your_df)."
        )

    cached = cache_path(dataset_id)
    if cached.exists() and not force:
        print(f"  ✓ {dataset_id} — loaded from cache ({cached})")
        return pd.read_parquet(cached)

    info = DOWNLOADERS[dataset_id]
    print(f"\n  Downloading {info['name']}...")
    print(f"  Source: {info['source']}")
    print(f"  Size:   {info['size_hint']}")

    method = info["method"]
    if method == "worldbank_api":
        df = _download_world_bank(dataset_id, n_sample)
    elif method == "fred_api":
        df = _download_fred(dataset_id, n_sample)
    elif method == "census_api":
        df = _download_census(dataset_id, n_sample)
    else:
        raise NotImplementedError(
            f"Downloader for '{dataset_id}' (method: {method}) requires manual download.\n"
            f"See BULK_DOWNLOAD.md for step-by-step instructions."
        )

    # Cache it
    df.to_parquet(cached, index=False)
    print(f"  ✓ Cached {len(df):,} rows → {cached}")
    return df


def load_cached(dataset_id: str) -> pd.DataFrame | None:
    """
    Load cached real data if available, else return None.
    Used by load_seed() to prefer real data over hand-coded approximations.
    """
    p = cache_path(dataset_id)
    if p.exists():
        return pd.read_parquet(p)
    return None


def status() -> pd.DataFrame:
    """Show which datasets have been downloaded and cached."""
    rows = []
    for did, info in DOWNLOADERS.items():
        p = cache_path(did)
        if p.exists():
            df = pd.read_parquet(p)
            rows.append({
                "dataset": did,
                "status":  "✓ cached",
                "rows":    f"{len(df):,}",
                "size":    f"{p.stat().st_size // 1024:,} KB",
                "path":    str(p),
            })
        else:
            rows.append({
                "dataset": did,
                "status":  "○ not downloaded",
                "rows":    "—",
                "size":    "—",
                "path":    "—",
            })
    return pd.DataFrame(rows)
