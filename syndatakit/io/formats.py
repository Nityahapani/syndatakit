"""
syndatakit.io.formats
----------------------
Read/write synthetic data in all supported formats.

Supported output formats:
    csv       — universal, default
    parquet   — columnar, efficient for large datasets
    arrow     — Apache Arrow IPC (feather v2)
    json      — records-oriented JSON
    stata     — Stata .dta (requires pyreadstat)
    sas       — SAS .sas7bdat (requires pyreadstat, read-only)
    excel     — .xlsx (requires openpyxl)

Supported input formats (for real data ingestion):
    csv, parquet, arrow, json, stata, sas, excel
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


# ── Write ─────────────────────────────────────────────────────────────────────

def write(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str | None = None,
    **kwargs,
) -> Path:
    """
    Write a DataFrame to file.

    Parameters
    ----------
    df   : DataFrame to write
    path : output file path
    fmt  : format override (inferred from extension if None)

    Returns the Path written to.
    """
    path = Path(path)
    fmt  = fmt or _infer_format(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    writers = {
        "csv":     _write_csv,
        "parquet": _write_parquet,
        "arrow":   _write_arrow,
        "json":    _write_json,
        "stata":   _write_stata,
        "excel":   _write_excel,
    }

    if fmt not in writers:
        raise ValueError(f"Unsupported format '{fmt}'. Available: {', '.join(writers)}")

    writers[fmt](df, path, **kwargs)
    return path


def _write_csv(df, path, **kw):
    df.to_csv(path, index=False, **kw)

def _write_parquet(df, path, **kw):
    df.to_parquet(path, index=False, **kw)

def _write_arrow(df, path, **kw):
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
        table = pa.Table.from_pandas(df)
        feather.write_feather(table, str(path))
    except ImportError:
        raise ImportError("Arrow format requires: pip install pyarrow")

def _write_json(df, path, **kw):
    df.to_json(path, orient="records", indent=2, **kw)

def _write_stata(df, path, **kw):
    # Stata can't handle string columns > 244 chars or certain dtypes
    df_stata = df.copy()
    for col in df_stata.select_dtypes(include="object").columns:
        df_stata[col] = df_stata[col].astype(str).str[:244]
    df_stata.to_stata(path, write_index=False, version=118, **kw)

def _write_excel(df, path, **kw):
    try:
        df.to_excel(path, index=False, engine="openpyxl", **kw)
    except ImportError:
        raise ImportError("Excel format requires: pip install openpyxl")


# ── Read ──────────────────────────────────────────────────────────────────────

def read(path: str | Path, fmt: str | None = None, **kwargs) -> pd.DataFrame:
    """
    Read a DataFrame from file.

    Parameters
    ----------
    path : input file path
    fmt  : format override (inferred from extension if None)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = fmt or _infer_format(path)
    readers = {
        "csv":     lambda p, **kw: pd.read_csv(p, **kw),
        "parquet": lambda p, **kw: pd.read_parquet(p, **kw),
        "arrow":   _read_arrow,
        "json":    lambda p, **kw: pd.read_json(p, **kw),
        "stata":   lambda p, **kw: pd.read_stata(p, **kw),
        "sas":     _read_sas,
        "excel":   lambda p, **kw: pd.read_excel(p, **kw),
    }

    if fmt not in readers:
        raise ValueError(f"Unsupported format '{fmt}'. Available: {', '.join(readers)}")

    return readers[fmt](path, **kwargs)


def _read_arrow(path, **kw):
    try:
        import pyarrow.feather as feather
        return feather.read_feather(str(path)).to_pandas()
    except ImportError:
        raise ImportError("Arrow format requires: pip install pyarrow")

def _read_sas(path, **kw):
    try:
        return pd.read_sas(str(path), format="sas7bdat", encoding="latin-1", **kw)
    except Exception as e:
        raise RuntimeError(f"Could not read SAS file: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

_EXT_MAP = {
    ".csv":      "csv",
    ".parquet":  "parquet",
    ".pq":       "parquet",
    ".feather":  "arrow",
    ".arrow":    "arrow",
    ".ipc":      "arrow",
    ".json":     "json",
    ".dta":      "stata",
    ".sas7bdat": "sas",
    ".xlsx":     "excel",
    ".xls":      "excel",
}

def _infer_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in _EXT_MAP:
        raise ValueError(
            f"Cannot infer format from extension '{ext}'. "
            f"Supported: {', '.join(_EXT_MAP)}"
        )
    return _EXT_MAP[ext]


def supported_formats() -> list[str]:
    """Return list of all supported format strings."""
    return sorted(set(_EXT_MAP.values()))
