# Bulk Data Download Guide

Some datasets require manual download because the source doesn't have a simple API.
This guide covers step-by-step instructions for each one.

Once downloaded, place the file in `~/.syndatakit/cache/` as a `.parquet` file
and syndatakit will use it automatically.

---

## HMDA Mortgage Applications

**Source:** Consumer Financial Protection Bureau  
**Size:** ~3 GB (14 million rows)  
**Time:** 5–10 minutes download

1. Go to **ffiec.cfpb.gov/data-download**
2. Click **LAR** (Loan Application Register)
3. Select **Year: 2022**, **Geography: Nationwide**
4. Click **Download**  
5. Unzip the file — you'll get a `.txt` or `.csv`

Then convert and cache with Python:

```python
import pandas as pd
from syndatakit.catalog.downloader import cache_path

# Load (use chunks for memory efficiency)
chunks = []
for chunk in pd.read_csv("hmda_2022_nationwide_lar.txt",
                          sep="|", chunksize=100_000,
                          usecols=["loan_amount","income","action_taken",
                                   "loan_purpose","derived_dwelling_category",
                                   "debt_to_income_ratio","state_code"]):
    chunk = chunk.rename(columns={
        "income":                    "applicant_income",
        "derived_dwelling_category": "property_type",
        "debt_to_income_ratio":      "debt_to_income",
        "state_code":                "state",
    })
    chunk = chunk.dropna()
    chunks.append(chunk)

df = pd.concat(chunks).sample(50_000, random_state=42)
df.to_parquet(cache_path("hmda"), index=False)
print(f"Cached {len(df):,} rows")
```

---

## FDIC Bank Call Reports

**Source:** FDIC Statistics on Depository Institutions  
**Size:** ~50 MB  
**Time:** 2–3 minutes

The FDIC has a free API — no manual download needed:

```python
from syndatakit.catalog.downloader import download
df = download("fdic")
```

Or manually:
1. Go to **banks.data.fdic.gov**
2. Click **Download Data** → **Financial Data** → **All Active Banks**
3. Select fields: `ASSET, DEP, LNLSNET, RBC1AAJ, NIM, ROA, ROE, P3ASSET, LNLSDEPR`

---

## BLS Employment & Wages

**Source:** Bureau of Labor Statistics QCEW  
**Size:** ~500 MB  
**Time:** 5 minutes

1. Go to **bls.gov/cew/downloadable-data.htm**
2. Click **"Annual Averages"** → **2022** → Download the CSV zip
3. Unzip — you get one file per state

```python
import pandas as pd, glob
from syndatakit.catalog.downloader import cache_path

dfs = []
for f in glob.glob("2022.annual.by_area/*.csv"):
    try:
        dfs.append(pd.read_csv(f, low_memory=False))
    except Exception:
        pass

df = pd.concat(dfs)
# Keep the columns we need
cols = ["industry_code","own_code","st","avg_wkly_wage","annual_avg_emplvl","qtrly_estabs"]
df = df[[c for c in cols if c in df.columns]].dropna()
df.to_parquet(cache_path("bls"), index=False)
print(f"Cached {len(df):,} rows")
```

---

## SEC EDGAR Financial Statements

**Source:** SEC EDGAR XBRL  
**Size:** ~2 GB  
**Time:** 10–15 minutes

```python
import urllib.request, zipfile, io, json
import pandas as pd
from syndatakit.catalog.downloader import cache_path

# Download company facts bulk file
url = "https://data.sec.gov/submissions/companyfacts.zip"
print("Downloading SEC bulk file (~2GB)...")
# Note: this is large — consider using the individual company API instead
# for a smaller targeted sample:

# Individual company API (much faster):
import urllib.request, json
ciks = ["0000320193", "0000789019", "0001318605"]  # Apple, Microsoft, Tesla etc.
rows = []
for cik in ciks:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        with urllib.request.urlopen(url) as r:
            facts = json.loads(r.read())
        # Extract revenue, net income etc from us-gaap
        # ... (parsing logic)
    except Exception:
        pass

# Or download the pre-processed quarterly zips:
# https://data.sec.gov/Archives/edgar/full-index/2023/QTR4/company.gz
```

---

## IRS Statistics of Income

**Source:** IRS SOI  
**Size:** ~5 MB (Excel files)  
**Time:** 2 minutes

1. Go to **irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income**
2. Download the Excel file for **Tax Year 2021**
3. Convert:

```python
import pandas as pd
from syndatakit.catalog.downloader import cache_path

df = pd.read_excel("21in11si.xlsx", skiprows=5)
# Clean columns, rename to syndatakit schema
df.to_parquet(cache_path("irs_soi"), index=False)
```

---

## Quick check — what's cached

```bash
syndatakit download status
```

Or in Python:
```python
from syndatakit.catalog.downloader import status
print(status().to_string(index=False))
```
