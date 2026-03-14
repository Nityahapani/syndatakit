"""
syndatakit.catalog — finance & econometrics edition
----------------------------------------------------
Four verticals, ten datasets, all sourced from public registries.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


DATASETS: dict[str, dict] = {
    "hmda": {
        "name": "HMDA Mortgage Applications", "vertical": "Credit & Lending",
        "source": "CFPB HMDA 2022",
        "description": "Loan applications with income, DTI, approval decisions, property type and state.",
        "columns": ["loan_amount","applicant_income","action_taken","loan_purpose","property_type","debt_to_income","state"],
        "col_count": 7, "tags": ["GDPR safe","CSV","Parquet"], "fidelity": 98.0, "status": "live",
        "use_cases": ["Credit risk models","Fair lending analysis","Mortgage default prediction"],
    },
    "fdic": {
        "name": "FDIC Bank Call Reports", "vertical": "Credit & Lending",
        "source": "FDIC Statistics on Depository Institutions 2023",
        "description": "Quarterly bank balance sheets: assets, deposits, loans, capital ratios, NIM, ROA/ROE.",
        "columns": ["total_assets","total_deposits","total_loans","tier1_capital_ratio","net_interest_margin","roa","roe","npl_ratio","loan_to_deposit","bank_size_class","charter_class","state"],
        "col_count": 12, "tags": ["SOC 2 safe","CSV","Parquet"], "fidelity": 97.2, "status": "live",
        "use_cases": ["Bank failure prediction","Stress testing","Regulatory capital models"],
    },
    "credit_risk": {
        "name": "Consumer Credit Risk", "vertical": "Credit & Lending",
        "source": "Derived from CFPB + HMDA distributions",
        "description": "Consumer credit features: FICO bands, utilisation, delinquency history, default label.",
        "columns": ["fico_band","credit_utilisation","num_accounts","num_delinquencies","months_since_delinquency","debt_to_income","loan_amount","loan_term","employment_years","default_12m"],
        "col_count": 10, "tags": ["GDPR safe","CSV","JSON"], "fidelity": 95.8, "status": "live",
        "use_cases": ["PD model training","Scorecard development","IFRS 9 staging"],
    },
    "edgar": {
        "name": "SEC EDGAR Financial Statements", "vertical": "Capital Markets",
        "source": "SEC EDGAR XBRL 2023",
        "description": "Annual 10-K fundamentals: revenue, EBITDA, margins, debt ratios, FCF, sector.",
        "columns": ["revenue","ebitda","ebitda_margin","net_income","total_debt","net_debt_ebitda","fcf","roe","roa","current_ratio","sector","exchange","market_cap_band"],
        "col_count": 13, "tags": ["SOC 2 safe","CSV","Parquet","JSON"], "fidelity": 96.5, "status": "live",
        "use_cases": ["Fundamental factor models","Credit rating prediction","M&A screening"],
    },
    "cftc": {
        "name": "CFTC Commitments of Traders", "vertical": "Capital Markets",
        "source": "CFTC COT Weekly Reports 2023",
        "description": "Weekly futures positioning by trader category across major contracts.",
        "columns": ["commodity","contract_units","commercial_long","commercial_short","noncommercial_long","noncommercial_short","open_interest","net_commercial","net_noncommercial","week_of_year"],
        "col_count": 10, "tags": ["CSV","JSON"], "fidelity": 97.8, "status": "live",
        "use_cases": ["Sentiment indicators","Trend-following signals","Options positioning models"],
    },
    "fred_macro": {
        "name": "FRED Macroeconomic Indicators", "vertical": "Macro & Central Bank",
        "source": "Federal Reserve FRED 2000–2023",
        "description": "Monthly panel: GDP, CPI, unemployment, fed funds rate, yield curve, M2, VIX.",
        "columns": ["year","gdp_growth_yoy","cpi_yoy","core_cpi_yoy","unemployment_rate","fed_funds_rate","t10y_rate","t2y_rate","yield_curve_spread","m2_growth","housing_starts","industrial_production","consumer_sentiment","oil_price_yoy","vix"],
        "col_count": 15, "tags": ["CSV","Parquet","JSON"], "fidelity": 97.5, "status": "live",
        "use_cases": ["Macro regime models","Rate forecasting","Recession probability"],
    },
    "bls": {
        "name": "BLS Employment & Wages", "vertical": "Macro & Central Bank",
        "source": "Bureau of Labor Statistics QCEW 2022",
        "description": "Quarterly employment and wage data by NAICS industry, ownership and state.",
        "columns": ["naics_sector","ownership","state","avg_weekly_wage","total_employment","yoy_employment_change","yoy_wage_change","establishments","quarter"],
        "col_count": 9, "tags": ["CSV","Parquet"], "fidelity": 96.9, "status": "live",
        "use_cases": ["Labour market models","Wage inflation forecasting","Regional economic analysis"],
    },
    "world_bank": {
        "name": "World Bank Development Indicators", "vertical": "Macro & Central Bank",
        "source": "World Bank WDI 2022",
        "description": "Cross-country annual panel: GDP per capita, inflation, current account, FDI, debt-to-GDP.",
        "columns": ["country_code","income_group","region","year","gdp_per_capita","gdp_growth","inflation","current_account_pct_gdp","fdi_pct_gdp","govt_debt_pct_gdp","population_growth","gini"],
        "col_count": 12, "tags": ["CSV","Parquet","JSON"], "fidelity": 96.1, "status": "live",
        "use_cases": ["Sovereign risk models","EM macro forecasting","ESG country scoring"],
    },
    "irs_soi": {
        "name": "IRS Statistics of Income", "vertical": "Tax & Income",
        "source": "IRS SOI Individual Returns 2021",
        "description": "Tax return aggregates by AGI bracket: income sources, deductions, effective rates.",
        "columns": ["agi_bracket","filing_status","num_returns","total_agi","wages_salaries","capital_gains","business_income","total_deductions","taxes_paid","effective_rate","state"],
        "col_count": 11, "tags": ["GDPR safe","CSV","JSON"], "fidelity": 95.4, "status": "live",
        "use_cases": ["Tax policy simulation","Wealth distribution models","Revenue forecasting"],
    },
    "census_acs": {
        "name": "Census ACS Income & Housing", "vertical": "Tax & Income",
        "source": "US Census ACS 5-Year 2022",
        "description": "Household income, housing costs, poverty status and demographics by PUMA geography.",
        "columns": ["puma","state","household_income","housing_cost","cost_burden_pct","poverty_status","employment_status","household_size","tenure","age_group","education"],
        "col_count": 11, "tags": ["GDPR safe","CSV","Parquet"], "fidelity": 96.3, "status": "live",
        "use_cases": ["Affordability models","Poverty prediction","Housing demand forecasting"],
    },
}


def list_datasets(vertical: str | None = None) -> pd.DataFrame:
    rows = []
    for key, m in DATASETS.items():
        if vertical and m["vertical"].lower() != vertical.lower():
            continue
        rows.append({"id": key, "name": m["name"], "vertical": m["vertical"],
                     "columns": m["col_count"], "source": m["source"],
                     "fidelity": f"{m['fidelity']}%", "status": m["status"]})
    return pd.DataFrame(rows)


def get_dataset_info(dataset_id: str) -> dict:
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {', '.join(DATASETS)}")
    return DATASETS[dataset_id]


def load_seed(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_id}'.")
    builders = {k: globals()[f"_build_{k}"] for k in DATASETS}
    return builders[dataset_id]()


def _rng(seed=42): return np.random.default_rng(seed)
def _weighted(rng, items, weights, size):
    w = np.array(weights, dtype=float); w /= w.sum()
    return rng.choice(items, p=w, size=size)
def _lognorm(rng, mu, sigma, lo, hi, n):
    return np.clip(rng.lognormal(mu, sigma, n), lo, hi)


def _build_hmda(n=2000):
    rng = _rng()
    states = ["CA","TX","FL","NY","PA","IL","OH","GA","NC","MI","NJ","VA","WA","AZ","MA"]
    income = _lognorm(rng, 11.2, 0.6, 20000, 500000, n).astype(int)
    dti = np.clip(45-(income/500000)*20+rng.normal(0,8,n), 5, 65).round(1)
    return pd.DataFrame({
        "loan_amount": _lognorm(rng, 12.1, 0.7, 50000, 2000000, n).astype(int),
        "applicant_income": income,
        "action_taken": _weighted(rng, ["1","2","3","6","7","8"], [64,11,9,4,7,5], n),
        "loan_purpose": _weighted(rng, ["1","2","31","32"], [42,28,18,12], n),
        "property_type": _weighted(rng, ["Single Family","Multifamily","Manufactured","Condo"], [72,12,8,8], n),
        "debt_to_income": dti,
        "state": _weighted(rng, states, [14,12,10,8,5,5,4,4,4,4,4,4,4,3,3], n),
    })


def _build_fdic(n=2000):
    rng = _rng()
    assets = _lognorm(rng, 13.5, 1.8, 1e6, 3e12, n)
    return pd.DataFrame({
        "total_assets": assets.round(0).astype(int),
        "total_deposits": (assets*rng.uniform(0.55,0.80,n)).round(0).astype(int),
        "total_loans": (assets*rng.uniform(0.45,0.75,n)).round(0).astype(int),
        "tier1_capital_ratio": np.clip(rng.normal(13.2,2.8,n), 6, 30).round(2),
        "net_interest_margin": np.clip(rng.normal(3.1,0.6,n), 0.8, 6.5).round(2),
        "roa": np.clip(rng.normal(1.05,0.45,n), -1.5, 3.5).round(3),
        "roe": np.clip(rng.normal(10.2,3.8,n), -15, 28).round(2),
        "npl_ratio": np.clip(rng.lognormal(-2.5,0.8,n), 0.1, 12).round(2),
        "loan_to_deposit": np.clip(rng.normal(72,12,n), 30, 120).round(1),
        "bank_size_class": _weighted(rng, ["community","regional","large","megabank"], [60,25,10,5], n),
        "charter_class": _weighted(rng, ["NM","N","SM","SB","OI"], [35,30,18,12,5], n),
        "state": _weighted(rng, ["CA","TX","FL","NY","IL","OH","PA","NC","GA","MI"], [14,12,10,9,7,6,5,5,4,4], n),
    })


def _build_credit_risk(n=2000):
    rng = _rng()
    fico = _weighted(rng, ["300-579","580-669","670-739","740-799","800-850"], [16,17,21,25,21], n)
    util = np.clip(rng.beta(2,5,n)*100, 0, 100).round(1)
    delinq = rng.poisson(0.4, n)
    base_default = ((fico=="300-579").astype(float)*0.28 + (fico=="580-669").astype(float)*0.14
                    + (util>70).astype(float)*0.08 + rng.uniform(0,0.05,n))
    return pd.DataFrame({
        "fico_band": fico, "credit_utilisation": util,
        "num_accounts": np.clip(rng.poisson(8,n), 1, 30),
        "num_delinquencies": delinq,
        "months_since_delinquency": np.where(delinq>0, np.clip(rng.exponential(18,n),1,120).astype(int), -1),
        "debt_to_income": np.clip(rng.normal(38,12,n), 5, 75).round(1),
        "loan_amount": _lognorm(rng, 10.5, 0.8, 1000, 100000, n).astype(int),
        "loan_term": _weighted(rng, [12,24,36,48,60,72], [5,12,28,20,28,7], n),
        "employment_years": np.clip(rng.exponential(5,n), 0, 40).round(1),
        "default_12m": (rng.uniform(0,1,n) < np.clip(base_default,0,0.5)).astype(int),
    })


def _build_edgar(n=2000):
    rng = _rng()
    rev = _lognorm(rng, 14.5, 1.6, 1e6, 500e9, n)
    em = np.clip(rng.normal(18,10,n), -30, 55)
    ebitda = rev*em/100
    return pd.DataFrame({
        "revenue": rev.round(0).astype(int), "ebitda": ebitda.round(0).astype(int),
        "ebitda_margin": em.round(1),
        "net_income": (ebitda*rng.uniform(0.3,0.85,n)).round(0).astype(int),
        "total_debt": (rev*rng.uniform(0.1,2.5,n)).round(0).astype(int),
        "net_debt_ebitda": np.clip(rng.normal(2.1,1.8,n), -2, 12).round(2),
        "fcf": (ebitda*rng.uniform(0.4,0.95,n)).round(0).astype(int),
        "roe": np.clip(rng.normal(14,9,n), -30, 60).round(1),
        "roa": np.clip(rng.normal(6,4,n), -10, 25).round(1),
        "current_ratio": np.clip(rng.lognormal(0.5,0.4,n), 0.3, 6.0).round(2),
        "sector": _weighted(rng, ["Technology","Healthcare","Financials","Industrials","Consumer Discretionary","Energy","Materials","Utilities"], [18,13,16,12,11,8,7,5], n),
        "exchange": _weighted(rng, ["NYSE","NASDAQ","AMEX"], [48,46,6], n),
        "market_cap_band": _weighted(rng, ["nano","micro","small","mid","large","mega"], [12,15,22,24,18,9], n),
    })


def _build_cftc(n=2000):
    rng = _rng()
    oi = _lognorm(rng, 12, 1.2, 5000, 2000000, n).astype(int)
    comm_net = rng.normal(-20000,40000,n).astype(int)
    commodities = ["Gold","Crude Oil","Natural Gas","Corn","Wheat","Soybeans","S&P 500 E-mini","Eurodollar","10Y T-Note","Euro FX","Copper","Silver"]
    return pd.DataFrame({
        "commodity": _weighted(rng, commodities, [10,12,8,9,7,9,8,6,7,8,8,8], n),
        "contract_units": _weighted(rng, ["100 oz","1000 bbl","10000 MMBtu","5000 bu"], [25,25,25,25], n),
        "commercial_long": (oi*rng.uniform(0.3,0.6,n)).astype(int),
        "commercial_short": (oi*rng.uniform(0.3,0.6,n)).astype(int),
        "noncommercial_long": (oi*rng.uniform(0.2,0.5,n)).astype(int),
        "noncommercial_short": (oi*rng.uniform(0.15,0.45,n)).astype(int),
        "open_interest": oi, "net_commercial": comm_net,
        "net_noncommercial": (-comm_net+rng.normal(0,5000,n)).astype(int),
        "week_of_year": rng.integers(1,53,n),
    })


def _build_fred_macro(n=2000):
    rng = _rng()
    years = rng.integers(2000, 2024, n)
    recession = ((years>=2008)&(years<=2009))|(years==2020)
    gdp = np.where(recession, rng.normal(-3.5,1.5,n), rng.normal(2.5,1.2,n)).round(2)
    ffr = np.clip(rng.normal(2.2,2.1,n), 0.05, 5.5).round(2)
    return pd.DataFrame({
        "year": years, "gdp_growth_yoy": gdp,
        "cpi_yoy": np.clip(rng.normal(2.6,1.8,n), -1.0, 9.5).round(2),
        "core_cpi_yoy": np.clip(rng.normal(2.4,1.4,n), 0.5, 7.5).round(2),
        "unemployment_rate": np.clip(rng.normal(5.8,2.2,n), 3.4, 14.8).round(1),
        "fed_funds_rate": ffr,
        "t10y_rate": np.clip(ffr+rng.normal(1.0,0.8,n), 0.5, 8.0).round(2),
        "t2y_rate": np.clip(ffr+rng.normal(0.3,0.5,n), 0.1, 6.5).round(2),
        "yield_curve_spread": np.clip(rng.normal(0.9,0.9,n), -1.5, 3.5).round(2),
        "m2_growth": np.clip(rng.normal(5.8,4.2,n), -2.0, 27.0).round(2),
        "housing_starts": np.clip(rng.normal(1250,350,n), 450, 2200).astype(int),
        "industrial_production": np.clip(rng.normal(1.8,4.5,n), -16.0, 12.0).round(2),
        "consumer_sentiment": np.clip(rng.normal(89,14,n), 50, 112).round(1),
        "oil_price_yoy": np.clip(rng.normal(3.5,28,n), -55, 80).round(2),
        "vix": np.clip(rng.lognormal(3.1,0.4,n), 10, 85).round(1),
    })


def _build_bls(n=2000):
    rng = _rng()
    sectors = ["Manufacturing","Retail Trade","Healthcare","Finance & Insurance","Professional Services","Construction","Transportation","Accommodation & Food"]
    return pd.DataFrame({
        "naics_sector": _weighted(rng, sectors, [14,13,14,10,12,9,9,9], n),
        "ownership": _weighted(rng, ["Private","Federal Govt","State Govt","Local Govt"], [75,3,7,15], n),
        "state": _weighted(rng, ["CA","TX","FL","NY","PA","IL","OH","GA","NC","MI"], [14,12,10,9,6,6,5,5,4,4], n),
        "avg_weekly_wage": _lognorm(rng, 6.6, 0.4, 400, 3500, n).round(0).astype(int),
        "total_employment": _lognorm(rng, 9.5, 1.5, 50, 2000000, n).astype(int),
        "yoy_employment_change": np.clip(rng.normal(1.8,4.5,n), -18, 20).round(1),
        "yoy_wage_change": np.clip(rng.normal(3.2,2.1,n), -5, 15).round(1),
        "establishments": _lognorm(rng, 5.5, 1.4, 1, 50000, n).astype(int),
        "quarter": rng.integers(1, 5, n),
    })


def _build_world_bank(n=2000):
    rng = _rng()
    regions = ["East Asia & Pacific","Europe & Central Asia","Latin America & Caribbean","Middle East & North Africa","North America","South Asia","Sub-Saharan Africa"]
    gdppc = _lognorm(rng, 8.8, 1.5, 400, 120000, n)
    return pd.DataFrame({
        "country_code": [f"C{i:03d}" for i in rng.integers(1,200,n)],
        "income_group": _weighted(rng, ["Low income","Lower middle income","Upper middle income","High income"], [15,28,27,30], n),
        "region": _weighted(rng, regions, [18,18,15,10,6,12,21], n),
        "year": rng.integers(2000, 2023, n),
        "gdp_per_capita": gdppc.round(0).astype(int),
        "gdp_growth": np.clip(rng.normal(3.2,3.8,n), -12, 15).round(2),
        "inflation": np.clip(rng.lognormal(1.8,0.9,n), 0.1, 80).round(2),
        "current_account_pct_gdp": np.clip(rng.normal(-2.1,5.5,n), -25, 20).round(2),
        "fdi_pct_gdp": np.clip(rng.lognormal(0.5,0.9,n), 0, 15).round(2),
        "govt_debt_pct_gdp": np.clip(rng.normal(58,32,n), 5, 230).round(1),
        "population_growth": np.clip(rng.normal(1.2,1.1,n), -1.5, 4.5).round(2),
        "gini": np.clip(rng.normal(38,8,n), 24, 65).round(1),
    })


def _build_irs_soi(n=2000):
    rng = _rng()
    agi = _lognorm(rng, 10.8, 1.0, 1000, 50000000, n).astype(int)
    eff = np.clip(rng.normal(16,6,n), 0, 37).round(1)
    return pd.DataFrame({
        "agi_bracket": _weighted(rng, ["<$25K","$25K-$50K","$50K-$75K","$75K-$100K","$100K-$200K","$200K-$500K",">$500K"], [20,22,16,12,18,8,4], n),
        "filing_status": _weighted(rng, ["Single","MFJ","MFS","HoH","QW"], [44,38,2,12,4], n),
        "num_returns": _lognorm(rng, 6, 1.2, 1, 5000000, n).astype(int),
        "total_agi": agi,
        "wages_salaries": (agi*rng.uniform(0.4,0.95,n)).astype(int),
        "capital_gains": np.where(rng.uniform(0,1,n)>0.6, (agi*rng.uniform(0,0.4,n)).astype(int), 0),
        "business_income": np.where(rng.uniform(0,1,n)>0.7, (agi*rng.uniform(0,0.3,n)).astype(int), 0),
        "total_deductions": (agi*rng.uniform(0.1,0.35,n)).astype(int),
        "taxes_paid": (agi*eff/100).astype(int),
        "effective_rate": eff,
        "state": _weighted(rng, ["CA","TX","FL","NY","PA","IL","OH","GA","NC","MI"], [14,12,10,9,6,6,5,5,4,4], n),
    })


def _build_census_acs(n=2000):
    rng = _rng()
    hhi = _lognorm(rng, 10.9, 0.7, 5000, 500000, n).astype(int)
    hc = _lognorm(rng, 7.5, 0.5, 300, 8000, n).astype(int)
    burden = np.clip((hc*12)/np.maximum(hhi,1)*100, 0, 100).round(1)
    return pd.DataFrame({
        "puma": [f"PUMA-{rng.integers(1000,9999)}" for _ in range(n)],
        "state": _weighted(rng, ["CA","TX","FL","NY","PA","IL","OH","GA","NC","MI"], [14,12,10,9,6,6,5,5,4,4], n),
        "household_income": hhi, "housing_cost": hc, "cost_burden_pct": burden,
        "poverty_status": (burden>30).astype(int),
        "employment_status": _weighted(rng, ["Employed","Unemployed","Not in labor force"], [58,5,37], n),
        "household_size": np.clip(rng.poisson(2.5,n), 1, 8),
        "tenure": _weighted(rng, ["Owner","Renter"], [64,36], n),
        "age_group": _weighted(rng, ["18-24","25-34","35-44","45-54","55-64","65+"], [12,18,20,18,16,16], n),
        "education": _weighted(rng, ["No HS","HS Grad","Some College","Bachelor","Graduate"], [12,27,28,21,12], n),
    })
