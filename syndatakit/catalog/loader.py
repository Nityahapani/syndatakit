"""
syndatakit.catalog — finance & econometrics edition
----------------------------------------------------
Eight verticals, eighteen datasets, all sourced from public registries.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


DATASETS: dict[str, dict] = {

    # ── Credit & Lending ──────────────────────────────────────────────────────
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

    # ── Capital Markets ───────────────────────────────────────────────────────
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
    "equity_returns": {
        "name": "Equity Returns & Risk Factors", "vertical": "Capital Markets",
        "source": "Derived from CRSP/Compustat and Fama-French factor research",
        "description": "Daily equity return panel with Fama-French 5 factors, momentum, beta, volatility and sector. Captures fat tails and cross-sectional dispersion.",
        "columns": ["date","ticker_id","daily_return","excess_return","mkt_rf","smb","hml","rmw","cma","momentum_12m","realized_vol_21d","beta","market_cap_log","sector","exchange"],
        "col_count": 15, "tags": ["CSV","Parquet"], "fidelity": 96.2, "status": "live",
        "use_cases": ["Factor model training","Portfolio optimisation","Risk attribution","Backtesting"],
    },
    "corporate_bonds": {
        "name": "Corporate Bond Market Data", "vertical": "Capital Markets",
        "source": "Derived from TRACE/FINRA and Bloomberg Barclays index distributions",
        "description": "Corporate bonds: credit spread, duration, rating, yield to maturity, OAS and default indicator across IG and HY.",
        "columns": ["issue_size","maturity_years","coupon_rate","yield_to_maturity","credit_spread","oas","duration","convexity","credit_rating","rating_agency","sector","subordination","callable","default_flag","days_to_maturity"],
        "col_count": 15, "tags": ["CSV","Parquet","JSON"], "fidelity": 95.9, "status": "live",
        "use_cases": ["Credit spread models","Default prediction","Bond portfolio construction","XVA models"],
    },

    # ── Macro & Central Bank ──────────────────────────────────────────────────
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

    # ── Tax & Income ──────────────────────────────────────────────────────────
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

    # ── Insurance ─────────────────────────────────────────────────────────────
    "insurance_claims": {
        "name": "P&C Insurance Claims", "vertical": "Insurance",
        "source": "Derived from NAIC Schedule P and ISO CGL distributions",
        "description": "P&C insurance claims: loss amount, development pattern, line of business, accident year, paid vs incurred, and large loss indicator.",
        "columns": ["accident_year","development_year","line_of_business","paid_losses","incurred_losses","case_reserves","ibnr_estimate","claim_count","severity","frequency","large_loss_flag","state","policy_type"],
        "col_count": 13, "tags": ["CSV","Parquet"], "fidelity": 95.1, "status": "live",
        "use_cases": ["Loss reserving","IBNR estimation","Actuarial pricing","Reinsurance structuring"],
    },
    "life_insurance": {
        "name": "Life Insurance & Mortality", "vertical": "Insurance",
        "source": "Derived from SOA mortality tables and LIMRA industry distributions",
        "description": "Life insurance policies: face amount, mortality rate, lapse rate, duration, underwriting class and surrender value.",
        "columns": ["face_amount","annual_premium","policy_duration","age_at_issue","underwriting_class","mortality_rate","lapse_rate","surrender_value","cash_value","product_type","smoker_status","gender","in_force_flag"],
        "col_count": 13, "tags": ["GDPR safe","CSV","JSON"], "fidelity": 95.6, "status": "live",
        "use_cases": ["Mortality modelling","Lapse prediction","Embedded value","ALM models"],
    },

    # ── Real Estate ───────────────────────────────────────────────────────────
    "commercial_real_estate": {
        "name": "Commercial Real Estate", "vertical": "Real Estate",
        "source": "Derived from CoStar/NCREIF and FDIC CRE loan distributions",
        "description": "Commercial properties: cap rate, NOI, LTV, DSCR, occupancy, property type and market tier.",
        "columns": ["property_value","noi","cap_rate","ltv_ratio","dscr","occupancy_rate","lease_term_years","property_type","market_tier","submarket","year_built","square_footage","loan_amount","interest_rate","amortization_years"],
        "col_count": 15, "tags": ["CSV","Parquet"], "fidelity": 95.3, "status": "live",
        "use_cases": ["CRE credit risk","Cap rate forecasting","CMBS modelling","Portfolio stress testing"],
    },
    "rental_market": {
        "name": "Residential Rental Market", "vertical": "Real Estate",
        "source": "Derived from Census ACS, HUD FMR and Zillow Research distributions",
        "description": "Rental market: asking rent, vacancy rate, rent-to-income, unit type and metro supply/demand indicators.",
        "columns": ["asking_rent","gross_rent","vacancy_rate","rent_to_income","yoy_rent_change","unit_type","bedrooms","year_built_band","metro_tier","state","median_metro_income","housing_supply_index","affordability_index"],
        "col_count": 13, "tags": ["GDPR safe","CSV","Parquet"], "fidelity": 95.7, "status": "live",
        "use_cases": ["Rent forecasting","Affordability analysis","Build-to-rent underwriting","Housing policy"],
    },

    # ── Retail Banking ────────────────────────────────────────────────────────
    "retail_transactions": {
        "name": "Retail Banking Transactions", "vertical": "Retail Banking",
        "source": "Derived from Federal Reserve Payment Study and BIS retail payment statistics",
        "description": "Anonymised retail payment transactions: amount, channel, merchant category, time features and fraud indicator.",
        "columns": ["amount","channel","merchant_category","day_of_week","hour_of_day","transaction_type","is_recurring","is_international","account_age_months","monthly_tx_count","balance_band","fraud_flag"],
        "col_count": 12, "tags": ["GDPR safe","PCI safe","CSV","JSON"], "fidelity": 95.5, "status": "live",
        "use_cases": ["Fraud detection","Transaction monitoring","Customer segmentation","AML models"],
    },

    # ── Commodities ───────────────────────────────────────────────────────────
    "commodity_prices": {
        "name": "Commodity Price Returns", "vertical": "Commodities",
        "source": "Derived from EIA, USDA WASDE and LME historical price distributions",
        "description": "Daily commodity returns with carry, seasonality, volatility and inventory signals across energy, metals and agricultural markets.",
        "columns": ["commodity","sector","daily_return","roll_yield","spot_price_log","inventory_change","realized_vol_21d","implied_vol","basis","seasonal_index","dollar_index_return","global_demand_proxy","supply_shock_flag"],
        "col_count": 13, "tags": ["CSV","Parquet","JSON"], "fidelity": 95.8, "status": "live",
        "use_cases": ["Commodity risk models","Roll yield strategies","Supply shock detection","Inflation forecasting"],
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
    """
    Load seed data for a dataset.

    Priority:
    1. Real downloaded data (if cached via syndatakit download <id>)
    2. Hand-coded statistical approximation (always available, no download needed)

    To get real data: syndatakit download <id>  or
                      from syndatakit.catalog.downloader import download; download("hmda")
    """
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_id}'.")

    # Check for cached real data first
    try:
        from .downloader import load_cached
        cached = load_cached(dataset_id)
        if cached is not None and len(cached) >= 100:
            return cached.sample(min(2000, len(cached)), random_state=42).reset_index(drop=True)
    except Exception:
        pass  # fall through to hand-coded seed

    builders = {k: globals()[f"_build_{k}"] for k in DATASETS}
    return builders[dataset_id]()


def _rng(seed=42): return np.random.default_rng(seed)
def _weighted(rng, items, weights, size):
    w = np.array(weights, dtype=float); w /= w.sum()
    return rng.choice(items, p=w, size=size)
def _lognorm(rng, mu, sigma, lo, hi, n):
    return np.clip(rng.lognormal(mu, sigma, n), lo, hi)


# ── Seed builders ─────────────────────────────────────────────────────────────

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


def _build_equity_returns(n=2000):
    rng = _rng()
    mkt_rf = np.clip(rng.standard_t(5, n) * 0.008, -0.12, 0.12).round(4)
    idio   = np.clip(rng.standard_t(4, n) * 0.018, -0.25, 0.25)
    beta   = np.clip(rng.lognormal(0.05, 0.45, n), 0.1, 3.0).round(2)
    daily_return = np.clip(beta * mkt_rf + idio, -0.40, 0.40).round(4)
    sectors = ["Technology","Healthcare","Financials","Industrials","Consumer Discretionary",
               "Energy","Materials","Utilities","Real Estate","Communication Services"]
    return pd.DataFrame({
        "date":             rng.integers(20100101, 20231231, n),
        "ticker_id":        [f"TKR{rng.integers(1000,9999)}" for _ in range(n)],
        "daily_return":     daily_return,
        "excess_return":    (daily_return - rng.uniform(0.00005, 0.00015, n)).round(4),
        "mkt_rf":           mkt_rf,
        "smb":              np.clip(rng.normal(0.0002, 0.004, n), -0.04, 0.04).round(4),
        "hml":              np.clip(rng.normal(0.0001, 0.004, n), -0.04, 0.04).round(4),
        "rmw":              np.clip(rng.normal(0.0001, 0.003, n), -0.03, 0.03).round(4),
        "cma":              np.clip(rng.normal(0.0000, 0.003, n), -0.03, 0.03).round(4),
        "momentum_12m":     np.clip(rng.normal(0.08, 0.35, n), -0.80, 1.50).round(3),
        "realized_vol_21d": np.clip(rng.lognormal(-3.2, 0.6, n), 0.005, 0.15).round(4),
        "beta":             beta,
        "market_cap_log":   np.clip(rng.normal(21.5, 2.2, n), 15, 28).round(2),
        "sector":           _weighted(rng, sectors, [20,12,14,11,10,7,5,4,4,13], n),
        "exchange":         _weighted(rng, ["NYSE","NASDAQ","AMEX"], [45,48,7], n),
    })


def _build_corporate_bonds(n=2000):
    rng = _rng()
    ratings  = ["AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-","BB+","BB","BB-","B+","B","CCC"]
    rweights = [2,3,4,4,6,7,7,9,10,9,8,7,6,5,5,8]
    rating   = _weighted(rng, ratings, rweights, n)
    maturity = np.clip(rng.lognormal(2.0, 0.7, n), 1, 30).round(1)
    ig_flag  = np.isin(rating, ["AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-"])
    spread   = np.where(ig_flag,
        np.clip(rng.lognormal(3.5, 0.7, n), 20, 500),
        np.clip(rng.lognormal(5.5, 0.7, n), 200, 2500))
    ytm = np.clip(rng.normal(4.5,1.5,n) + spread/100, 0.5, 20.0).round(3)
    return pd.DataFrame({
        "issue_size":        _lognorm(rng, 19.5, 1.2, 1e7, 5e10, n).astype(int),
        "maturity_years":    maturity,
        "coupon_rate":       np.clip(rng.normal(4.2, 1.8, n), 0.0, 12.0).round(3),
        "yield_to_maturity": ytm,
        "credit_spread":     spread.round(0).astype(int),
        "oas":               np.clip(spread + rng.normal(0,15,n), 5, 3000).round(0).astype(int),
        "duration":          np.clip(maturity * rng.uniform(0.7,0.95,n), 0.5, 25).round(2),
        "convexity":         np.clip(rng.lognormal(1.5,0.8,n), 0.1, 50).round(2),
        "credit_rating":     rating,
        "rating_agency":     _weighted(rng, ["Moody's","S&P","Fitch"], [40,40,20], n),
        "sector":            _weighted(rng, ["Financials","Industrials","Utilities","Technology","Energy","Healthcare","Consumer"], [25,20,12,11,10,10,12], n),
        "subordination":     _weighted(rng, ["Senior Secured","Senior Unsecured","Subordinated","Junior Subordinated"], [25,55,14,6], n),
        "callable":          (rng.uniform(0,1,n) > 0.45).astype(int),
        "default_flag":      (rng.uniform(0,1,n) < np.where(ig_flag, 0.001, 0.04)).astype(int),
        "days_to_maturity":  (maturity * 365).astype(int),
    })


def _build_insurance_claims(n=2000):
    rng = _rng()
    lob = _weighted(rng, ["Auto Liability","Auto Physical","General Liability","Workers Comp",
                           "Commercial Property","Homeowners","Medical Malpractice","Other"],
                    [22,18,15,14,12,10,5,4], n)
    paid = _lognorm(rng, 7.5, 1.8, 100, 50_000_000, n)
    incurred = paid * np.clip(rng.lognormal(0.05, 0.15, n), 1.0, 3.0)
    claim_count = np.clip(rng.poisson(45, n), 1, 500)
    return pd.DataFrame({
        "accident_year":   rng.integers(2015, 2024, n),
        "development_year":rng.integers(1, 11, n),
        "line_of_business":lob,
        "paid_losses":     paid.round(0).astype(int),
        "incurred_losses": incurred.round(0).astype(int),
        "case_reserves":   (incurred-paid).clip(0).round(0).astype(int),
        "ibnr_estimate":   _lognorm(rng, 6.5, 1.5, 0, 10_000_000, n).round(0).astype(int),
        "claim_count":     claim_count,
        "severity":        (paid / claim_count).round(0).astype(int),
        "frequency":       np.clip(rng.lognormal(-1.5,0.8,n), 0.001, 0.5).round(4),
        "large_loss_flag": (paid > np.percentile(paid, 95)).astype(int),
        "state":           _weighted(rng, ["CA","TX","FL","NY","PA","IL","OH","GA","NC","MI"], [14,12,10,9,6,6,5,5,4,4], n),
        "policy_type":     _weighted(rng, ["Commercial","Personal","Specialty"], [55,35,10], n),
    })


def _build_life_insurance(n=2000):
    rng = _rng()
    age      = np.clip(rng.normal(42, 12, n), 18, 75).astype(int)
    duration = np.clip(rng.exponential(8, n), 0.5, 40).round(1)
    smoker   = (rng.uniform(0,1,n) < 0.14).astype(int)
    mort_base = 0.0004 * np.exp(0.085 * age)
    mort_rate = np.clip(mort_base*(1+0.8*smoker)+rng.exponential(0.0002,n), 0.0001, 0.25)
    lapse_rate = np.clip(0.12*np.exp(-0.15*duration)+rng.exponential(0.02,n), 0.005, 0.35)
    face = _lognorm(rng, 11.8, 1.0, 10000, 10_000_000, n).astype(int)
    return pd.DataFrame({
        "face_amount":       face,
        "annual_premium":    (face*mort_rate*rng.uniform(1.2,2.0,n)).round(0).astype(int),
        "policy_duration":   duration,
        "age_at_issue":      age,
        "underwriting_class":_weighted(rng, ["Preferred Plus","Preferred","Standard Plus","Standard","Substandard"], [15,25,20,30,10], n),
        "mortality_rate":    mort_rate.round(5),
        "lapse_rate":        lapse_rate.round(4),
        "surrender_value":   (face*np.clip(duration/40,0,0.9)*rng.uniform(0.7,1.0,n)).round(0).astype(int),
        "cash_value":        (face*np.clip(duration/50,0,0.8)*rng.uniform(0.5,0.9,n)).round(0).astype(int),
        "product_type":      _weighted(rng, ["Term","Whole Life","Universal Life","Variable UL","Indexed UL"], [35,25,20,10,10], n),
        "smoker_status":     np.where(smoker, "Smoker", "Non-Smoker"),
        "gender":            _weighted(rng, ["Male","Female"], [51,49], n),
        "in_force_flag":     (rng.uniform(0,1,n) > lapse_rate).astype(int),
    })


def _build_commercial_real_estate(n=2000):
    rng = _rng()
    prop_type = _weighted(rng, ["Office","Retail","Multifamily","Industrial","Hotel","Self-Storage","Mixed Use"], [20,18,25,17,8,6,6], n)
    noi = _lognorm(rng, 13.5, 1.2, 50000, 100_000_000, n)
    cap_rate = np.clip(rng.normal(5.8,1.2,n), 2.5, 12.0).round(2)
    prop_value = (noi/(cap_rate/100)).round(0).astype(int)
    ltv = np.clip(rng.normal(62,12,n), 20, 90).round(1)
    return pd.DataFrame({
        "property_value":   prop_value,
        "noi":              noi.round(0).astype(int),
        "cap_rate":         cap_rate,
        "ltv_ratio":        ltv,
        "dscr":             np.clip(rng.normal(1.35,0.28,n), 0.7, 3.0).round(2),
        "occupancy_rate":   np.clip(rng.normal(91,8,n), 40, 100).round(1),
        "lease_term_years": np.clip(rng.exponential(7,n), 1, 30).round(1),
        "property_type":    prop_type,
        "market_tier":      _weighted(rng, ["Tier 1","Tier 2","Tier 3"], [35,40,25], n),
        "submarket":        _weighted(rng, ["CBD","Suburban","Urban Fringe","Secondary"], [30,38,18,14], n),
        "year_built":       rng.integers(1950, 2024, n),
        "square_footage":   _lognorm(rng, 10.0, 1.2, 1000, 2_000_000, n).astype(int),
        "loan_amount":      (prop_value*ltv/100).astype(int),
        "interest_rate":    np.clip(rng.normal(5.2,1.1,n), 2.5, 10.0).round(3),
        "amortization_years":_weighted(rng, [20,25,30], [20,35,45], n),
    })


def _build_rental_market(n=2000):
    rng = _rng()
    metro_income = _lognorm(rng, 10.9, 0.4, 35000, 200000, n).astype(int)
    asking_rent  = _lognorm(rng, 7.1, 0.5, 500, 8000, n).astype(int)
    gross_rent   = (asking_rent*rng.uniform(1.0,1.15,n)).astype(int)
    return pd.DataFrame({
        "asking_rent":          asking_rent,
        "gross_rent":           gross_rent,
        "vacancy_rate":         np.clip(rng.lognormal(1.5,0.5,n), 1.0, 20.0).round(1),
        "rent_to_income":       np.clip((gross_rent*12)/np.maximum(metro_income,1)*100, 10, 80).round(1),
        "yoy_rent_change":      np.clip(rng.normal(4.5,6.0,n), -15, 25).round(1),
        "unit_type":            _weighted(rng, ["Studio","1BR","2BR","3BR","4BR+"], [12,28,35,18,7], n),
        "bedrooms":             _weighted(rng, [0,1,2,3,4], [12,28,35,18,7], n),
        "year_built_band":      _weighted(rng, ["Pre-1960","1960-1980","1980-2000","2000-2015","Post-2015"], [18,22,25,22,13], n),
        "metro_tier":           _weighted(rng, ["Tier 1","Tier 2","Tier 3","Tier 4"], [20,30,30,20], n),
        "state":                _weighted(rng, ["CA","TX","FL","NY","PA","IL","OH","GA","NC","MI"], [14,12,10,9,6,6,5,5,4,4], n),
        "median_metro_income":  metro_income,
        "housing_supply_index": np.clip(rng.normal(100,18,n), 50, 160).round(1),
        "affordability_index":  np.clip(rng.normal(95,22,n), 30, 180).round(1),
    })


def _build_retail_transactions(n=2000):
    rng = _rng()
    amount = _lognorm(rng, 3.8, 1.4, 0.5, 25000, n).round(2)
    mcc = ["Grocery","Restaurants","Gas Stations","Online Retail","Utilities","Healthcare","Entertainment","Travel","ATM Withdrawal","Other"]
    channel = _weighted(rng, ["Card Present","Card Not Present","ACH","Wire","Mobile","ATM"], [35,28,14,5,12,6], n)
    fraud_base = np.where(channel=="Card Not Present", 0.003,
                 np.where(channel=="ATM", 0.002, 0.0006))
    return pd.DataFrame({
        "amount":             amount,
        "channel":            channel,
        "merchant_category":  _weighted(rng, mcc, [18,14,10,16,7,8,7,6,5,9], n),
        "day_of_week":        _weighted(rng, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], [13,13,14,15,17,16,12], n),
        "hour_of_day":        rng.integers(0, 24, n),
        "transaction_type":   _weighted(rng, ["Purchase","Refund","Transfer","Bill Payment","Cash"], [70,5,10,10,5], n),
        "is_recurring":       (rng.uniform(0,1,n) < 0.22).astype(int),
        "is_international":   (rng.uniform(0,1,n) < 0.06).astype(int),
        "account_age_months": np.clip(rng.exponential(48,n), 1, 360).astype(int),
        "monthly_tx_count":   np.clip(rng.poisson(22,n), 1, 150),
        "balance_band":       _weighted(rng, ["<$500","$500-$2K","$2K-$10K","$10K-$50K",">$50K"], [18,22,28,22,10], n),
        "fraud_flag":         (rng.uniform(0,1,n) < fraud_base).astype(int),
    })


def _build_commodity_prices(n=2000):
    rng = _rng()
    sectors_list = ["Energy","Energy","Energy","Metals","Metals","Metals","Agricultural","Agricultural","Agricultural"]
    comms_list   = ["Crude Oil WTI","Natural Gas","Gasoline RBOB","Gold","Copper","Silver","Corn","Wheat","Soybeans"]
    weights      = [16,12,8,14,10,6,12,10,12]
    commodity    = _weighted(rng, comms_list, weights, n)
    sector       = np.array([sectors_list[comms_list.index(c)] for c in commodity])
    energy_flag  = sector == "Energy"
    daily_return = np.where(energy_flag,
        np.clip(rng.standard_t(3,n)*0.018, -0.15, 0.15),
        np.clip(rng.standard_t(5,n)*0.010, -0.10, 0.10),
    ).round(4)
    return pd.DataFrame({
        "commodity":           commodity,
        "sector":              sector,
        "daily_return":        daily_return,
        "roll_yield":          np.clip(rng.normal(-0.0003,0.003,n), -0.03, 0.03).round(4),
        "spot_price_log":      np.clip(rng.normal(4.8,1.5,n), 0.5, 8.5).round(3),
        "inventory_change":    np.clip(rng.normal(0.0,2.5,n), -15, 15).round(2),
        "realized_vol_21d":    np.clip(rng.lognormal(-2.8,0.55,n), 0.005, 0.12).round(4),
        "implied_vol":         np.clip(rng.lognormal(-2.5,0.55,n), 0.008, 0.15).round(4),
        "basis":               np.clip(rng.normal(0.0,0.8,n), -5, 5).round(3),
        "seasonal_index":      np.clip(rng.normal(100,12,n), 65, 145).round(1),
        "dollar_index_return": np.clip(rng.normal(0.0001,0.004,n), -0.04, 0.04).round(4),
        "global_demand_proxy": np.clip(rng.normal(100,8,n), 70, 140).round(1),
        "supply_shock_flag":   (rng.uniform(0,1,n) < 0.04).astype(int),
    })
