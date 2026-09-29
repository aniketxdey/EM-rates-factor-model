"""
config.py -- universe, data-source identifiers, release lags, inflation
targets, instrument conventions and transaction costs.

Every number in this file is either a source identifier, a published policy
parameter, or a cost assumption with its reference.  Nothing here is fitted.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "output")

# --------------------------------------------------------------------------
# Universe: ccy -> identifiers in each source
#   bis  : BIS SDMX REF_AREA (2-letter)
#   iso3 : IMF / World Bank country code
#   tvc  : TradingView TVC government-bond prefix  (TVC:{tvc}05Y)
# --------------------------------------------------------------------------
UNIVERSE = {
    "BRL": dict(name="Brazil",       bis="BR", iso3="BRA", tvc="BR"),
    "CLP": dict(name="Chile",        bis="CL", iso3="CHL", tvc="CL"),
    "COP": dict(name="Colombia",     bis="CO", iso3="COL", tvc="CO"),
    "CZK": dict(name="Czechia",      bis="CZ", iso3="CZE", tvc="CZ"),
    "HUF": dict(name="Hungary",      bis="HU", iso3="HUN", tvc="HU"),
    "IDR": dict(name="Indonesia",    bis="ID", iso3="IDN", tvc="ID"),
    "INR": dict(name="India",        bis="IN", iso3="IND", tvc="IN"),
    "KRW": dict(name="South Korea",  bis="KR", iso3="KOR", tvc="KR"),
    "MXN": dict(name="Mexico",       bis="MX", iso3="MEX", tvc="MX"),
    "MYR": dict(name="Malaysia",     bis="MY", iso3="MYS", tvc="MY"),
    "PEN": dict(name="Peru",         bis="PE", iso3="PER", tvc="PE"),
    "PHP": dict(name="Philippines",  bis="PH", iso3="PHL", tvc="PH"),
    "PLN": dict(name="Poland",       bis="PL", iso3="POL", tvc="PL"),
    "RON": dict(name="Romania",      bis="RO", iso3="ROU", tvc="RO"),
    "THB": dict(name="Thailand",     bis="TH", iso3="THA", tvc="TH"),
    "TRY": dict(name="Turkey",       bis="TR", iso3="TUR", tvc="TR"),
    "ZAR": dict(name="South Africa", bis="ZA", iso3="ZAF", tvc="ZA"),
}
CCYS = list(UNIVERSE)

# OECD/FRED 10y monthly-average yields, used ONLY as a data-quality
# cross-check on the TradingView 10y series (never as a model input).
FRED_10Y_CHECK = {
    "CLP": "IRLTLT01CLM156N", "CZK": "IRLTLT01CZM156N", "HUF": "IRLTLT01HUM156N",
    "KRW": "IRLTLT01KRM156N", "MXN": "IRLTLT01MXM156N", "PLN": "IRLTLT01PLM156N",
    "ZAR": "IRLTLT01ZAM156N",
}

# IMF Commodity Terms of Trade: net export commodity price index, weighted by
# net commodity exports / GDP, rolling weights (Gruss & Kebhaj 2019).
CTOT_INDICATOR = "CEMPI_CTOTXM_GDP"
CTOT_WEIGHT = "R_RW_IX"

# --------------------------------------------------------------------------
# Point-in-time release lags.
# A series with reference period P is treated as KNOWN at month-end cutoff t
# only if  period_end(P) + lag <= t.  Lags are conservative upper bounds on
# the latest national/international publication across the 17 countries.
# --------------------------------------------------------------------------
RELEASE_LAG_MONTHS = {
    # CPI for month m: latest publisher is Malaysia/S.Africa (~3rd-4th week
    # of m+1).  So m is known by the end of m+1.
    "cpi": 1,
    # BIS broad REER (monthly average) published mid m+1.
    "reer": 1,
    # IMF CTOT monthly: observed 4-month publication lag (Sep-26 vintage ends May-26).
    "ctot": 4,
    # BIS credit-to-GDP gap, quarterly: ~2 quarters after quarter end.
    "credit_gap": 6,
    # General government net lending (WEO), annual: prior-year outturn first
    # published in the April WEO -> known from end-April of Y+1.
    "fiscal": 4,
    # World Bank WDI trade openness, annual: ~mid-year of Y+1; take 12m.
    "openness": 12,
}

# --------------------------------------------------------------------------
# Official inflation targets (midpoint of band), by effective year.
# Sources: central bank monetary-policy framework pages
#   BCB (Resoluções CMN), BCCh, BanRep, CNB, MNB, BI, RBI (FIT Aug-2016),
#   BoK, Banxico, BCRP, BSP, NBP, BNR, BoT, CBRT, SARB/National Treasury
#   (3% point target announced 12-Nov-2025).
# Malaysia (BNM) has no numeric target -> None: effective target falls back
# to delivered inflation only.
# --------------------------------------------------------------------------
INFLATION_TARGETS = {
    "BRL": {2005: 4.5, 2019: 4.25, 2020: 4.0, 2021: 3.75, 2022: 3.5, 2023: 3.25, 2024: 3.0},
    "CLP": {2005: 3.0},
    "COP": {2005: 5.0, 2006: 4.5, 2007: 4.0, 2009: 5.0, 2010: 3.0},
    "CZK": {2005: 3.0, 2010: 2.0},
    "HUF": {2005: 3.5, 2007: 3.0},
    "IDR": {2005: 6.0, 2008: 5.0, 2013: 4.5, 2015: 4.0, 2018: 3.5, 2020: 3.0, 2024: 2.5},
    "INR": {2005: 4.0},
    "KRW": {2005: 3.0, 2016: 2.0},
    "MXN": {2005: 3.0},
    "MYR": {2005: None},
    "PEN": {2005: 2.0},
    "PHP": {2005: 5.0, 2010: 4.5, 2011: 4.0, 2015: 3.0},
    "PLN": {2005: 2.5},
    "RON": {2005: 7.5, 2007: 4.0, 2008: 3.8, 2009: 3.5, 2011: 3.0, 2013: 2.5},
    "THB": {2005: 1.75, 2015: 2.5, 2020: 2.0},
    "TRY": {2005: 8.0, 2006: 5.0, 2009: 7.5, 2010: 6.5, 2011: 5.5, 2012: 5.0},
    "ZAR": {2005: 4.5, 2026: 3.0},
}

# --------------------------------------------------------------------------
# Tradable instrument per market (what the live book actually trades) and
# typical FULL bid-offer in bp of yield for a 5y clip in normal conditions.
#
# References for the cost levels:
#  [ADB] AsianBondsOnline, Asia Bond Monitor Mar-2026, Fig.7 (2025 average
#        on-the-run LCY govt bond bid-ask, Bloomberg data): KR ~1, TH ~2-3,
#        MY ~2-3, ID ~7, PH ~9 bp.
#  [BIS] BIS Papers No 67 (2012) Table A5, typical bid-ask on most liquid
#        issue, central-bank survey.
#  [GFSR] IMF GFSR Oct-2025 ch.3: bid-ask spreads widen ~0.7-0.9bp per 10pt
#        VIX -> costs are scaled up with realised vol in backtest.py.
#  Dealer convention for G-EM IRS runs (PLN/CZK/HUF/ZAR/MXN/KRW/INR/THB).
# All values are deliberately set at or above the top of the published range.
# Replace with your own executable quotes -- see HANDOFF.md.
# --------------------------------------------------------------------------
INSTRUMENTS = {
    "BRL": dict(instr="DI1 futures (B3), ~5y contract", bid_offer_bp=2.0),
    "CLP": dict(instr="CLP Camara (ICP) OIS/NDS 5y", bid_offer_bp=4.0),
    "COP": dict(instr="COP IBR OIS/NDS 5y", bid_offer_bp=6.0),
    "CZK": dict(instr="CZK IRS 5y vs 6M PRIBOR", bid_offer_bp=2.0),
    "HUF": dict(instr="HUF IRS 5y vs 6M BUBOR", bid_offer_bp=4.0),
    "IDR": dict(instr="INDOGB 5y benchmark (FX-hedged) / IDR NDS", bid_offer_bp=8.0),
    "INR": dict(instr="INR MIBOR OIS 5y (NDOIS offshore)", bid_offer_bp=1.5),
    "KRW": dict(instr="KRW IRS 5y vs 3M CD", bid_offer_bp=1.5),
    "MXN": dict(instr="MXN TIIE IRS 5y (65x1)", bid_offer_bp=1.5),
    "MYR": dict(instr="MYR IRS 5y vs 3M KLIBOR (NDIRS)", bid_offer_bp=3.0),
    "PEN": dict(instr="Peru Soberano 5y bond (FX-hedged)", bid_offer_bp=10.0),
    "PHP": dict(instr="RPGB 5y / PHP NDIRS", bid_offer_bp=10.0),
    "PLN": dict(instr="PLN IRS 5y vs 6M WIBOR", bid_offer_bp=1.5),
    "RON": dict(instr="ROMGB 5y bond (FX-hedged)", bid_offer_bp=10.0),
    "THB": dict(instr="THB THOR OIS 5y", bid_offer_bp=3.0),
    "TRY": dict(instr="TURKGB 5y / TRY IRS", bid_offer_bp=30.0),
    "ZAR": dict(instr="ZAR IRS 5y vs 3M JIBAR", bid_offer_bp=2.0),
}

# Constant-maturity 5y exposure is maintained by re-striking the full
# position once a year (unwind + new trade = one full bid-offer per year).
ANNUAL_ROLL_FRACTION = 1.0

# --------------------------------------------------------------------------
# Portfolio construction
# --------------------------------------------------------------------------
TARGET_VOL = 10.0          # % annualised, ex-ante, on risk capital
MAX_GROSS_LEVERAGE = 20.0  # sum |notional| / capital (swap notional)
MAX_RISK_SHARE = 0.25      # no market > 25% of gross risk
MIN_MARKETS = 6            # no book if fewer tradable markets
STALE_DAYS = 10            # yield older than this at cutoff -> untradable
QA_MIN_CORR_5_10 = 0.5     # trailing 52w corr of weekly 5y vs 10y changes
QA_MAX_STALE = 0.25        # max share of unchanged daily prints (last 250)
VOL_HALFLIFE_D = 63        # ex-ante leg vol (daily yield changes)
COV_WINDOW_W = 104         # weeks of history for ex-ante covariance
COV_SHRINK = 0.3           # shrink correlations toward zero
MIN_TRAIN_MONTHS = 36      # before ridge weights are used
# Partial adjustment toward target each month (Garleanu-Pedersen style cost
# control).  Fixed a priori at 0.5, NOT tuned; run.py reports 1.0 and 0.33.
REBALANCE_SPEED = 0.5
