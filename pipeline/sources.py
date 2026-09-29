"""
Fetchers for every real data source used by the model.  Each returns a tidy
long DataFrame and never touches disk; fetch_all.py handles caching.

Source                        | Endpoint (no API key required)
------------------------------|-----------------------------------------------
BIS consumer prices           | stats.bis.org/api/v2  WS_LONG_CPI
BIS central-bank policy rates | stats.bis.org/api/v2  WS_CBPOL (daily)
BIS effective exchange rates  | stats.bis.org/api/v2  WS_EER (real, broad)
BIS credit-to-GDP gaps        | stats.bis.org/api/v2  WS_CREDIT_GAP
IMF commodity terms of trade  | api.imf.org SDMX 2.1  IMF.RES,CTOT
IMF World Economic Outlook    | api.imf.org SDMX 2.1  IMF.RES,WEO
World Bank WDI                | api.worldbank.org/v2  NE.TRD.GNFS.ZS
TradingView TVC bond yields   | wss://data.tradingview.com (unauthenticated)
FRED (OECD MEI 10y)           | fred.stlouisfed.org/graph/fredgraph.csv
"""

from __future__ import annotations

import io

import pandas as pd

from config import CTOT_INDICATOR, CTOT_WEIGHT, UNIVERSE
from pipeline import tradingview
from pipeline.net import get

BIS = "https://stats.bis.org/api/v2/data/dataflow/BIS"
IMF = "https://api.imf.org/external/sdmx/2.1/data"
IMF_CSV = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}

BIS_CODES = "+".join(v["bis"] for v in UNIVERSE.values())
ISO3_CODES = "+".join(v["iso3"] for v in UNIVERSE.values())
BIS_TO_CCY = {v["bis"]: k for k, v in UNIVERSE.items()}
ISO3_TO_CCY = {v["iso3"]: k for k, v in UNIVERSE.items()}


def _bis(flow: str, key: str, start: str, area_col: str = "REF_AREA") -> pd.DataFrame:
    url = f"{BIS}/{flow}/1.0/{key}?format=csv&startPeriod={start}"
    d = pd.read_csv(io.StringIO(get(url).text))
    d = d[[area_col, "TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={area_col: "area", "TIME_PERIOD": "period", "OBS_VALUE": "value"})
    d["ccy"] = d["area"].map(BIS_TO_CCY)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna(subset=["ccy", "value"])[["ccy", "period", "value"]], url


def bis_cpi(start="2000-01"):
    """CPI index level (2010=100 or national base), monthly, NSA."""
    return _bis("WS_LONG_CPI", f"M.{BIS_CODES}.628", start)


def bis_policy_daily(start="2004-01-01"):
    return _bis("WS_CBPOL", f"D.{BIS_CODES}", start)


def bis_reer(start="2000-01"):
    """Real broad effective exchange rate, monthly average (2020=100)."""
    return _bis("WS_EER", f"M.R.B.{BIS_CODES}", start)


def bis_credit_gap(start="2000-Q1"):
    """Credit-to-GDP gap (actual minus HP trend), private non-fin sector, pp."""
    return _bis("WS_CREDIT_GAP", f"Q.{BIS_CODES}.P.A.C", start, area_col="BORROWERS_CTY")


def imf_ctot(start="2000-01"):
    key = f"{ISO3_CODES}.{CTOT_INDICATOR}.{CTOT_WEIGHT}.M"
    url = f"{IMF}/IMF.RES,CTOT/{key}?startPeriod={start}"
    d = pd.read_csv(io.StringIO(get(url, headers=IMF_CSV).text), low_memory=False)
    d = d[["COUNTRY", "TIME_PERIOD", "OBS_VALUE"]]
    d["ccy"] = d["COUNTRY"].map(ISO3_TO_CCY)
    d["period"] = d["TIME_PERIOD"].str.replace("-M", "-", regex=False)
    d["value"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    return d.dropna(subset=["ccy", "value"])[["ccy", "period", "value"]], url


def imf_weo_fiscal(vintage_flow: str = "WEO", start="2000"):
    """General government net lending/borrowing, % of GDP (annual)."""
    url = f"{IMF}/IMF.RES,{vintage_flow}/{ISO3_CODES}.GGXCNL_NGDP.A?startPeriod={start}"
    d = pd.read_csv(io.StringIO(get(url, headers=IMF_CSV).text), low_memory=False)
    d = d[["COUNTRY", "TIME_PERIOD", "OBS_VALUE"]]
    d["ccy"] = d["COUNTRY"].map(ISO3_TO_CCY)
    d["period"] = d["TIME_PERIOD"].astype(int)
    d["value"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    d["vintage"] = vintage_flow
    return d.dropna(subset=["ccy", "value"])[["ccy", "period", "value", "vintage"]], url


def wdi_openness(start=2000, end=2030):
    iso = ";".join(v["iso3"] for v in UNIVERSE.values())
    url = (f"https://api.worldbank.org/v2/country/{iso}/indicator/NE.TRD.GNFS.ZS"
           f"?format=json&date={start}:{end}&per_page=2000")
    js = get(url).json()
    rows = [(r["countryiso3code"], int(r["date"]), r["value"]) for r in js[1] if r["value"] is not None]
    d = pd.DataFrame(rows, columns=["iso3", "period", "value"])
    d["ccy"] = d["iso3"].map(ISO3_TO_CCY)
    return d.dropna(subset=["ccy"])[["ccy", "period", "value"]], url


def tvc_yield(ccy: str, tenor: str = "05Y"):
    sym = f"TVC:{UNIVERSE[ccy]['tvc']}{tenor}"
    d = tradingview.daily_bars_retry(sym)
    d["ccy"] = ccy
    d["tenor"] = tenor
    d = d.rename(columns={"close": "value"})
    return d[["ccy", "tenor", "date", "value"]], f"tradingview {sym}"


def fred(series: str):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    # FRED stalls requests carrying a browser User-Agent; identify honestly.
    d = pd.read_csv(io.StringIO(get(url, headers={"User-Agent": "rates-factor-model/1.0"}, timeout=30).text))
    d.columns = ["date", "value"]
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna(), url
