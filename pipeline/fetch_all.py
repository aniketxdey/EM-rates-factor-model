"""
Download every raw input into data/raw/ and log provenance to
data/raw/manifest.json.

    python -m pipeline.fetch_all            # everything
    python -m pipeline.fetch_all --only yields
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from config import CCYS, FRED_10Y_CHECK
from pipeline import sources as S
from pipeline.net import record


def fetch_macro():
    for name, fn in [("bis_cpi", S.bis_cpi), ("bis_policy_daily", S.bis_policy_daily),
                     ("bis_reer", S.bis_reer), ("bis_credit_gap", S.bis_credit_gap),
                     ("imf_ctot", S.imf_ctot), ("wdi_openness", S.wdi_openness)]:
        df, src = fn()
        record(name, df, src)
        print(f"  {name:18s} {len(df):7d} rows  {df['ccy'].nunique()} ccys")
    frames = []
    for flow in ("WEO", "WEO_2025_OCT_VINTAGE"):
        try:
            df, src = S.imf_weo_fiscal(flow)
            frames.append(df)
        except Exception as e:
            print(f"  WEO {flow} failed: {e}")
    weo = pd.concat(frames, ignore_index=True)
    record("imf_weo_fiscal", weo, "api.imf.org IMF.RES WEO GGXCNL_NGDP (latest + Oct-2025 vintage)")
    print(f"  imf_weo_fiscal     {len(weo):7d} rows")


def fetch_yields(tenors=("05Y", "10Y")):
    frames, failed = [], []
    for ccy in CCYS:
        for t in tenors:
            try:
                df, _ = S.tvc_yield(ccy, t)
                frames.append(df)
                print(f"  {ccy} {t}: {df['date'].min().date()} .. {df['date'].max().date()}  n={len(df)}")
            except Exception as e:
                failed.append(f"{ccy}{t}")
                print(f"  {ccy} {t}: FAILED ({e})")
            time.sleep(0.5)
    y = pd.concat(frames, ignore_index=True)
    record("tvc_yields", y, "TradingView TVC:{cc}05Y / {cc}10Y daily closes")
    fetch_fred_check()
    return failed


def fetch_fred_check():
    chk = []
    for ccy, sid in FRED_10Y_CHECK.items():
        try:
            d, _ = S.fred(sid)
            d["ccy"] = ccy
            chk.append(d)
        except Exception as e:
            print(f"  FRED {sid} failed (QA only, non-fatal): {e}")
    if chk:
        record("fred_10y_check", pd.concat(chk), "FRED OECD MEI IRLTLT01xxM156N")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["macro", "yields"])
    a = ap.parse_args(argv)
    failed = []
    if a.only in (None, "macro"):
        print("macro:")
        fetch_macro()
    if a.only in (None, "yields"):
        print("yields:")
        failed = fetch_yields()
    if failed:
        print("FAILED:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
