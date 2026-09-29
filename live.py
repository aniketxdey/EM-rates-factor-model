"""
live.py -- today's target book and trade list.

    python live.py --capital 10000000                 # uses cached data
    python live.py --capital 10000000 --refresh       # re-download first
    python live.py --capital 10000000 --holdings positions/current.csv

Holdings file (optional) columns:  ccy, dv01_usd
    dv01_usd = USD PnL per 1bp FALL in the 5y rate (positive = receiving).
Without a holdings file the book is assumed flat.

Run on the last business day of each month, after local closes; execute at
the next session, matching the backtest convention.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import backtest as B
import factors as F
import learn as L
import panel as P
from config import (INSTRUMENTS, MAX_GROSS_LEVERAGE, MAX_RISK_SHARE, MIN_MARKETS,
                    OUT_DIR, REBALANCE_SPEED, ROOT, TARGET_VOL, UNIVERSE)

LIVE_DIR = os.path.join(OUT_DIR, "live")
MAX_PRINT_AGE_DAYS = 4


def load_holdings(path: str | None) -> pd.Series:
    if not path:
        return pd.Series(dtype=float)
    h = pd.read_csv(path)
    return h.groupby("ccy")["dv01_usd"].sum()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, required=True, help="risk capital, USD")
    ap.add_argument("--holdings", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="trade all the way to target (default: partial adjustment as backtested)")
    ap.add_argument("--month-end-only", action="store_true",
                    help="exit quietly unless today is the last business day of the month")
    a = ap.parse_args(argv)

    today_d = pd.Timestamp.today().normalize()
    if a.month_end_only and today_d != today_d + pd.offsets.BMonthEnd(0):
        print(f"{today_d.date()} is not the last business day of the month; nothing to do.")
        return 0

    if a.refresh:
        from pipeline import fetch_all
        if fetch_all.main([]) != 0:
            print("WARNING: some series failed to refresh; check output above")

    panel = P.build_panel()
    fac = F.build_factors(panel)
    wk = P.weekly_returns_matrix()
    weights = L.sequential_weights(fac)
    sig = B.weighted_signal(fac, weights)
    parts = B.attribution_parts(fac, weights)
    book = B.run_book(fac, sig, wk, parts=parts, live_last=True)

    last = fac["date"].max()
    today = fac.loc[fac["date"] == last].set_index("ccy")
    run_date = today_d
    pos = book["positions"]
    pos = pos[pos["date"] == last].set_index("ccy")
    pnl_row = book["pnl"].loc[last]

    # ---- pre-trade checks ----------------------------------------------
    checks = []
    y5 = P.load_yields()["05Y"]
    last_print = y5.apply(lambda s: s.last_valid_index())
    age = (run_date - pd.to_datetime(last_print)).dt.days
    stale = [c for c in pos.index if age.get(c, 99) > MAX_PRINT_AGE_DAYS]
    checks.append(("markets in book >= MIN_MARKETS", len(pos) >= MIN_MARKETS, f"{len(pos)}"))
    checks.append(("5y prints no older than 4 days", not stale, ",".join(stale) or "ok"))
    checks.append(("ex-ante vol at target", abs(pnl_row["exante_vol"] - TARGET_VOL) < 0.5
                   or pnl_row["gross_lev"] >= MAX_GROSS_LEVERAGE - 1e-6, f"{pnl_row['exante_vol']:.2f}%"))
    checks.append(("gross notional within cap", pnl_row["gross_lev"] <= MAX_GROSS_LEVERAGE + 1e-6,
                   f"{pos['target'].abs().sum():.2f}x"))
    risk = (pos["target"].abs() * pos["ret_vol"])
    checks.append(("max single-market risk share", (risk / risk.sum()).max() <= MAX_RISK_SHARE + 0.02,
                   f"{(risk / risk.sum()).max():.2%}"))
    w_now = weights.loc[last]
    checks.append(("factor weights from ridge (not warm-up parity)", w_now["source"] == "ridge",
                   w_now["source"]))

    # ---- sizing ---------------------------------------------------------
    cap = a.capital
    cur = load_holdings(a.holdings)
    ccys = sorted(set(pos.index) | set(cur.index))
    rows = []
    for c in ccys:
        if c in pos.index:
            n_t = pos.loc[c, "target"]
            dur = pos.loc[c, "mod_dur"]
        else:
            n_t, dur = 0.0, float(today.loc[c, "mod_dur"]) if c in today.index else 4.3
        tgt_dv01 = n_t * dur * cap * 1e-4
        cur_dv01 = float(cur.get(c, 0.0))
        speed = 1.0 if a.full else REBALANCE_SPEED
        new_dv01 = cur_dv01 + speed * (tgt_dv01 - cur_dv01) if c in pos.index else 0.0
        trade = new_dv01 - cur_dv01
        bo = INSTRUMENTS[c]["bid_offer_bp"]
        contrib = {F.FACTOR_LABELS[f]: float(w_now[f] * today.loc[c, f]) if c in today.index
                   and np.isfinite(today.loc[c, f]) else np.nan for f in F.FACTORS}
        rows.append({
            "ccy": c, "country": UNIVERSE[c]["name"], "instrument": INSTRUMENTS[c]["instr"],
            "signal_z": pos.loc[c, "signal"] if c in pos.index else np.nan,
            "yld_5y": today.loc[c, "yld_5y"] if c in today.index else np.nan,
            "last_print": pd.to_datetime(last_print.get(c)).date() if pd.notna(last_print.get(c)) else None,
            "target_dv01_usd": tgt_dv01,
            "current_dv01_usd": cur_dv01,
            "new_dv01_usd": new_dv01,
            "TRADE_dv01_usd": trade,
            "ACTION": ("RECEIVE" if trade > 0 else "PAY") if abs(trade) >= 1 else "-",
            "approx_notional_usd": abs(trade) / (dur * 1e-4) if dur else np.nan,
            "est_cost_usd": abs(trade) * bo / 2.0,
            **{f"z_{k}": v for k, v in contrib.items()},
        })
    t = pd.DataFrame(rows).sort_values("target_dv01_usd", ascending=False)

    # ---- output ---------------------------------------------------------
    os.makedirs(LIVE_DIR, exist_ok=True)
    tag = run_date.strftime("%Y-%m-%d")
    t.round(2).to_csv(os.path.join(LIVE_DIR, f"trades_{tag}.csv"), index=False)
    pd.DataFrame({"ccy": t["ccy"], "dv01_usd": t["new_dv01_usd"].round(2)}).to_csv(
        os.path.join(LIVE_DIR, f"holdings_after_{tag}.csv"), index=False)

    pd.set_option("display.width", 220)
    cutoff = today["cutoff"].max().date()
    print("=" * 96)
    print(f"LIVE BOOK  run {tag}  data cutoff {cutoff}  capital ${cap:,.0f}  "
          f"target vol {TARGET_VOL:.0f}%  speed {'1.0 (full)' if a.full else REBALANCE_SPEED}")
    print("=" * 96)
    print("\nFactor weights in use:",
          ", ".join(f"{F.FACTOR_LABELS[f]} {w_now[f]:.2f}" for f in F.FACTORS))
    print("\nPRE-TRADE CHECKS")
    ok_all = True
    for name, ok, info in checks:
        ok_all &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")
    cols = ["ccy", "instrument", "signal_z", "yld_5y", "last_print", "target_dv01_usd",
            "current_dv01_usd", "TRADE_dv01_usd", "ACTION", "approx_notional_usd", "est_cost_usd"]
    print("\nTRADE LIST  (DV01 = USD per 1bp fall in the 5y rate; + = receive fixed)\n")
    show = t[cols].copy()
    for c in ["target_dv01_usd", "current_dv01_usd", "TRADE_dv01_usd", "approx_notional_usd", "est_cost_usd"]:
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    print(show.round({"signal_z": 2, "yld_5y": 2}).to_string(index=False))
    print(f"\n  net DV01 after trades: {t['new_dv01_usd'].sum():,.0f} USD/bp "
          f"(beta-hedged, not DV01-neutral: legs are risk-weighted)")
    print(f"  estimated execution cost: ${t['est_cost_usd'].sum():,.0f}")
    excluded = sorted(set(UNIVERSE) - set(pos.index))
    if excluded:
        why = today.reindex(excluded)[["yld_age", "qa_corr510", "qa_stale"]].round(2)
        print("\n  excluded by data rules (stale print / QA):")
        print("  " + why.to_string().replace("\n", "\n  "))
    print(f"\nWrote {os.path.relpath(LIVE_DIR, ROOT)}/trades_{tag}.csv and holdings_after_{tag}.csv")
    if not ok_all:
        print("\n*** One or more pre-trade checks FAILED -- do not execute until resolved ***")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
