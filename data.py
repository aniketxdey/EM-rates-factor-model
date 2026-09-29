"""
data.py -- data layer for the EM rates relative-value model.

TWO SOURCES, KEPT STRICTLY SEPARATE:

1. SNAPSHOT (real).  Hand-transcribed, source-cited macro data for 17 EM
   currency areas as of Sep 2026.  Drives the live scorecard.  Every number
   is traceable to the citation in SOURCES below.

2. PANEL (synthetic, calibrated).  A monthly history 2015-01 .. 2026-08 used
   to exercise the backtest.  It is NOT real history.  It is generated so
   that its *statistical properties* (yield vol by country, inflation cycle
   timing, factor cross-correlations, signal-to-return information
   coefficient) match published estimates.  Backtest statistics computed on
   it measure whether the PIPELINE is correct, not whether the STRATEGY
   works.  See README section "What the backtest does and does not show".

To run on real history, replace load_panel() with a loader that returns the
same schema from JPMaQS / Bloomberg / Haver.  Nothing downstream changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# SOURCES for the real snapshot
# --------------------------------------------------------------------------
SOURCES = {
    "cpi_yoy": (
        "Trading Economics, 'Inflation Rate Forecast 2026/2027', "
        "latest print Aug/2026, retrieved 2026-09-23. "
        "https://tradingeconomics.com/forecast/inflation-rate"
    ),
    "cpi_fcast": (
        "Trading Economics consensus forecast path Q3/26-Q2/27, same page."
    ),
    "policy_rate": (
        "Wikipedia, 'List of countries by central bank interest rates', "
        "compiled from central bank decision pages, retrieved 2026-09-23. "
        "https://en.wikipedia.org/wiki/List_of_countries_by_central_bank_interest_rates"
    ),
    "inflation_target": (
        "Official central bank inflation targets (midpoint of band where a "
        "band is published). Stable published policy parameters."
    ),
    "structural": (
        "ESTIMATED. Trade openness, energy import dependence, commodity export "
        "basket, fiscal balance and foreign-ownership figures below are "
        "approximate analyst estimates, NOT transcribed from source. They are "
        "the weakest link in the snapshot -- replace with World Bank WDI, IMF "
        "WEO and national debt-office data before using the scorecard live."
    ),
}

# --------------------------------------------------------------------------
# REAL SNAPSHOT -- Sep 2026
# --------------------------------------------------------------------------
# cpi_yoy        : latest headline CPI %y/y (Aug 2026 print)          [REAL]
# cpi_path       : consensus %y/y for Q3/26,Q4/26,Q1/27,Q2/27         [REAL]
# policy_rate    : current policy rate %                              [REAL]
# target         : official inflation target midpoint %               [REAL]
# openness       : (exports+imports)/GDP, ratio                       [EST]
# energy_dep     : net energy imports as share of energy use, ratio   [EST]
#                  negative = net exporter
# comdty_beta    : sensitivity of terms of trade to broad commodity   [EST]
#                  index, ratio (positive = exporter)
# fiscal_bal     : general govt balance %GDP                          [EST]
# fiscal_chg     : y/y change in fiscal balance, pp (+ = tightening)  [EST]
# credit_gap     : private credit growth minus nominal trend, pp      [EST]
# foreign_own    : non-resident share of local govt debt, ratio       [EST]
# yld_vol        : annualised s.d. of 5y yield changes, bp            [EST]
SNAPSHOT = {
    #                cpi   cpi_path                 pol   tgt  open  enrg  cmdy  fisc  fchg  crd   fown  yvol
    "BRL": dict(cpi_yoy=4.22, cpi_path=[4.6, 4.9, 4.7, 4.4], policy_rate=13.75, target=3.0,
                openness=0.38, energy_dep=-0.10, comdty_beta=0.55, fiscal_bal=-7.8, fiscal_chg=-0.6,
                credit_gap=1.8, foreign_own=0.10, yld_vol=145.0, name="Brazil"),
    "CLP": dict(cpi_yoy=4.10, cpi_path=[4.5, 4.6, 4.2, 3.8], policy_rate=4.50, target=3.0,
                openness=0.66, energy_dep=0.62, comdty_beta=0.80, fiscal_bal=-2.4, fiscal_chg=0.3,
                credit_gap=-1.2, foreign_own=0.14, yld_vol=105.0, name="Chile"),
    "COP": dict(cpi_yoy=6.24, cpi_path=[6.7, 6.9, 6.2, 5.8], policy_rate=12.00, target=3.0,
                openness=0.42, energy_dep=-0.45, comdty_beta=0.70, fiscal_bal=-6.9, fiscal_chg=-1.1,
                credit_gap=0.4, foreign_own=0.23, yld_vol=150.0, name="Colombia"),
    "CZK": dict(cpi_yoy=1.90, cpi_path=[2.7, 2.5, 2.2, 2.3], policy_rate=3.75, target=2.0,
                openness=1.48, energy_dep=0.38, comdty_beta=-0.25, fiscal_bal=-2.1, fiscal_chg=0.4,
                credit_gap=-0.6, foreign_own=0.36, yld_vol=92.0, name="Czechia"),
    "HUF": dict(cpi_yoy=1.30, cpi_path=[1.5, 1.9, 2.1, 2.2], policy_rate=5.50, target=3.0,
                openness=1.64, energy_dep=0.60, comdty_beta=-0.30, fiscal_bal=-4.6, fiscal_chg=0.2,
                credit_gap=-1.9, foreign_own=0.31, yld_vol=138.0, name="Hungary"),
    "IDR": dict(cpi_yoy=3.19, cpi_path=[3.3, 2.8, 2.4, 2.6], policy_rate=5.75, target=2.5,
                openness=0.43, energy_dep=0.05, comdty_beta=0.45, fiscal_bal=-2.7, fiscal_chg=-0.3,
                credit_gap=1.1, foreign_own=0.14, yld_vol=98.0, name="Indonesia"),
    "INR": dict(cpi_yoy=4.82, cpi_path=[4.9, 5.6, 5.2, 4.1], policy_rate=5.25, target=4.0,
                openness=0.46, energy_dep=0.72, comdty_beta=-0.40, fiscal_bal=-7.4, fiscal_chg=0.4,
                credit_gap=2.3, foreign_own=0.04, yld_vol=62.0, name="India"),
    "KRW": dict(cpi_yoy=3.10, cpi_path=[3.3, 2.9, 2.6, 2.4], policy_rate=3.00, target=2.0,
                openness=0.88, energy_dep=0.82, comdty_beta=-0.50, fiscal_bal=-2.9, fiscal_chg=0.1,
                credit_gap=-0.8, foreign_own=0.21, yld_vol=68.0, name="South Korea"),
    "MXN": dict(cpi_yoy=3.26, cpi_path=[3.4, 3.5, 3.4, 3.3], policy_rate=6.50, target=3.0,
                openness=0.84, energy_dep=0.18, comdty_beta=0.25, fiscal_bal=-4.2, fiscal_chg=0.5,
                credit_gap=0.2, foreign_own=0.15, yld_vol=118.0, name="Mexico"),
    "MYR": dict(cpi_yoy=1.90, cpi_path=[2.1, 2.1, 2.0, 1.7], policy_rate=2.75, target=2.0,
                openness=1.33, energy_dep=-0.12, comdty_beta=0.40, fiscal_bal=-4.0, fiscal_chg=0.6,
                credit_gap=-0.3, foreign_own=0.22, yld_vol=58.0, name="Malaysia"),
    "PEN": dict(cpi_yoy=4.44, cpi_path=[4.7, 4.0, 3.6, 3.1], policy_rate=4.25, target=2.0,
                openness=0.52, energy_dep=0.22, comdty_beta=0.78, fiscal_bal=-3.1, fiscal_chg=0.2,
                credit_gap=-1.4, foreign_own=0.38, yld_vol=112.0, name="Peru"),
    "PHP": dict(cpi_yoy=6.10, cpi_path=[6.4, 5.9, 5.5, 2.0], policy_rate=5.00, target=3.0,
                openness=0.72, energy_dep=0.55, comdty_beta=-0.20, fiscal_bal=-5.6, fiscal_chg=-0.2,
                credit_gap=1.6, foreign_own=0.03, yld_vol=88.0, name="Philippines"),
    "PLN": dict(cpi_yoy=3.40, cpi_path=[3.6, 3.5, 2.5, 2.3], policy_rate=3.75, target=2.5,
                openness=1.12, energy_dep=0.45, comdty_beta=-0.28, fiscal_bal=-6.1, fiscal_chg=-0.8,
                credit_gap=-0.9, foreign_own=0.16, yld_vol=104.0, name="Poland"),
    "RON": dict(cpi_yoy=6.20, cpi_path=[5.5, 5.4, 5.0, 4.9], policy_rate=6.50, target=2.5,
                openness=0.86, energy_dep=0.28, comdty_beta=-0.15, fiscal_bal=-8.4, fiscal_chg=-1.4,
                credit_gap=0.9, foreign_own=0.19, yld_vol=126.0, name="Romania"),
    "THB": dict(cpi_yoy=2.53, cpi_path=[2.6, 2.5, 2.4, 1.9], policy_rate=1.00, target=2.0,
                openness=1.28, energy_dep=0.48, comdty_beta=-0.22, fiscal_bal=-3.5, fiscal_chg=0.3,
                credit_gap=-2.1, foreign_own=0.11, yld_vol=64.0, name="Thailand"),
    "TRY": dict(cpi_yoy=31.51, cpi_path=[31.3, 31.0, 30.0, 28.0], policy_rate=37.00, target=5.0,
                openness=0.72, energy_dep=0.75, comdty_beta=-0.55, fiscal_bal=-5.2, fiscal_chg=-0.4,
                credit_gap=4.2, foreign_own=0.08, yld_vol=420.0, name="Turkey"),
    "ZAR": dict(cpi_yoy=4.30, cpi_path=[5.4, 5.7, 4.9, 3.7], policy_rate=7.00, target=4.5,
                openness=0.68, energy_dep=-0.08, comdty_beta=0.62, fiscal_bal=-5.0, fiscal_chg=0.2,
                credit_gap=-1.1, foreign_own=0.25, yld_vol=132.0, name="South Africa"),
}

UNIVERSE = list(SNAPSHOT.keys())

# Markets are excluded from trading in periods where FX was pegged, capital
# controls bound, or the local market was effectively untradable.  Following
# the literature, blown-up markets are RETAINED in the panel and blacklisted
# for specific windows -- they are not dropped, which would create
# survivorship bias.
BLACKLIST = {
    # ccy: [(start, end), ...]  inclusive, monthly
    "TRY": [("2023-06", "2024-03")],   # post-election policy discontinuity
}


def snapshot_frame() -> pd.DataFrame:
    """Real macro snapshot as a tidy DataFrame, one row per currency."""
    rows = []
    for ccy, d in SNAPSHOT.items():
        r = {k: v for k, v in d.items() if k != "cpi_path"}
        r["ccy"] = ccy
        # consensus 4-quarter-ahead path -> disinflation slope (pp per year)
        path = d["cpi_path"]
        r["cpi_1q"] = path[0]
        r["cpi_4q"] = path[3]
        r["cpi_slope"] = path[3] - d["cpi_yoy"]
        rows.append(r)
    df = pd.DataFrame(rows).set_index("ccy")
    return df.loc[UNIVERSE]


# --------------------------------------------------------------------------
# CALIBRATED SYNTHETIC PANEL
# --------------------------------------------------------------------------
# Calibration targets, all from published estimates discussed in the README:
#   - monthly IC of composite signal vs forward return ~ 0.04
#   - cross-country 5y IRS return correlation 0.30-0.70
#   - annualised vol of vol-targeted position = 10% by construction
#   - inflation cycle: surge 2021H2-2022H2, disinflation 2023-2024,
#     partial re-acceleration 2026 (energy shock)
PANEL_START = "2015-01"
PANEL_END = "2026-08"
_TRUE_IC = 0.045          # embedded signal strength (deliberately weak)
_GLOBAL_LOAD = 0.55       # common factor loading on duration returns


def _inflation_cycle(n_months: int, rng: np.random.Generator) -> np.ndarray:
    """Global inflation cycle shape, in pp deviation from steady state."""
    t = np.arange(n_months)
    cyc = np.zeros(n_months)
    # COVID disinflation dip
    cyc += -1.2 * np.exp(-0.5 * ((t - 62) / 5.0) ** 2)
    # 2021-22 surge, peaking ~Sep 2022 (index 92)
    cyc += 5.4 * np.exp(-0.5 * ((t - 92) / 11.0) ** 2)
    # 2023-24 disinflation undershoot
    cyc += -0.9 * np.exp(-0.5 * ((t - 118) / 9.0) ** 2)
    # 2026 energy-driven re-acceleration
    cyc += 1.6 * np.exp(-0.5 * ((t - 136) / 7.0) ** 2)
    return cyc


def load_panel(seed: int = 20260923) -> pd.DataFrame:
    """
    Monthly panel, long format.

    Returns columns:
      date, ccy, cpi_yoy, cpi_3m3m, cpi_6m6m, cpi_surprise, target,
      yld_5y, real_yld, carry, reer_chg, tot_chg, fiscal_bal, fiscal_chg,
      credit_gap, foreign_own, ret_fwd_1m, tradable

    ret_fwd_1m is the return over [t, t+1] on a 10%-vol-targeted 5y fixed
    receiver position, in % of risk capital.  It is the target variable and
    is deliberately NOT observable at t.
    """
    rng = np.random.default_rng(seed)
    dates = pd.period_range(PANEL_START, PANEL_END, freq="M")
    n, k = len(dates), len(UNIVERSE)

    cycle = _inflation_cycle(n, rng)
    # common global duration factor
    global_ret = rng.normal(0, 1.0, n)

    frames = []
    for j, ccy in enumerate(UNIVERSE):
        s = SNAPSHOT[ccy]
        tgt = s["target"]
        beta_cycle = 0.6 + 1.1 * rng.random()          # country cycle beta
        if ccy == "TRY":
            beta_cycle = 4.5                           # idiosyncratic blowup

        # --- inflation process: anchor terminal value on the REAL Aug-2026 print
        idio = np.cumsum(rng.normal(0, 0.28, n)) * 0.45
        idio -= idio.mean()
        cpi = tgt + beta_cycle * cycle + idio
        if ccy == "TRY":
            cpi += 22.0 * np.clip((np.arange(n) - 70) / 40.0, 0, 1.4)
        # shift so the final month equals the verified print
        cpi += s["cpi_yoy"] - cpi[-1]
        cpi = np.maximum(cpi, -2.0)

        # shorter-horizon annualised trends lead the y/y measure
        cpi_3m3m = np.concatenate([cpi[:1], np.diff(cpi)]) * 5.0 + cpi
        cpi_6m6m = 0.5 * cpi + 0.5 * cpi_3m3m
        # ARMA-residual style surprise, scaled by country vol
        surprise = rng.normal(0, 0.30 + 0.02 * beta_cycle, n)

        # --- yields: policy-anchored, anchored on real current policy rate
        term_prem = 0.6 + 0.8 * rng.random()
        yld = cpi * 0.55 + tgt * 0.4 + term_prem + np.cumsum(
            rng.normal(0, s["yld_vol"] / 100.0 / np.sqrt(12) * 0.35, n))
        yld += (s["policy_rate"] + 0.4) - yld[-1]
        yld = np.maximum(yld, 0.15)
        real_yld = yld - cpi
        carry = (yld - (s["policy_rate"] * 0.0 + cpi * 0.15 + tgt * 0.6)) * 0.35

        # --- external / structural series
        reer = np.cumsum(rng.normal(0, 1.6, n))
        reer_chg = pd.Series(reer).diff(6).fillna(0).to_numpy()
        tot = np.cumsum(rng.normal(0, 1.0, n)) * s["comdty_beta"]
        tot_chg = pd.Series(tot).diff(6).fillna(0).to_numpy()

        fis = s["fiscal_bal"] + np.cumsum(rng.normal(0, 0.12, n))
        fis -= fis[-1] - s["fiscal_bal"]
        fis_chg = pd.Series(fis).diff(12).fillna(0).to_numpy()

        crd = s["credit_gap"] + np.cumsum(rng.normal(0, 0.22, n))
        crd -= crd[-1] - s["credit_gap"]

        fown = np.clip(s["foreign_own"] + np.cumsum(rng.normal(0, 0.004, n)), 0.01, 0.6)
        fown -= fown[-1] - s["foreign_own"]

        frames.append(pd.DataFrame({
            "date": dates, "ccy": ccy,
            "cpi_yoy": cpi, "cpi_3m3m": cpi_3m3m, "cpi_6m6m": cpi_6m6m,
            "cpi_surprise": surprise, "target": tgt,
            "yld_5y": yld, "real_yld": real_yld, "carry": carry,
            "reer_chg": reer_chg, "tot_chg": tot_chg,
            "fiscal_bal": fis, "fiscal_chg": fis_chg,
            "credit_gap": crd, "foreign_own": fown,
            "openness": s["openness"], "energy_dep": s["energy_dep"],
            "comdty_beta": s["comdty_beta"], "yld_vol": s["yld_vol"],
            "_gidx": np.arange(n), "_j": j,
        }))

    panel = pd.concat(frames, ignore_index=True)

    # --- forward returns -------------------------------------------------
    # Build a *latent* driver from the same economics the factors measure, so
    # the embedded IC is realistic rather than mechanical, then add a large
    # common factor and idiosyncratic noise.
    piv = panel.pivot(index="date", columns="ccy")

    def _xs_z(col):
        x = piv[col]
        return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)

    latent = (
        -0.9 * _xs_z("cpi_yoy").sub(piv["target"].mean(axis=1), axis=0).pipe(
            lambda d: d.div(d.std(axis=1).replace(0, np.nan), axis=0))
        + 0.8 * _xs_z("real_yld")
        + 0.4 * _xs_z("reer_chg")
        + 0.5 * _xs_z("tot_chg")
        + 0.3 * _xs_z("fiscal_bal")
        - 0.3 * _xs_z("credit_gap")
    )
    latent = latent.sub(latent.mean(axis=1), axis=0)
    latent = latent.div(latent.std(axis=1).replace(0, np.nan), axis=0).fillna(0)

    noise = pd.DataFrame(rng.normal(0, 1.0, (n, k)), index=piv.index, columns=UNIVERSE)
    gmat = pd.DataFrame(np.tile(global_ret[:, None], (1, k)),
                        index=piv.index, columns=UNIVERSE)

    # monthly vol of a 10%-annual-vol position
    mvol = 10.0 / np.sqrt(12)
    raw = _TRUE_IC * latent + _GLOBAL_LOAD * gmat + np.sqrt(
        max(1.0 - _GLOBAL_LOAD ** 2 - _TRUE_IC ** 2, 0.05)) * noise
    ret = raw / raw.stack().std() * mvol

    fwd = ret.shift(-1).stack(future_stack=True).rename("ret_fwd_1m").reset_index()
    panel = panel.merge(fwd, on=["date", "ccy"], how="left")

    # --- tradability blacklist -------------------------------------------
    panel["tradable"] = True
    for ccy, windows in BLACKLIST.items():
        for a, b in windows:
            m = (panel["ccy"] == ccy) & (panel["date"] >= pd.Period(a, "M")) \
                & (panel["date"] <= pd.Period(b, "M"))
            panel.loc[m, "tradable"] = False

    return panel.drop(columns=["_gidx", "_j"])


if __name__ == "__main__":
    snap = snapshot_frame()
    print("SNAPSHOT (real, Sep 2026)")
    print(snap[["name", "cpi_yoy", "policy_rate", "target", "cpi_slope"]].to_string())
    p = load_panel()
    print("\nPANEL", p.shape, p["date"].min(), "->", p["date"].max())
    print(p.groupby("ccy")["cpi_yoy"].last().round(2).to_string())
