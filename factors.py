"""
factors.py -- six conceptual factors for cross-country EM duration RV.

DESIGN RULES (each traceable to the literature audit):

R1  Ratio scaling.  Excess inflation in raw pp is not comparable across a
    2%-target and a 5%-target economy.  Every inflation quantity is divided
    by max(effective_target, 2.0) before normalisation.

R2  Effective, not official, targets.  The effective target is a blend of the
    stated target and trailing delivered inflation.  It measures risk to
    established credibility rather than distance from an aspiration the
    market stopped believing.

R3  Sequential (expanding-window) normalisation.  At every date t, means and
    standard deviations use ONLY the panel up to t.  Normalising on the full
    sample leaks the future through the scaling denominator -- a subtle and
    common backtest bug.

R4  Winsorise at +/- 3 s.d. after normalisation.

R5  Relative, not absolute.  Each factor is expressed versus the
    concurrently-tradable basket mean.  A directional book is a different
    strategy with different (much higher) benchmark correlation.

R6  Signs are imposed from theory, never fitted.  Every factor is oriented so
    that POSITIVE = expect the local 5y fixed-receiver position to
    OUTPERFORM the basket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS = [
    "inflation_pressure",
    "real_carry",
    "fx_valuation",
    "terms_of_trade",
    "fiscal_thrust",
    "credit_gap_shortfall",
]

FACTOR_LABELS = {
    "inflation_pressure": "Inflation pressure",
    "real_carry": "Real yield & carry",
    "fx_valuation": "FX valuation",
    "terms_of_trade": "Terms of trade",
    "fiscal_thrust": "Fiscal thrust",
    "credit_gap_shortfall": "Credit shortfall",
}

FACTOR_RATIONALE = {
    "inflation_pressure":
        "Excess CPI vs effective target blended with CPI surprises. High "
        "pressure implies asymmetric tightening risk and rising inflation "
        "risk premia, so it predicts UNDERperformance of duration.",
    "real_carry":
        "Ex-ante 5y real yield plus vol-targeted carry. High real yields "
        "indicate risk premium or implicit policy subsidy and predict "
        "OUTperformance.",
    "fx_valuation":
        "Openness-adjusted real effective appreciation. A stronger real "
        "exchange rate suppresses imported inflation and biases policy "
        "easier, predicting OUTperformance.",
    "terms_of_trade":
        "Commodity-basket-weighted terms-of-trade change. Improving terms of "
        "trade support the currency and the external balance, predicting "
        "OUTperformance.",
    "fiscal_thrust":
        "Fiscal balance level and 12m change. Austerity curbs demand and "
        "term premia, predicting OUTperformance. Tightest theoretical prior "
        "of the six in the post-2022 regime.",
    "credit_gap_shortfall":
        "Private credit growth below nominal trend. Weak credit calls for "
        "accommodation and lower real yields, predicting OUTperformance.",
}

MIN_OBS = 24          # months of panel history before a score is emitted
WINSOR = 3.0
TARGET_FLOOR = 2.0    # R1: the max(target, 2%) floor


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def effective_target(panel: pd.DataFrame, halflife: int = 48) -> pd.Series:
    """
    R2.  Effective target = 0.5 * official + 0.5 * EW-mean of delivered
    inflation, computed causally (expanding, no future data).
    """
    out = []
    for ccy, g in panel.groupby("ccy", sort=False):
        g = g.sort_values("date")
        delivered = g["cpi_yoy"].shift(1).ewm(halflife=halflife, min_periods=6).mean()
        eff = 0.5 * g["target"] + 0.5 * delivered.fillna(g["target"])
        out.append(pd.Series(eff.to_numpy(), index=g.index))
    return pd.concat(out).sort_index()


def sequential_zscore(panel: pd.DataFrame, col: str,
                      min_obs: int = MIN_OBS) -> pd.Series:
    """
    R3.  Expanding-window normalisation around a natural zero.

    At each date t the scaling s.d. uses the whole panel (all countries, all
    dates) observed UP TO AND INCLUDING t.  Deliberately does not de-mean by
    a full-sample mean -- the factors are built around a theoretical neutral
    level of zero.
    """
    wide = panel.pivot(index="date", columns="ccy", values=col).sort_index()
    # running count and running sum of squares across the whole panel
    flat = wide.to_numpy(dtype=float)
    valid = ~np.isnan(flat)
    cnt = np.cumsum(valid.sum(axis=1))
    ssq = np.cumsum(np.nansum(flat ** 2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(ssq / np.maximum(cnt - 1, 1))
    sd = np.where(cnt >= min_obs, sd, np.nan)
    sd = np.where(sd > 1e-9, sd, np.nan)
    z = wide.div(pd.Series(sd, index=wide.index), axis=0)
    z = z.clip(-WINSOR, WINSOR)                                   # R4
    return z.stack(future_stack=True).rename(col + "_z")


def relative(panel: pd.DataFrame, col: str) -> pd.Series:
    """R5.  Subtract the concurrently-tradable cross-sectional mean."""
    w = panel.pivot(index="date", columns="ccy", values=col)
    trad = panel.pivot(index="date", columns="ccy", values="tradable").fillna(False)
    masked = w.where(trad)
    return (masked.sub(masked.mean(axis=1), axis=0)
            .stack(future_stack=True).rename(col + "_rel"))


# --------------------------------------------------------------------------
# factor construction
# --------------------------------------------------------------------------
def build_factors(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy().sort_values(["ccy", "date"]).reset_index(drop=True)
    p["eff_target"] = effective_target(p)
    scale = np.maximum(p["eff_target"], TARGET_FLOOR)             # R1

    # --- 1. inflation pressure -------------------------------------------
    # average of three lookbacks (y/y, 6m/6m, 3m/3m) in excess of the
    # effective target, expressed as a RATIO of that target, then blended
    # with a decayed surprise trend.
    excess = ((p["cpi_yoy"] - p["eff_target"]) / scale
              + (p["cpi_6m6m"] - p["eff_target"]) / scale
              + (p["cpi_3m3m"] - p["eff_target"]) / scale) / 3.0
    p["_excess_infl"] = excess

    surp = []
    for ccy, g in p.groupby("ccy", sort=False):
        g = g.sort_values("date")
        sd = g["cpi_surprise"].shift(1).ewm(halflife=36, min_periods=6).std()
        z = (g["cpi_surprise"] / sd.replace(0, np.nan)).clip(-WINSOR, WINSOR)
        # decayed moving sum: surprises matter for weeks, not years
        surp.append(pd.Series(z.ewm(halflife=1.0, min_periods=1).mean().to_numpy(),
                              index=g.index))
    p["_surp_trend"] = pd.concat(surp).sort_index().fillna(0.0)

    zi = sequential_zscore(p, "_excess_infl").reset_index()
    p = p.merge(zi, on=["date", "ccy"], how="left")
    zs = sequential_zscore(p, "_surp_trend").reset_index()
    p = p.merge(zs, on=["date", "ccy"], how="left")

    # NEGATIVE sign: pressure predicts underperformance (R6)
    p["inflation_pressure"] = -(0.65 * p["_excess_infl_z"].fillna(0)
                                + 0.35 * p["_surp_trend_z"].fillna(0))

    # --- 2. real yield & carry -------------------------------------------
    p["_real_carry_raw"] = (p["real_yld"] / scale) * 0.6 + p["carry"] * 0.4
    zr = sequential_zscore(p, "_real_carry_raw").reset_index()
    p = p.merge(zr, on=["date", "ccy"], how="left")
    p["real_carry"] = p["_real_carry_raw_z"].fillna(0)

    # --- 3. FX valuation --------------------------------------------------
    # real appreciation scaled by trade openness: the same % REER move means
    # more for inflation in a very open economy.
    p["_fx_raw"] = p["reer_chg"] * np.sqrt(p["openness"].clip(lower=0.1))
    zf = sequential_zscore(p, "_fx_raw").reset_index()
    p = p.merge(zf, on=["date", "ccy"], how="left")
    p["fx_valuation"] = p["_fx_raw_z"].fillna(0)

    # --- 4. terms of trade ------------------------------------------------
    # ToT change, with an energy-dependence overlay: an energy importer is
    # hurt by an energy-led commodity rally even if it exports metals.
    p["_tot_raw"] = p["tot_chg"] - 0.35 * p["energy_dep"] * p["tot_chg"].abs()
    zt = sequential_zscore(p, "_tot_raw").reset_index()
    p = p.merge(zt, on=["date", "ccy"], how="left")
    p["terms_of_trade"] = p["_tot_raw_z"].fillna(0)

    # --- 5. fiscal thrust -------------------------------------------------
    # level of the balance plus its 12m change; positive = austere
    p["_fisc_raw"] = 0.6 * (p["fiscal_bal"] / 5.0) + 0.4 * p["fiscal_chg"]
    zfi = sequential_zscore(p, "_fisc_raw").reset_index()
    p = p.merge(zfi, on=["date", "ccy"], how="left")
    p["fiscal_thrust"] = p["_fisc_raw_z"].fillna(0)

    # --- 6. credit shortfall ---------------------------------------------
    p["_credit_raw"] = -p["credit_gap"]
    zc = sequential_zscore(p, "_credit_raw").reset_index()
    p = p.merge(zc, on=["date", "ccy"], how="left")
    p["credit_gap_shortfall"] = p["_credit_raw_z"].fillna(0)

    # --- relative versions + composite -----------------------------------
    for f in FACTORS:
        rel = relative(p, f).reset_index()
        p = p.merge(rel, on=["date", "ccy"], how="left")

    rel_cols = [f + "_rel" for f in FACTORS]
    # conceptual parity: equal weight, no fitting (robust to regime shifts)
    p["composite"] = p[rel_cols].mean(axis=1)
    # re-normalise the composite so one unit = one cross-sectional s.d.
    comp = p.pivot(index="date", columns="ccy", values="composite")
    comp_sd = comp.std(axis=1).replace(0, np.nan)
    comp = comp.div(comp_sd, axis=0).clip(-WINSOR, WINSOR)
    p = p.drop(columns=["composite"]).merge(
        comp.stack(future_stack=True).rename("composite").reset_index(),
        on=["date", "ccy"], how="left")

    p.loc[~p["tradable"], rel_cols + ["composite"]] = np.nan
    return p


def factor_correlation(fac: pd.DataFrame) -> pd.DataFrame:
    """Cross-correlation of the six relative factors (pooled)."""
    cols = [f + "_rel" for f in FACTORS]
    c = fac[cols].corr()
    c.index = [FACTOR_LABELS[f] for f in FACTORS]
    c.columns = [FACTOR_LABELS[f] for f in FACTORS]
    return c
