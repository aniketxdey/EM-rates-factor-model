"""
factors.py -- six macro factors for cross-country EM 5y receiver RV.

Every factor is oriented so that POSITIVE = expect this market's 5y receiver
to OUTPERFORM the basket.  Signs are fixed by economics, never fitted.

Construction rules
  R1  Ratio scaling: inflation quantities are divided by
      max(effective_target, 2%) so a 2pp miss in Czechia and Turkey are not
      treated as equal.
  R2  Effective target = 0.5 * official + 0.5 * trailing-36m delivered
      inflation (delivered only, where no numeric target exists: MYR).
  R3  Cross-sectional standardisation at each cutoff, over markets whose data
      passes QA at that cutoff: robust clip at median +/- 3 * 1.4826 * MAD,
      then z-score.  Uses only same-date cross-section -> no look-ahead.
  R4  Missing input for a market -> factor = 0 (neutral) for that market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS = [
    "inflation_pressure",
    "real_carry",
    "terms_of_trade",
    "fiscal_thrust",
    "fx_passthrough",
    "credit_shortfall",
]

FACTOR_LABELS = {
    "inflation_pressure": "Inflation pressure",
    "real_carry": "Real yield & carry",
    "terms_of_trade": "Terms of trade",
    "fiscal_thrust": "Fiscal thrust",
    "fx_passthrough": "FX pass-through",
    "credit_shortfall": "Credit shortfall",
}

FACTOR_INPUTS = {
    "inflation_pressure": "BIS CPI y/y (lag 1m) vs effective target; 6m change in y/y",
    "real_carry": "5y yield (TradingView) minus CPI y/y; 5y minus BIS policy rate per bp of yield vol",
    "terms_of_trade": "IMF CTOT net commodity export price index (GDP-weighted), 6m log change, lag 4m",
    "fiscal_thrust": "IMF WEO general govt net lending %GDP: last outturn level + y/y change, lag to April",
    "fx_passthrough": "BIS real broad EER 6m log change x sqrt(WDI trade openness)",
    "credit_shortfall": "minus BIS credit-to-GDP gap (lag 6m); PE/PH/RO not covered -> neutral",
}

TARGET_FLOOR = 2.0
CLIP_MAD = 3.0


def xs_zscore(wide: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """R3: robust cross-sectional z-score per row, over masked-in markets."""
    x = wide.where(mask)
    med = x.median(axis=1)
    mad = (x.sub(med, axis=0)).abs().median(axis=1) * 1.4826
    lo = med - CLIP_MAD * mad
    hi = med + CLIP_MAD * mad
    x = x.clip(lower=lo, upper=hi, axis=0)
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def raw_factors(p: pd.DataFrame) -> pd.DataFrame:
    """Raw (un-normalised) factor inputs, same row order as p."""
    scale = np.maximum(p["eff_target"], TARGET_FLOOR)
    out = pd.DataFrame(index=p.index)

    excess = (p["cpi_yoy"] - p["eff_target"]) / scale
    momentum = (p["cpi_yoy"] - p["cpi_yoy_6m_ago"]) / scale
    out["inflation_pressure__level"] = -excess
    out["inflation_pressure__mom"] = -momentum

    out["real_carry__real"] = (p["yld_5y"] - p["cpi_yoy"]) / scale
    out["real_carry__carry"] = (p["yld_5y"] - p["policy"]) * 100.0 / p["yvol_bp"]

    out["terms_of_trade__chg"] = p["ctot_chg_6m"]

    out["fiscal_thrust__level"] = p["fiscal_bal"]
    out["fiscal_thrust__chg"] = p["fiscal_chg"]

    out["fx_passthrough__chg"] = p["reer_chg_6m"] * np.sqrt((p["openness"] / 100.0).clip(lower=0.1))

    out["credit_shortfall__gap"] = -p["credit_gap"]
    return out


def build_factors(p: pd.DataFrame) -> pd.DataFrame:
    """
    Adds FACTORS columns (cross-sectional z) and `parity` composite to p.
    Normalisation universe at each date = markets with data_ok.
    """
    p = p.copy()
    raw = raw_factors(p)
    mask = p.pivot(index="date", columns="ccy", values="data_ok").fillna(False).astype(bool)

    def _z(col):
        w = p.assign(_v=raw[col]).pivot(index="date", columns="ccy", values="_v")
        return xs_zscore(w, mask)

    for f in FACTORS:
        subs = [c for c in raw.columns if c.startswith(f + "__")]
        zs = [_z(c) for c in subs]
        comb = sum(z.fillna(0.0) for z in zs) / len(zs)
        # re-standardise the blended factor, keep neutral (0) where missing
        any_obs = sum(z.notna().astype(int) for z in zs) > 0
        comb = xs_zscore(comb.where(any_obs), mask & any_obs).fillna(0.0).where(mask)
        s = comb.stack(future_stack=True).rename(f)
        s.index.names = ["date", "ccy"]
        p = p.drop(columns=[f], errors="ignore").merge(s.reset_index(), on=["date", "ccy"], how="left")

    p["parity"] = combine(p, {f: 1.0 / len(FACTORS) for f in FACTORS})
    return p


def combine(p: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted factor sum, re-standardised cross-sectionally per date."""
    s = sum(w * p[f].fillna(0.0) for f, w in weights.items())
    s = s.where(p["data_ok"])
    g = s.groupby(p["date"])
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).clip(-3, 3)


def factor_correlation(fac: pd.DataFrame) -> pd.DataFrame:
    d = fac.loc[fac["data_ok"], FACTORS]
    c = d.corr()
    c.index = c.columns = [FACTOR_LABELS[f] for f in FACTORS]
    return c
