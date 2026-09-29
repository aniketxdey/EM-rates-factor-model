"""
backtest.py -- market-neutral, vol-targeted book of 5y receivers.

TARGET POSITIONS (all quantities known at cutoff t)
  1. a   = signal demeaned over tradable markets
  2. n0  = a / sigma_i        sigma_i = ex-ante annual vol of 1 unit notional
  3. cap = no market above MAX_RISK_SHARE of gross standalone risk
  4. beta hedge: remove the component along the equal-risk EM basket so the
     target has ZERO ex-ante beta to that basket (ex-ante covariance from
     trailing weekly yield changes, correlations shrunk toward zero)
  5. scale to TARGET_VOL ex-ante, subject to MAX_GROSS_LEVERAGE
EXECUTION
  held = prev + REBALANCE_SPEED * (target - prev); markets that stop being
  tradable are closed in full.  Steps 1-5 are linear in the signal on a given
  date and partial adjustment is linear in positions, so PnL decomposes
  exactly into factor contributions (plus a small clip residual).

COSTS (% of capital)
  trade : |delta notional| * duration * half bid-offer * stress
  roll  : |notional| * duration * full bid-offer * stress / 12
          (re-strike once a year to hold constant 5y maturity)
  stress = max(1, current yield vol / expanding median yield vol), following
  IMF GFSR (Oct-2025) evidence that EM bid-offer widens with volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (ANNUAL_ROLL_FRACTION, COV_SHRINK, COV_WINDOW_W, INSTRUMENTS,
                    MAX_GROSS_LEVERAGE, MAX_RISK_SHARE, MIN_MARKETS,
                    REBALANCE_SPEED, TARGET_VOL)
from factors import FACTORS

MONTHS = 12
RESIDUAL = "clip_residual"


# --------------------------------------------------------------------------
# ex-ante risk
# --------------------------------------------------------------------------
def ex_ante_cov(wk_bp: pd.DataFrame, cut: pd.Timestamp, ccys: list,
                sigma: pd.Series) -> np.ndarray:
    """Annual return covariance (%^2) per unit notional."""
    w = wk_bp.loc[:cut, ccys].tail(COV_WINDOW_W)
    corr = w.corr(min_periods=26).reindex(index=ccys, columns=ccys).fillna(0.0).to_numpy()
    corr = (1 - COV_SHRINK) * corr + COV_SHRINK * np.eye(len(ccys))
    np.fill_diagonal(corr, 1.0)
    s = sigma.reindex(ccys).to_numpy()
    return corr * np.outer(s, s)


def position_operator(sigma: np.ndarray, cov: np.ndarray, a_total: np.ndarray):
    """
    Returns (op, exante_vol): op maps a signal vector to target notionals via
    steps 1-5, with clip multipliers and scale fixed by the TOTAL signal so
    op is linear.
    """
    n = len(sigma)
    a = a_total - a_total.mean()
    risk = np.abs(a)
    share = risk / risk.sum() if risk.sum() > 0 else np.zeros(n)
    c = np.where(share > MAX_RISK_SHARE, MAX_RISK_SHARE / np.maximum(share, 1e-12), 1.0)
    b = (1.0 / n) / sigma
    cb = cov @ b
    denom = b @ cb

    def raw(x):
        x = x - x.mean()
        n1 = c * x / sigma
        return n1 - (n1 @ cb) / denom * b

    n2 = raw(a_total)
    vol = np.sqrt(max(n2 @ cov @ n2, 1e-12))
    k = TARGET_VOL / vol
    gross = np.abs(n2).sum()
    if gross * k > MAX_GROSS_LEVERAGE:
        k = MAX_GROSS_LEVERAGE / gross
    return (lambda x: k * raw(x)), k * vol


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------
def run_book(fac: pd.DataFrame, signal: pd.Series, wk_bp: pd.DataFrame,
             parts: dict[str, pd.Series] | None = None,
             cost_mult: float = 1.0, speed: float = REBALANCE_SPEED,
             live_last: bool = False) -> dict:
    """
    fac       : panel with factors (long)
    signal    : composite signal aligned to fac rows
    parts     : optional {factor: Series} summing to the pre-clip signal, for
                exact factor attribution
    live_last : on the final date, size positions for markets with clean
                data even though the entry print is not yet observable
                (this is the live trade the model wants today)
    """
    d = fac.assign(_sig=signal, _row=fac.index)
    last_date = d["date"].max()
    bo = pd.Series({c: v["bid_offer_bp"] for c, v in INSTRUMENTS.items()})
    names = (list(parts) + [RESIDUAL]) if parts is not None else []
    vol_hist: dict[str, list] = {}
    held = pd.Series(dtype=float)
    held_parts = {k: pd.Series(dtype=float) for k in names}
    prev_dur = pd.Series(dtype=float)
    recs, pos_rows, attr_rows = [], [], []

    for t, g in d.groupby("date", sort=True):
        g = g.set_index("ccy")
        for c, v in g["yvol_bp"].dropna().items():
            vol_hist.setdefault(c, []).append(v)
        is_live = live_last and t == last_date
        # A position whose exit cannot be marked (feed gap > 31 days, 3 cases
        # in the sample) is kept and credited zero return rather than being
        # dropped with hindsight.
        ok = (g["data_ok"] if is_live else g["tradable"] & g["ret_closed"]) & g["_sig"].notna()
        live = g[ok]
        ccys = list(live.index)

        target = pd.Series(0.0, index=ccys)
        target_parts = {k: pd.Series(0.0, index=ccys) for k in names}
        exante = np.nan
        if len(ccys) >= MIN_MARKETS:
            sigma = live["ret_vol"]
            cov = ex_ante_cov(wk_bp, live["cutoff"].iloc[0], ccys, sigma)
            op, exante = position_operator(sigma.to_numpy(), cov, live["_sig"].to_numpy())
            target = pd.Series(op(live["_sig"].to_numpy()), index=ccys)
            if parts is not None:
                acc = np.zeros(len(ccys))
                for k in parts:
                    v = op(parts[k].loc[live["_row"]].fillna(0.0).to_numpy())
                    target_parts[k] = pd.Series(v, index=ccys)
                    acc += v
                target_parts[RESIDUAL] = pd.Series(target.to_numpy() - acc, index=ccys)
        else:
            ccys = []

        def _adjust(prev, tgt):
            p = prev.reindex(ccys).fillna(0.0)
            return p + speed * (tgt.reindex(ccys).fillna(0.0) - p)

        new = _adjust(held, target)
        new_parts = {k: _adjust(held_parts[k], target_parts[k]) for k in names}

        ret = g["ret_fwd"].reindex(ccys).fillna(0.0)
        gross = float((new * ret).sum()) if ccys and not is_live else 0.0

        # ---- costs ------------------------------------------------------
        allc = new.index.union(held.index)
        dur = g["mod_dur"].reindex(allc).fillna(prev_dur.reindex(allc)).fillna(4.3)
        stress = pd.Series({c: max(1.0, g["yvol_bp"].get(c, np.nan) / np.median(vol_hist[c]))
                            if c in vol_hist and np.isfinite(g["yvol_bp"].get(c, np.nan)) else 1.0
                            for c in allc}, dtype=float)
        dn = new.reindex(allc).fillna(0.0) - held.reindex(allc).fillna(0.0)
        bo_c = bo.reindex(allc) * stress * cost_mult
        trade_cost = float((dn.abs() * dur * bo_c / 2.0).sum() / 100.0)
        roll_cost = float((new.abs() * dur.reindex(new.index) * bo_c.reindex(new.index)).sum()
                          / 100.0 * ANNUAL_ROLL_FRACTION / MONTHS) if len(new) else 0.0

        if parts is not None and ccys and not is_live:
            for k in names:
                attr_rows.append(dict(date=t, factor=k, pnl=float((new_parts[k] * ret).sum())))

        for c in ccys:
            pos_rows.append(dict(date=t, ccy=c, target=target[c], notional=new[c],
                                 dv01_yrs=new[c] * dur[c], trade=dn[c], ret=ret[c],
                                 pnl=np.nan if is_live else new[c] * ret[c],
                                 signal=live.loc[c, "_sig"], mod_dur=dur[c],
                                 yld_5y=live.loc[c, "yld_5y"], ret_vol=live.loc[c, "ret_vol"]))
        for c in held.index.difference(ccys):
            if abs(held[c]) > 0:
                pos_rows.append(dict(date=t, ccy=c, target=0.0, notional=0.0, dv01_yrs=0.0,
                                     trade=-held[c], ret=np.nan, pnl=0.0, signal=np.nan,
                                     mod_dur=dur[c], yld_5y=np.nan, ret_vol=np.nan))

        recs.append(dict(date=t, gross=gross, trade_cost=trade_cost, roll_cost=roll_cost,
                         net=gross - trade_cost - roll_cost, n_mkts=len(ccys),
                         gross_lev=float(new.abs().sum()), exante_vol=exante,
                         dv01_turnover=float((dn.abs() * dur).sum()), live=is_live))
        held, held_parts, prev_dur = new, new_parts, dur.reindex(new.index)

    res = pd.DataFrame(recs).set_index("date")
    return {
        "pnl": res,
        "positions": pd.DataFrame(pos_rows),
        "attribution": pd.DataFrame(attr_rows),
    }


def attribution_parts(fac: pd.DataFrame, weights: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Factor parts x_f = w_f(t) * z_f / sd_t, where sd_t is the cross-sectional
    s.d. used to standardise the composite, so sum_f x_f = composite (pre-clip).
    """
    w = weights.reindex(fac["date"]).reset_index(drop=True)
    w.index = fac.index
    raw = sum(w[f] * fac[f].fillna(0.0) for f in FACTORS).where(fac["data_ok"])
    sd = raw.groupby(fac["date"]).transform("std").replace(0, np.nan)
    return {f: (w[f] * fac[f].fillna(0.0) / sd).where(fac["data_ok"]) for f in FACTORS}


def weighted_signal(fac: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Composite from date-varying factor weights, standardised per date."""
    w = weights.reindex(fac["date"]).reset_index(drop=True)
    w.index = fac.index
    raw = sum(w[f] * fac[f].fillna(0.0) for f in FACTORS).where(fac["data_ok"])
    g = raw.groupby(fac["date"])
    return ((raw - g.transform("mean")) / g.transform("std").replace(0, np.nan)).clip(-3, 3)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def stats(pnl: pd.Series) -> dict:
    p = pnl.dropna()
    if len(p) < 12:
        return {}
    ann_ret = p.mean() * MONTHS
    ann_vol = p.std() * np.sqrt(MONTHS)
    cum = p.cumsum()
    sr = ann_ret / ann_vol if ann_vol > 1e-9 else np.nan
    return {
        "ann_return_pct": ann_ret,
        "ann_vol_pct": ann_vol,
        "sharpe": sr,
        "sharpe_t": sr * np.sqrt(len(p) / MONTHS),
        "sortino": ann_ret / (p[p < 0].std() * np.sqrt(MONTHS)),
        "max_drawdown_pct": (cum - cum.cummax()).min(),
        "hit_rate": (p > 0).mean(),
        "worst_month_pct": p.min(),
        "n_months": len(p),
        "start": str(p.index.min()),
        "end": str(p.index.max()),
    }


def em_basket(fac: pd.DataFrame) -> pd.Series:
    """Equal-weight basket of 10%-vol 5y receivers across tradable markets."""
    d = fac[fac["tradable"] & fac["ret_fwd"].notna()]
    r = d["ret_fwd"] * (TARGET_VOL / d["ret_vol"])
    return r.groupby(d["date"]).mean().rename("em_basket")


def beta_to_basket(pnl: pd.Series, basket: pd.Series) -> dict:
    x = pd.concat([pnl.rename("p"), basket.rename("b")], axis=1).dropna()
    if len(x) < 24:
        return {}
    X = np.column_stack([np.ones(len(x)), x["b"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, x["p"].to_numpy(), rcond=None)
    resid = x["p"].to_numpy() - X @ coef
    s2 = resid.var(ddof=2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return {"beta": coef[1], "beta_t": coef[1] / se, "alpha_ann_pct": coef[0] * MONTHS,
            "corr": float(x.corr().iloc[0, 1]), "n": len(x)}


def information_coefficient(fac: pd.DataFrame, col: str, dates=None) -> dict:
    """Monthly Spearman IC of `col` vs next-month vol-adjusted receiver return."""
    d = fac[fac["tradable"] & fac["ret_fwd"].notna() & fac[col].notna()]
    if dates is not None:
        d = d[d["date"].isin(dates)]
    y = d["ret_fwd"] / d["ret_vol"]
    ics = []
    for _, idx in d.groupby("date").groups.items():
        if len(idx) >= MIN_MARKETS and d.loc[idx, col].std() > 1e-12:
            ics.append(d.loc[idx, col].rank().corr(y.loc[idx].rank()))
    ics = pd.Series(ics).dropna()
    se = ics.std() / np.sqrt(len(ics)) if len(ics) > 1 else np.nan
    return {"mean_ic": ics.mean(), "ic_t": ics.mean() / se if se else np.nan,
            "ic_hit": (ics > 0).mean(), "n_months": len(ics)}
