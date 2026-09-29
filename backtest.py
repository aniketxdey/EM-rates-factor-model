"""
backtest.py -- vol-targeted cross-country relative-value book and diagnostics.

POSITION CONVENTION
  One unit of signal = one unit of 10%-annualised-vol risk in the local 5y
  fixed receiver, versus an equally weighted basket of the same risk across
  all concurrently tradable markets.  Because every leg is already
  vol-targeted, position sizes are directly comparable between Malaysia and
  Turkey without a separate DV01 overlay.

TIMING
  Signals are taken at month end t and held over [t, t+1].  Returns are
  strictly forward.  One day of slippage is assumed and charged through the
  cost model rather than by shifting the return series.

WHAT IS DELIBERATELY NOT DONE
  No optimisation of factor weights, no factor selection, no parameter
  search on the evaluation sample.  Conceptual parity only.  The regularised
  learner in learn.py is reported separately and is sequential.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors import FACTORS, FACTOR_LABELS

ANNUAL_VOL_TARGET = 10.0    # % of risk capital
MONTHS = 12


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------
def signal_to_positions(sig_wide: pd.DataFrame,
                        tradable: pd.DataFrame,
                        max_pos: float = 3.0) -> pd.DataFrame:
    """
    Convert a cross-sectional signal into a market-neutral relative book.

    Each row is demeaned over tradable markets and scaled so gross risk per
    period is constant, so PnL is not mechanically larger when the signal
    happens to be dispersed.
    """
    s = sig_wide.where(tradable).clip(-max_pos, max_pos)
    s = s.sub(s.mean(axis=1), axis=0)                       # market neutral
    gross = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(gross, axis=0).fillna(0.0)


def threshold_positions(sig_wide: pd.DataFrame,
                        tradable: pd.DataFrame,
                        entry: float = 1.0,
                        exit_: float = 0.15) -> pd.DataFrame:
    """
    Transaction-cost-friendly variant: unit long/short with hysteresis.
    Enter at |signal| > entry, hold until |signal| < exit_.  This is the
    single most effective turnover control in the literature.
    """
    s = sig_wide.where(tradable)
    pos = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    state = pd.Series(0.0, index=s.columns)
    for t in s.index:
        row = s.loc[t]
        for c in s.columns:
            v = row[c]
            if not np.isfinite(v):
                state[c] = 0.0
                continue
            if state[c] == 0.0:
                if v > entry:
                    state[c] = 1.0
                elif v < -entry:
                    state[c] = -1.0
            else:
                if abs(v) < exit_ or np.sign(v) != state[c]:
                    state[c] = 0.0
        pos.loc[t] = state.to_numpy()
    pos = pos.where(tradable, 0.0)
    pos = pos.sub(pos.mean(axis=1), axis=0)
    gross = pos.abs().sum(axis=1).replace(0, np.nan)
    return pos.div(gross, axis=0).fillna(0.0)


# --------------------------------------------------------------------------
# pnl
# --------------------------------------------------------------------------
def run_book(pos: pd.DataFrame, ret_wide: pd.DataFrame,
             cost_bp_per_unit_turnover: float = 4.0,
             scale_to_target: bool = True) -> dict:
    """
    Compute gross and net PnL.

    cost_bp_per_unit_turnover: round-trip cost in bp of risk capital for a
    full 100% gross turnover.  EM local rates bid-offer varies widely and is
    poorly documented publicly; 4bp is a deliberately conservative placeholder.
    """
    pos = pos.reindex(columns=ret_wide.columns).fillna(0.0)
    aligned = ret_wide.reindex(pos.index)
    gross = (pos * aligned).sum(axis=1)

    turnover = pos.diff().abs().sum(axis=1).fillna(pos.abs().sum(axis=1))
    cost = turnover * cost_bp_per_unit_turnover / 100.0
    net = gross - cost

    k = 1.0
    if scale_to_target:
        v = gross.std() * np.sqrt(MONTHS)
        k = ANNUAL_VOL_TARGET / v if v > 1e-9 else 1.0

    return {
        "gross": gross * k,
        "net": net * k,
        "turnover": turnover,
        "positions": pos,
        "leverage": k,
    }


def stats(pnl: pd.Series, bench: pd.Series | None = None) -> dict:
    p = pnl.dropna()
    if len(p) < 12:
        return {}
    ann_ret = p.mean() * MONTHS
    ann_vol = p.std() * np.sqrt(MONTHS)
    downside = p[p < 0].std() * np.sqrt(MONTHS)
    cum = p.cumsum()
    dd = (cum - cum.cummax()).min()
    top5 = p.nlargest(max(int(len(p) * 0.05), 1)).sum()
    out = {
        "ann_return_pct": ann_ret,
        "ann_vol_pct": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 1e-9 else np.nan,
        "sortino": ann_ret / downside if downside and downside > 1e-9 else np.nan,
        "max_drawdown_pct": dd,
        "hit_rate": (p > 0).mean(),
        "pct_pnl_from_top5pct_months": top5 / p.sum() if abs(p.sum()) > 1e-9 else np.nan,
        "n_months": len(p),
    }
    if bench is not None:
        b = bench.reindex(p.index).dropna()
        j = p.reindex(b.index)
        if len(b) > 12 and b.std() > 1e-9:
            out["corr_to_benchmark"] = float(np.corrcoef(j, b)[0, 1])
    return out


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def information_coefficient(fac: pd.DataFrame, sig_col: str = "composite",
                            ret_col: str = "ret_fwd_1m") -> dict:
    """
    Monthly cross-sectional Spearman IC, plus a pooled t-stat that accounts
    for the panel structure by clustering on date (the cross-sectional mean
    of ICs, tested against its own time-series standard error).
    """
    d = fac[["date", "ccy", sig_col, ret_col]].dropna()
    ics = []
    for t, g in d.groupby("date"):
        if len(g) >= 6 and g[sig_col].std() > 1e-9:
            ics.append(g[sig_col].corr(g[ret_col], method="spearman"))
    ics = pd.Series(ics).dropna()
    se = ics.std() / np.sqrt(len(ics)) if len(ics) > 1 else np.nan
    return {
        "mean_ic": ics.mean(),
        "ic_std": ics.std(),
        "ic_t_stat": ics.mean() / se if se and se > 1e-9 else np.nan,
        "ic_hit_rate": (ics > 0).mean(),
        "n_periods": len(ics),
    }


def factor_attribution(fac: pd.DataFrame, tradable: pd.DataFrame,
                       ret_wide: pd.DataFrame) -> pd.DataFrame:
    """Standalone vol-scaled PnL for each factor traded on its own."""
    rows = []
    for f in FACTORS:
        w = fac.pivot(index="date", columns="ccy", values=f + "_rel")
        pos = signal_to_positions(w, tradable)
        r = run_book(pos, ret_wide)
        st = stats(r["gross"])
        ic = information_coefficient(fac, sig_col=f + "_rel")
        rows.append({
            "factor": FACTOR_LABELS[f],
            "sharpe": st.get("sharpe", np.nan),
            "sortino": st.get("sortino", np.nan),
            "max_dd_pct": st.get("max_drawdown_pct", np.nan),
            "mean_ic": ic.get("mean_ic", np.nan),
            "ic_t": ic.get("ic_t_stat", np.nan),
            "ann_turnover_x": r["turnover"].mean() * MONTHS,
        })
    return pd.DataFrame(rows).set_index("factor")


def country_attribution(pnl_positions: pd.DataFrame,
                        ret_wide: pd.DataFrame,
                        leverage: float) -> pd.DataFrame:
    """Total PnL contribution by country, in % of risk capital."""
    contrib = (pnl_positions * ret_wide.reindex(pnl_positions.index)) * leverage
    out = pd.DataFrame({
        "total_pnl_pct": contrib.sum(),
        "avg_abs_position": pnl_positions.abs().mean(),
        "months_held": (pnl_positions.abs() > 1e-6).sum(),
    })
    return out.sort_values("total_pnl_pct", ascending=False)


def beta_decomposition(pnl: pd.Series, ret_wide: pd.DataFrame) -> dict:
    """
    Regress strategy PnL on the equal-weighted basket return, the proxy for
    'am I just long EM duration beta'.  A relative-value book should show
    near-zero loading.
    """
    bench = ret_wide.mean(axis=1)
    d = pd.concat([pnl.rename("pnl"), bench.rename("bench")], axis=1).dropna()
    if len(d) < 24:
        return {}
    x = d["bench"].to_numpy()
    y = d["pnl"].to_numpy()
    beta = np.cov(x, y)[0, 1] / np.var(x)
    alpha = y.mean() - beta * x.mean()
    resid = y - (alpha + beta * x)
    return {
        "beta_to_em_basket": beta,
        "alpha_ann_pct": alpha * MONTHS,
        "r_squared": 1 - resid.var() / y.var(),
        "corr": float(np.corrcoef(x, y)[0, 1]),
    }
