"""
panel.py -- point-in-time monthly panel built from the raw cache.

TIMING CONVENTION
  cutoff(m)  = last calendar day of month m.  Every input is taken as it was
               KNOWN at the cutoff, using the release lags in config.
  entry(m)   = first 5y yield print strictly after cutoff(m): the trade is
               executed at the next session's close.
  return(m)  = full revaluation of a 5y par receiver struck at entry(m),
               marked at entry(m+1), minus floating-leg funding.  Nothing in
               return(m) is observable at cutoff(m).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (CCYS, COV_WINDOW_W, INFLATION_TARGETS, QA_MAX_STALE,
                    QA_MIN_CORR_5_10, RELEASE_LAG_MONTHS, STALE_DAYS)
from pipeline.net import load

TENOR = 5.0
ENTRY_WINDOW_DAYS = 7


# --------------------------------------------------------------------------
# instrument maths
# --------------------------------------------------------------------------
def par_mod_duration(y_pct: np.ndarray | float) -> np.ndarray:
    """Modified duration of a 5y annual-pay par instrument at yield y (%)."""
    y = np.maximum(np.asarray(y_pct, dtype=float) / 100.0, 1e-6)
    return (1.0 - (1.0 + y) ** (-TENOR)) / y


def receiver_excess_return(y0, y1, fund, dt) -> np.ndarray:
    """
    Excess return (% of notional) of receiving fixed y0 on a 5y par swap for
    dt years, marked at yield y1, paying floating `fund` (%).  Exact dirty
    revaluation of the fixed leg: captures carry, duration, convexity and
    pull-to-par of the aged instrument.
    """
    y0 = np.asarray(y0, float) / 100.0
    y1 = np.asarray(y1, float) / 100.0
    f = np.asarray(fund, float) / 100.0
    dt = np.asarray(dt, float)
    pv = np.zeros_like(y1)
    for k in range(1, int(TENOR) + 1):
        pv = pv + y0 / (1.0 + y1) ** (k - dt)
    pv = pv + 1.0 / (1.0 + y1) ** (TENOR - dt)
    return 100.0 * (pv - 1.0 - f * dt)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _months(start="2005-01", end=None) -> pd.PeriodIndex:
    end = end or pd.Timestamp.today().to_period("M")
    return pd.period_range(start, end, freq="M")


def _asof_monthly(ref: pd.DataFrame, months: pd.PeriodIndex, lag: int) -> pd.DataFrame:
    """
    ref: wide [period(M) x ccy].  Value known at cutoff m = latest ref period
    <= m - lag.  Returns wide [month x ccy].
    """
    ref = ref.sort_index()
    shifted = ref.copy()
    shifted.index = shifted.index + lag
    return shifted.reindex(shifted.index.union(months)).sort_index().ffill(limit=3).reindex(months)


def _official_target(ccy: str, year: int) -> float:
    sched = INFLATION_TARGETS[ccy]
    yrs = [y for y in sorted(sched) if y <= year]
    return np.nan if not yrs or sched[yrs[-1]] is None else float(sched[yrs[-1]])


# --------------------------------------------------------------------------
# daily market data
# --------------------------------------------------------------------------
def remove_spikes(s: pd.Series) -> pd.Series:
    """
    Drop one-day spikes that fully reverse the next session (bad ticks):
    |dy_t| and |dy_t+1| both > max(30bp, 3% of level), opposite signs, and
    y_t+1 back within half the spike of y_t-1.
    """
    x = s.dropna()
    up, dn = x.diff(), x.shift(-1) - x
    thr = np.maximum(0.30, 0.03 * x.abs())
    back = (x.shift(-1) - x.shift(1)).abs() < 0.5 * np.minimum(up.abs(), dn.abs())
    bad = (up.abs() > thr) & (dn.abs() > thr) & (np.sign(up) != np.sign(dn)) & back
    return s.where(~s.index.isin(x.index[bad]))


def load_yields() -> dict[str, pd.DataFrame]:
    y = load("tvc_yields", parse_dates=["date"])
    out = {}
    for t, g in y.groupby("tenor"):
        w = g.pivot_table(index="date", columns="ccy", values="value", aggfunc="last").sort_index()
        # a zero or negative print on an EM 5y is a bad tick, not a yield
        w = w.where(w > 0.05).apply(remove_spikes)
        out[t] = w.reindex(columns=CCYS)
    return out


def feed_quality(y5: pd.DataFrame, y10: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trailing (causal) data-quality statistics per market:
      corr510 : 52-week correlation of weekly 5y and 10y changes.  A genuine
                5y benchmark co-moves with the 10y (>0.8 in clean markets);
                low values flag benchmark switches or stale quotes.
      stale   : share of unchanged daily prints over the last 250 prints.
    """
    w5 = y5.resample("W-FRI").last().diff()
    w10 = y10.reindex(columns=y5.columns).resample("W-FRI").last().diff()
    corr = w5.rolling(52, min_periods=30).corr(w10)
    stale = y5.apply(lambda s: (s.dropna().diff() == 0).astype(float)
                     .rolling(250, min_periods=120).mean().reindex(s.index)).ffill()
    return corr, stale


def load_policy() -> pd.DataFrame:
    p = load("bis_policy_daily")
    p["date"] = pd.to_datetime(p["period"])
    return p.pivot_table(index="date", columns="ccy", values="value").sort_index().reindex(columns=CCYS)


def _last_on_or_before(daily: pd.DataFrame, when: pd.Timestamp):
    sub = daily.loc[:when]
    val = sub.ffill().iloc[-1] if len(sub) else pd.Series(np.nan, index=daily.columns)
    last_date = sub.apply(lambda s: s.last_valid_index())
    age = (when - pd.to_datetime(last_date)).dt.days
    return val, age


def _first_after(daily: pd.DataFrame, when: pd.Timestamp, window: int = ENTRY_WINDOW_DAYS):
    sub = daily.loc[when + pd.Timedelta(days=1): when + pd.Timedelta(days=window)]
    vals, dates = {}, {}
    for c in daily.columns:
        s = sub[c].dropna()
        vals[c] = s.iloc[0] if len(s) else np.nan
        dates[c] = s.index[0] if len(s) else pd.NaT
    return pd.Series(vals), pd.Series(dates)


def weekly_changes(y5: pd.DataFrame) -> pd.DataFrame:
    """Friday-sampled 5y yield changes in bp (async closes averaged out)."""
    wk = y5.resample("W-FRI").last()
    fresh = y5.notna().astype(float).resample("W-FRI").max() > 0
    return (wk.diff() * 100.0).where(fresh)


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------
def build_panel(start="2008-01", asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Long panel indexed (date[M], ccy).  If `asof` is given, all daily data
    after `asof` is discarded first (used by the look-ahead test and by the
    live run).
    """
    ylds = load_yields()
    y5, y10 = ylds["05Y"], ylds["10Y"]
    pol = load_policy()
    if asof is not None:
        y5, y10, pol = y5.loc[:asof], y10.loc[:asof], pol.loc[:asof]
    qa_corr, qa_stale = feed_quality(y5, y10)
    last_day = y5.index.max()
    months = _months(start, (asof or last_day).to_period("M"))

    # ---- monthly macro, wide by reference period -------------------------
    cpi = load("bis_cpi")
    cpi["period"] = pd.PeriodIndex(cpi["period"], freq="M")
    cpi_w = cpi.pivot_table(index="period", columns="ccy", values="value").reindex(columns=CCYS)
    yoy = (cpi_w / cpi_w.shift(12) - 1.0) * 100.0
    yoy_known = _asof_monthly(yoy, months, RELEASE_LAG_MONTHS["cpi"])
    yoy_6m_ago = _asof_monthly(yoy.shift(6), months, RELEASE_LAG_MONTHS["cpi"])
    yoy_mean36 = _asof_monthly(yoy.rolling(36, min_periods=24).mean(), months, RELEASE_LAG_MONTHS["cpi"])

    reer = load("bis_reer")
    reer["period"] = pd.PeriodIndex(reer["period"], freq="M")
    reer_w = reer.pivot_table(index="period", columns="ccy", values="value").reindex(columns=CCYS)
    reer_chg = _asof_monthly(np.log(reer_w / reer_w.shift(6)) * 100.0, months, RELEASE_LAG_MONTHS["reer"])
    reer_dev = _asof_monthly(np.log(reer_w / reer_w.rolling(60, min_periods=36).mean()) * 100.0,
                             months, RELEASE_LAG_MONTHS["reer"])

    ctot = load("imf_ctot")
    ctot["period"] = pd.PeriodIndex(ctot["period"], freq="M")
    ctot_w = ctot.pivot_table(index="period", columns="ccy", values="value").reindex(columns=CCYS)
    ctot_chg = _asof_monthly(np.log(ctot_w / ctot_w.shift(6)) * 100.0, months, RELEASE_LAG_MONTHS["ctot"])

    gap = load("bis_credit_gap")
    gap["period"] = pd.PeriodIndex(gap["period"], freq="Q").asfreq("M", how="end")
    gap_w = gap.pivot_table(index="period", columns="ccy", values="value").reindex(columns=CCYS)
    gap_w = gap_w.reindex(pd.period_range(gap_w.index.min(), gap_w.index.max(), freq="M")).ffill(limit=2)
    credit_gap = _asof_monthly(gap_w, months, RELEASE_LAG_MONTHS["credit_gap"])

    weo = load("imf_weo_fiscal")
    weo = weo[weo["vintage"] == "WEO"]
    fis_a = weo.pivot_table(index="period", columns="ccy", values="value").reindex(columns=CCYS)
    fis_m = fis_a.copy()
    fis_m.index = pd.PeriodIndex([f"{y}-12" for y in fis_a.index], freq="M")
    fis_full = fis_m.reindex(pd.period_range(fis_m.index.min(), fis_m.index.max(), freq="M")).ffill()
    fis_chg_full = (fis_m - fis_m.shift(1)).reindex(fis_full.index).ffill()
    fiscal_bal = _asof_monthly(fis_full, months, RELEASE_LAG_MONTHS["fiscal"])
    fiscal_chg = _asof_monthly(fis_chg_full, months, RELEASE_LAG_MONTHS["fiscal"])

    opn = load("wdi_openness")
    op_m = opn.pivot_table(index="period", columns="ccy", values="value").reindex(columns=CCYS)
    op_m.index = pd.PeriodIndex([f"{y}-12" for y in op_m.index], freq="M")
    op_full = op_m.reindex(pd.period_range(op_m.index.min(), op_m.index.max(), freq="M")).ffill()
    openness = _asof_monthly(op_full, months, RELEASE_LAG_MONTHS["openness"]).ffill()

    # ---- daily market data sampled at cutoffs ----------------------------
    wchg = weekly_changes(y5)
    rows = []
    for m in months:
        cut = m.to_timestamp(how="end").normalize()
        if cut > last_day:
            cut = last_day
        yv, yage = _last_on_or_before(y5, cut)
        pv, _ = _last_on_or_before(pol, cut)
        ent, ent_d = _first_after(y5, cut)
        late, late_d = _first_after(y5, cut, window=31)
        wk = wchg.loc[:cut].tail(COV_WINDOW_W)
        ew = wk.ewm(halflife=26, min_periods=26).std().iloc[-1] if len(wk) else pd.Series(np.nan, index=CCYS)
        qc = qa_corr.loc[:cut].iloc[-1] if len(qa_corr.loc[:cut]) else pd.Series(np.nan, index=CCYS)
        qs = qa_stale.loc[:cut].iloc[-1] if len(qa_stale.loc[:cut]) else pd.Series(np.nan, index=CCYS)
        for c in CCYS:
            rows.append(dict(date=m, ccy=c, cutoff=cut, yld_5y=yv[c], yld_age=yage[c],
                             policy=pv[c], entry_yld=ent[c], entry_date=ent_d[c],
                             late_yld=late[c], late_date=late_d[c],
                             yvol_bp=ew[c] * np.sqrt(52) if np.isfinite(ew[c]) else np.nan,
                             qa_corr510=qc[c], qa_stale=qs[c]))
    p = pd.DataFrame(rows)

    def _stack(w, name):
        s = w.stack(future_stack=True).rename(name)
        s.index.names = ["date", "ccy"]
        return s

    macro = pd.concat([
        _stack(yoy_known, "cpi_yoy"), _stack(yoy_6m_ago, "cpi_yoy_6m_ago"),
        _stack(yoy_mean36, "cpi_mean36"), _stack(reer_chg, "reer_chg_6m"),
        _stack(reer_dev, "reer_dev_5y"), _stack(ctot_chg, "ctot_chg_6m"),
        _stack(credit_gap, "credit_gap"), _stack(fiscal_bal, "fiscal_bal"),
        _stack(fiscal_chg, "fiscal_chg"), _stack(openness, "openness"),
    ], axis=1).reset_index()
    p = p.merge(macro, on=["date", "ccy"], how="left")

    # ---- inflation targets -------------------------------------------------
    p["target"] = [_official_target(c, d.year) for c, d in zip(p["ccy"], p["date"])]
    delivered = p["cpi_mean36"]
    p["eff_target"] = np.where(p["target"].notna(),
                               0.5 * p["target"] + 0.5 * delivered.fillna(p["target"]),
                               delivered)

    # ---- instrument quantities -------------------------------------------
    p["mod_dur"] = par_mod_duration(p["yld_5y"])
    p["ret_vol"] = p["yvol_bp"] / 100.0 * p["mod_dur"]            # % of notional, ann.
    # Mark = next-session print; if the market was shut all week (holidays),
    # mark at the last print on/before the cutoff; if the feed has a longer
    # gap, at the first print within 31 days.  Only for months whose mark
    # window has fully elapsed, so the open month stays unmarked.
    fresh = p["yld_age"] <= STALE_DAYS
    closed = p["cutoff"] + pd.Timedelta(days=ENTRY_WINDOW_DAYS) < last_day
    prior_d = p["cutoff"] - pd.to_timedelta(p["yld_age"], unit="D")
    use_entry = p["entry_yld"].notna()
    use_prior = ~use_entry & fresh & closed
    use_late = ~use_entry & ~use_prior & p["late_yld"].notna()
    p["mark_yld"] = np.select([use_entry, use_prior, use_late],
                              [p["entry_yld"], p["yld_5y"], p["late_yld"]], np.nan)
    p["mark_date"] = pd.to_datetime(np.select(
        [use_entry, use_prior, use_late],
        [p["entry_date"].astype("datetime64[ns]"), prior_d, p["late_date"].astype("datetime64[ns]")],
        pd.NaT))
    p = p.sort_values(["ccy", "date"]).reset_index(drop=True)
    g = p.groupby("ccy")
    p["exit_yld"] = g["mark_yld"].shift(-1)
    p["exit_date"] = g["mark_date"].shift(-1)
    dt = (pd.to_datetime(p["exit_date"]) - pd.to_datetime(p["entry_date"])).dt.days / 365.0
    p["hold_years"] = dt
    p["ret_fwd"] = receiver_excess_return(p["entry_yld"], p["exit_yld"], p["policy"], dt)
    next_cut = (p["date"] + 1).map(lambda m: m.to_timestamp(how="end").normalize())
    p["ret_closed"] = next_cut + pd.Timedelta(days=ENTRY_WINDOW_DAYS) < last_day

    # ---- tradability (rule-based, known at cutoff) -----------------------
    # `data_ok` uses only information at or before the cutoff; `tradable`
    # additionally needs an executable entry print after the cutoff.
    p["data_ok"] = (
        p["yld_5y"].notna() & (p["yld_age"] <= STALE_DAYS)
        & p["policy"].notna() & p["ret_vol"].notna() & (p["ret_vol"] > 0)
        & (p["qa_corr510"] >= QA_MIN_CORR_5_10) & (p["qa_stale"] <= QA_MAX_STALE)
    )
    p["tradable"] = p["data_ok"] & p["entry_yld"].notna()
    return p.sort_values(["date", "ccy"]).reset_index(drop=True)


def weekly_returns_matrix(asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Weekly 5y yield changes in bp (for ex-ante covariance in backtest.py)."""
    y5 = load_yields()["05Y"]
    if asof is not None:
        y5 = y5.loc[:asof]
    return weekly_changes(y5)


if __name__ == "__main__":
    p = build_panel()
    pd.set_option("display.width", 220)
    t = p[p["tradable"]]
    print(t.groupby("ccy").agg(first=("date", "min"), last=("date", "max"), months=("date", "size"),
                               ret_mean=("ret_fwd", "mean"), ret_sd=("ret_fwd", "std"),
                               vol=("ret_vol", "median")).round(3).to_string())
    print(p.groupby("date")["tradable"].sum().describe())
    print(p[p["date"] == p["date"].max()].set_index("ccy")[
        ["yld_5y", "policy", "cpi_yoy", "eff_target", "reer_chg_6m", "ctot_chg_6m",
         "credit_gap", "fiscal_bal", "fiscal_chg", "ret_vol", "tradable"]].round(2).to_string())
