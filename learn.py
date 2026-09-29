"""
learn.py -- sequential, sign-constrained ridge factor weights.

At each cutoff t:
  * training rows are (date, market) pairs whose forward return is fully
    realised by t.  return(m) ends at entry(m+1), the session AFTER
    cutoff(m+1), so only months <= t-2 qualify;
  * target = next-month receiver return per unit of ex-ante monthly vol,
    demeaned across markets each month (the book is paid on RELATIVE,
    risk-adjusted performance);
  * sklearn Ridge(positive=True, fit_intercept=False): coefficients are
    non-negative because every factor is already oriented by theory;
  * the penalty is chosen by expanding-window CV that splits on DATES, never
    rows, scored on mean monthly rank IC;
  * weights are normalised to sum to one.  Before MIN_TRAIN_MONTHS of
    history, or if every coefficient is zero, parity weights are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from config import MIN_TRAIN_MONTHS
from factors import FACTORS

ALPHAS = (1.0, 10.0, 100.0, 1000.0)
N_FOLDS = 4


def training_target(fac: pd.DataFrame) -> pd.Series:
    r = fac["ret_fwd"] / (fac["ret_vol"] / np.sqrt(12.0))
    r = r.where(fac["tradable"])
    return r - r.groupby(fac["date"]).transform("mean")


def _mean_ic(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() >= 5 and np.std(pred[m]) > 1e-12:
            ics.append(spearmanr(pred[m], y[m]).statistic)
    return float(np.nanmean(ics)) if ics else -np.inf


def _fit(X, y, alpha):
    return Ridge(alpha=alpha, positive=True, fit_intercept=False).fit(X, y).coef_


def _choose_alpha(X, y, dates) -> float:
    uniq = np.unique(dates)
    blocks = np.array_split(uniq, N_FOLDS + 1)
    best, best_s = ALPHAS[-1], -np.inf
    for a in ALPHAS:
        scores = []
        for i in range(1, N_FOLDS + 1):
            tr = np.isin(dates, np.concatenate(blocks[:i]))
            te = np.isin(dates, blocks[i])
            if tr.sum() < 60 or te.sum() < 20:
                continue
            w = _fit(X[tr], y[tr], a)
            if w.sum() <= 0:
                continue
            scores.append(_mean_ic(X[te] @ w, y[te], dates[te]))
        s = np.mean(scores) if scores else -np.inf
        if s > best_s:
            best, best_s = a, s
    return best


def sequential_weights(fac: pd.DataFrame) -> pd.DataFrame:
    """DataFrame indexed by date: factor weights used AT that cutoff + alpha."""
    y_all = training_target(fac)
    d = fac.assign(_y=y_all)
    d = d[d["tradable"] & d["_y"].notna()]
    dates = sorted(fac["date"].unique())
    parity = np.ones(len(FACTORS)) / len(FACTORS)
    rows = []
    for t in dates:
        train = d[d["date"] <= t - 2]
        n_months = train["date"].nunique()
        w, alpha, src = parity, np.nan, "parity"
        if n_months >= MIN_TRAIN_MONTHS:
            X = train[FACTORS].fillna(0.0).to_numpy()
            y = train["_y"].to_numpy()
            dd = train["date"].map(lambda p: p.ordinal).to_numpy()
            alpha = _choose_alpha(X, y, dd)
            coef = _fit(X, y, alpha)
            if coef.sum() > 1e-12:
                w, src = coef / coef.sum(), "ridge"
        rows.append(dict(date=t, alpha=alpha, source=src, train_months=n_months,
                         **dict(zip(FACTORS, w))))
    return pd.DataFrame(rows).set_index("date")
