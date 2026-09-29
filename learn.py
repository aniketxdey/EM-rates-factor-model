"""
learn.py -- sequential, sign-constrained signal combination.

Conceptual parity (equal weights) is the baseline and the honest default.
This module provides the one alternative the literature supports: a
sequentially re-fit, regularised regression with NON-NEGATIVITY constraints
on the factor coefficients.

Three properties make it a valid backtest rather than a curve fit:

  1. EXPANDING WINDOW.  At each rebalance date the model is fit only on data
     available up to that date.  The weights used in month t never saw
     month t+1.
  2. SIGN CONSTRAINTS.  Coefficients are restricted to be non-negative.
     Since every factor is already oriented by theory (see factors.R6), this
     forbids the learner from discovering that, say, high inflation predicts
     bond rallies -- the classic symptom of fitting noise.
  3. MINIMUM HISTORY.  No signal is emitted until MIN_TRAIN months exist.

The bias-variance trade-off for macro panels is steep: macro regimes are
few and long. Regularisation is therefore strong by default and the model
grid is deliberately small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors import FACTORS, FACTOR_LABELS

MIN_TRAIN = 60         # months before the learner speaks
SHRINK = 0.5           # blend weight toward conceptual parity (fixed a priori)
ALPHAS = (1.0, 10.0, 100.0)


def _nnls_ridge(X: np.ndarray, y: np.ndarray, alpha: float,
                iters: int = 400) -> np.ndarray:
    """
    Ridge regression with non-negativity, by projected gradient.
    Small problem (6 features), so this is fast and dependency-free.
    """
    n, k = X.shape
    w = np.zeros(k)
    XtX = X.T @ X / n + alpha * np.eye(k)
    Xty = X.T @ y / n
    L = np.linalg.eigvalsh(XtX).max()
    step = 1.0 / max(L, 1e-9)
    for _ in range(iters):
        grad = XtX @ w - Xty
        w = np.maximum(w - step * grad, 0.0)
    return w


def _cv_score(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
              alpha: float, n_folds: int = 4) -> float:
    """
    Expanding-window panel CV.  Splits on DATES, never on rows, so the same
    month never appears in both train and test -- the panel analogue of
    scikit-learn's TimeSeriesSplit.
    """
    uniq = np.unique(dates)
    if len(uniq) < n_folds + 2:
        return -np.inf
    bounds = np.array_split(uniq, n_folds + 1)
    scores = []
    for i in range(1, n_folds + 1):
        tr_dates = np.concatenate(bounds[:i])
        te_dates = bounds[i]
        tr = np.isin(dates, tr_dates)
        te = np.isin(dates, te_dates)
        if tr.sum() < 40 or te.sum() < 10:
            continue
        w = _nnls_ridge(X[tr], y[tr], alpha)
        pred = X[te] @ w
        if pred.std() < 1e-12:
            continue
        # score = stylised long/short Sharpe, the criterion that matches use
        pnl = np.sign(pred - pred.mean()) * y[te]
        if pnl.std() < 1e-12:
            continue
        scores.append(pnl.mean() / pnl.std())
    return float(np.mean(scores)) if scores else -np.inf


def sequential_signal(fac: pd.DataFrame,
                      ret_col: str = "ret_fwd_1m") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (signal_long, weights_history).

    signal_long: date, ccy, ml_signal
    weights_history: date-indexed factor weights actually used
    """
    cols = [f + "_rel" for f in FACTORS]
    d = fac[["date", "ccy", ret_col] + cols].dropna().copy()

    # CRITICAL: the target must match the trading objective.  Raw forward
    # returns are dominated by a common global duration factor (cross-country
    # monthly return correlations of 0.3-0.7).  Regressing cross-sectional
    # factors on raw returns therefore fits mostly common-factor noise and
    # produces a signal with an information coefficient near zero.  A
    # relative-value book is paid on RELATIVE returns, so we demean the
    # target cross-sectionally at each date before fitting.
    d["_y"] = d.groupby("date")[ret_col].transform(lambda s: s - s.mean())
    ret_col = "_y"

    d = d.sort_values("date")
    all_dates = np.array(sorted(d["date"].unique()))

    sig_rows, w_rows = [], []
    for i, t in enumerate(all_dates):
        if i < MIN_TRAIN:
            continue
        # training data: strictly before t, and the target return for month
        # t-1 is only known at t, so we drop the most recent month too.
        train = d[d["date"] < all_dates[i - 1]]
        if len(train) < 100:
            continue
        Xtr = train[cols].to_numpy(dtype=float)
        ytr = train[ret_col].to_numpy(dtype=float)
        dtr = train["date"].to_numpy()

        best_a, best_s = ALPHAS[-1], -np.inf
        for a in ALPHAS:
            s = _cv_score(Xtr, ytr, dtr, a)
            if s > best_s:
                best_a, best_s = a, s
        w = _nnls_ridge(Xtr, ytr, best_a)
        if w.sum() < 1e-9:
            w = np.ones(len(cols)) / len(cols)
        w = w / w.sum()

        # SHRINKAGE TOWARD CONCEPTUAL PARITY.
        # Fitted weights are informative but unstable early in the sample,
        # when few macro regimes have been observed.  We therefore blend the
        # learned vector toward equal weights.  SHRINK is fixed a priori, not
        # tuned on results: it encodes the prior that all six factors have
        # genuine theoretical backing, so no factor should ever be zeroed out
        # on the basis of a short sample.
        parity = np.ones(len(cols)) / len(cols)
        w = SHRINK * parity + (1.0 - SHRINK) * w

        cur = d[d["date"] == t]
        pred = cur[cols].to_numpy(dtype=float) @ w
        sig_rows.append(pd.DataFrame({
            "date": cur["date"].to_numpy(),
            "ccy": cur["ccy"].to_numpy(),
            "ml_signal": pred,
        }))
        w_rows.append(dict(zip([FACTOR_LABELS[f] for f in FACTORS], w), date=t, alpha=best_a))

    if not sig_rows:
        return pd.DataFrame(columns=["date", "ccy", "ml_signal"]), pd.DataFrame()

    sig = pd.concat(sig_rows, ignore_index=True)
    # normalise cross-sectionally so it is comparable to the parity composite
    wide = sig.pivot(index="date", columns="ccy", values="ml_signal")
    wide = wide.sub(wide.mean(axis=1), axis=0)
    wide = wide.div(wide.std(axis=1).replace(0, np.nan), axis=0).clip(-3, 3)
    sig = wide.stack(future_stack=True).rename("ml_signal").reset_index()
    return sig, pd.DataFrame(w_rows).set_index("date")
