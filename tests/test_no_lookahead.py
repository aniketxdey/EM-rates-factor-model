"""
Look-ahead tests.  Run:  python -m pytest -q tests

1. Truncation: rebuilding the panel with all daily market data cut at a past
   date must leave every signal input, factor and tradability flag for
   earlier months unchanged.
2. Learner: scrambling forward returns from month t-1 onward must not change
   the ridge weights used at month t.
3. Book: positions at month t must not change when later returns change.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest as B  # noqa: E402
import factors as F  # noqa: E402
import learn as L  # noqa: E402
import panel as P  # noqa: E402

ASOF = pd.Timestamp("2019-06-28")
SIGNAL_COLS = ["yld_5y", "policy", "cpi_yoy", "cpi_yoy_6m_ago", "eff_target", "reer_chg_6m",
               "ctot_chg_6m", "credit_gap", "fiscal_bal", "fiscal_chg", "openness",
               "yvol_bp", "ret_vol", "qa_corr510", "qa_stale", "data_ok"]


@pytest.fixture(scope="module")
def full():
    p = P.build_panel()
    return p, F.build_factors(p)


def test_truncated_panel_matches(full):
    p_full, f_full = full
    p_cut = P.build_panel(asof=ASOF)
    f_cut = F.build_factors(p_cut)
    # compare months strictly before the truncation month (the spike filter
    # needs one print after a day to classify it)
    keep = lambda d: d[d["date"] < ASOF.to_period("M") - 1].set_index(["date", "ccy"]).sort_index()
    a, b = keep(f_full), keep(f_cut)
    cols = SIGNAL_COLS + F.FACTORS + ["parity"]
    pd.testing.assert_frame_equal(a[cols], b.loc[a.index, cols], check_dtype=False, atol=1e-10)


def test_ridge_weights_ignore_future_returns(full):
    _, fac = full
    t = pd.Period("2018-06", "M")
    base = L.sequential_weights(fac[fac["date"] <= t]).loc[t]
    rng = np.random.default_rng(0)
    scr = fac.copy()
    m = scr["date"] >= t - 1
    scr.loc[m, "ret_fwd"] = rng.normal(0, 5, m.sum())
    pert = L.sequential_weights(scr[scr["date"] <= t]).loc[t]
    pd.testing.assert_series_equal(base[F.FACTORS].astype(float), pert[F.FACTORS].astype(float))


def test_positions_ignore_future_returns(full):
    _, fac = full
    wk = P.weekly_returns_matrix()
    t = pd.Period("2017-12", "M")
    r1 = B.run_book(fac, fac["parity"], wk)["positions"]
    scr = fac.copy()
    m = scr["date"] >= t
    scr.loc[m, "ret_fwd"] = scr.loc[m, "ret_fwd"] * -3.0
    r2 = B.run_book(scr, scr["parity"], wk)["positions"]
    a = r1[r1["date"] <= t].set_index(["date", "ccy"])["notional"]
    b = r2[r2["date"] <= t].set_index(["date", "ccy"])["notional"]
    pd.testing.assert_series_equal(a, b)
