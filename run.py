"""
run.py -- full research report on real data.

    python -m pipeline.fetch_all     # refresh raw data (~5 min)
    python run.py                    # backtest + diagnostics -> ./output
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as B
import factors as F
import learn as L
import panel as P
from config import CCYS, INSTRUMENTS, OUT_DIR, UNIVERSE

os.makedirs(OUT_DIR, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

INK, GRID = "#1c1c1c", "#d8d8d8"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
})


def _fmt(d: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def build():
    panel = P.build_panel()
    fac = F.build_factors(panel)
    wk = P.weekly_returns_matrix()
    weights = L.sequential_weights(fac)
    ridge = B.weighted_signal(fac, weights)
    return panel, fac, wk, weights, ridge


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------
def plot_equity(curves: dict, path: str):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    colors = ["#245c48", "#6f9e8a", "#c07a2e", "#3a5f8a", "#8c2d2d"]
    for (k, v), c in zip(curves.items(), colors):
        ax.plot(v.index.to_timestamp(), v.cumsum(), lw=1.6, color=c, label=k)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_ylabel("cumulative return, % of risk capital")
    ax.set_title("EM 5y receiver RV book: cumulative PnL, 10% ex-ante vol target\n"
                 "Real data; out-of-sample ridge weights; costs per config.INSTRUMENTS",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_weights(w: pd.DataFrame, path: str):
    w = w[w["source"] == "ridge"]
    if w.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.stackplot(w.index.to_timestamp(), [w[f].to_numpy() for f in F.FACTORS],
                 labels=[F.FACTOR_LABELS[f] for f in F.FACTORS],
                 colors=["#8c2d2d", "#245c48", "#c9b37e", "#c07a2e", "#3a5f8a", "#6f9e8a"], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("weight")
    ax.set_title("Sequential sign-constrained ridge weights (expanding window, used out of sample)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_scorecard(z: pd.DataFrame, asof: str, path: str):
    cols = F.FACTORS + ["composite"]
    labels = [F.FACTOR_LABELS[c] for c in F.FACTORS] + ["COMPOSITE"]
    m = z[cols].to_numpy(dtype=float)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rv", ["#8c2d2d", "#c86a5a", "#f2efe9", "#6f9e8a", "#245c48"])
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(z) + 1.6))
    im = ax.imshow(m, cmap=cmap, vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([f"{c}  {UNIVERSE[c]['name']}" for c in z.index], fontsize=8.5)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=7.4,
                        color="white" if abs(m[i, j]) > 1.25 else INK,
                        weight="bold" if j == len(cols) - 1 else "normal")
    ax.axvline(len(F.FACTORS) - 0.5, color=INK, lw=1.4)
    ax.set_title(f"EM local rates RV scorecard, cutoff {asof}\n"
                 "Cross-sectional z-scores. Green = receive 5y vs basket", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02).set_label("z-score", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
def data_quality_report() -> pd.DataFrame:
    """TradingView 10y (monthly mean) vs OECD/FRED 10y monthly average."""
    try:
        from pipeline.net import load
        fred = load("fred_10y_check", parse_dates=["date"])
    except FileNotFoundError:
        return pd.DataFrame()
    y10 = P.load_yields()["10Y"]
    rows = []
    for ccy, g in fred.groupby("ccy"):
        f = g.set_index("date")["value"].resample("MS").last()
        t = y10[ccy].resample("MS").mean()
        x = pd.concat([f.rename("fred"), t.rename("tvc")], axis=1).dropna()
        if len(x) < 12:
            continue
        rows.append(dict(ccy=ccy, months=len(x), corr_levels=x.corr().iloc[0, 1],
                         corr_changes=x.diff().corr().iloc[0, 1],
                         mean_abs_diff_bp=(x["tvc"] - x["fred"]).abs().mean() * 100))
    return pd.DataFrame(rows).set_index("ccy")


def main():
    print("=" * 78)
    print("CROSS-COUNTRY EM LOCAL RATES RV -- REAL DATA BACKTEST")
    print("=" * 78)
    panel, fac, wk, weights, ridge = build()
    parts = B.attribution_parts(fac, weights)

    books = {
        "ridge": B.run_book(fac, ridge, wk, parts=parts),
        "parity": B.run_book(fac, fac["parity"], wk),
    }
    first_ridge = weights.index[weights["source"] == "ridge"].min()
    last_real = books["ridge"]["pnl"].index[books["ridge"]["pnl"]["n_mkts"] > 0].max()
    oos = lambda s: s.loc[first_ridge:last_real]
    basket = B.em_basket(fac)

    # ---- headline ----------------------------------------------------------
    head = {}
    for name, bk in books.items():
        pn = oos(bk["pnl"])
        head[f"{name} gross"] = B.stats(pn["gross"])
        head[f"{name} net"] = B.stats(pn["net"])
    hdr = pd.DataFrame(head)
    print(f"\n[1] HEADLINE, out-of-sample window {first_ridge} .. {last_real}\n")
    print(hdr.drop(["start", "end"]).astype(float).round(3).to_string())
    pn = oos(books["ridge"]["pnl"])
    cost_py = (pn["trade_cost"].mean() * 12, pn["roll_cost"].mean() * 12)
    print(f"\n  ridge costs: trade {cost_py[0]:.2f}%/yr, roll {cost_py[1]:.2f}%/yr; "
          f"avg gross notional {pn['gross_lev'].mean():.2f}x capital; "
          f"avg markets {pn['n_mkts'].mean():.1f}; DV01 turnover {pn['dv01_turnover'].mean() * 12:.1f} yrs/yr")

    beta = {k: B.beta_to_basket(oos(bk["pnl"])["net"], basket) for k, bk in books.items()}
    print("\n[2] BETA TO EQUAL-RISK EM 5Y RECEIVER BASKET (net PnL, monthly OLS)\n")
    print(pd.DataFrame(beta).round(3).to_string())

    # ---- IC ------------------------------------------------------------------
    oos_dates = pn.index
    fac_s = fac.assign(ridge=ridge)
    ic = {"Composite (ridge)": B.information_coefficient(fac_s, "ridge", oos_dates),
          "Composite (parity)": B.information_coefficient(fac_s, "parity", oos_dates)}
    for f in F.FACTORS:
        ic[F.FACTOR_LABELS[f]] = B.information_coefficient(fac, f, oos_dates)
    ic_df = pd.DataFrame(ic).T
    print("\n[3] MONTHLY RANK IC vs next-month vol-adjusted receiver return (OOS window)\n")
    print(ic_df.round(3).to_string())

    # ---- robustness ----------------------------------------------------------
    rob = {}
    for cm in (0.0, 1.0, 2.0, 3.0):
        r = B.run_book(fac, ridge, wk, cost_mult=cm)
        rob[f"cost x{cm:g}"] = B.stats(oos(r["pnl"])["net"])["sharpe"]
    for sp in (1.0, 0.33):
        r = B.run_book(fac, ridge, wk, speed=sp)
        rob[f"rebalance speed {sp:g}"] = B.stats(oos(r["pnl"])["net"])["sharpe"]
    net = pn["net"]
    for a, b in (("2011", "2015"), ("2016", "2020"), ("2021", "2026")):
        s = net.loc[a:b]
        if len(s) >= 12:
            rob[f"subperiod {a}-{b}"] = B.stats(s)["sharpe"]
    ex_try = fac.assign(tradable=fac["tradable"] & (fac["ccy"] != "TRY"),
                        data_ok=fac["data_ok"] & (fac["ccy"] != "TRY"))
    r = B.run_book(ex_try, B.weighted_signal(ex_try, weights), wk)
    rob["excluding TRY"] = B.stats(oos(r["pnl"])["net"])["sharpe"]
    rob_s = pd.Series(rob, name="net Sharpe")
    print("\n[4] ROBUSTNESS (ridge book, net Sharpe)\n")
    print(rob_s.round(3).to_string())

    # ---- attribution ---------------------------------------------------------
    at = books["ridge"]["attribution"]
    at = at[at["date"].between(first_ridge, last_real)]
    fa = at.pivot(index="date", columns="factor", values="pnl")
    fattr = pd.DataFrame({
        "total_pnl_pct": fa.sum(),
        "ann_contrib_pct": fa.mean() * 12,
        "standalone_sharpe": fa.mean() / fa.std() * np.sqrt(12),
    })
    fattr.index = [F.FACTOR_LABELS.get(i, i) for i in fattr.index]
    fattr = fattr.sort_values("total_pnl_pct", ascending=False)
    print("\n[5] PnL ATTRIBUTION BY FACTOR (gross, ridge book, OOS)\n")
    print(fattr.round(3).to_string())

    pos = books["ridge"]["positions"]
    pos = pos[pos["date"].between(first_ridge, last_real)]
    cattr = pos.groupby("ccy").agg(total_pnl_pct=("pnl", "sum"),
                                   avg_abs_dv01_yrs=("dv01_yrs", lambda s: s.abs().mean()),
                                   months_held=("notional", lambda s: (s.abs() > 1e-6).sum()),
                                   pct_months_receive=("notional", lambda s: (s > 0).mean()))
    cattr = cattr.sort_values("total_pnl_pct", ascending=False)
    print("\n[6] PnL ATTRIBUTION BY COUNTRY (gross, ridge book, OOS)\n")
    print(cattr.round(3).to_string())

    corr = F.factor_correlation(fac)
    print("\n[7] FACTOR CORRELATION (pooled, data-ok rows)\n")
    print(corr.round(2).to_string())
    wlast = weights.iloc[-1]
    print(f"\n[8] RIDGE WEIGHTS AT {weights.index[-1]} (alpha={wlast['alpha']:g})\n")
    print(wlast[F.FACTORS].rename(F.FACTOR_LABELS).astype(float).round(3).to_string())

    dq = data_quality_report()
    if not dq.empty:
        print("\n[9] DATA QA: TradingView 10y vs OECD/FRED 10y monthly averages\n")
        print(dq.round(3).to_string())

    # ---- live scorecard --------------------------------------------------------
    last = fac["date"].max()
    z = fac[(fac["date"] == last) & fac["data_ok"]].set_index("ccy")[F.FACTORS].copy()
    z["composite"] = ridge[(fac["date"] == last) & fac["data_ok"]].to_numpy()
    z = z.sort_values("composite", ascending=False)
    cut = fac.loc[fac["date"] == last, "cutoff"].max().date()
    plot_scorecard(z, str(cut), os.path.join(OUT_DIR, "scorecard.png"))
    z.round(3).to_csv(os.path.join(OUT_DIR, "scorecard.csv"))
    print(f"\n[10] SCORECARD at cutoff {cut} (markets passing data QA)\n")
    print(z.round(2).to_string())

    # ---- files -----------------------------------------------------------------
    plot_equity({"Ridge (net)": oos(books["ridge"]["pnl"])["net"],
                 "Ridge (gross)": oos(books["ridge"]["pnl"])["gross"],
                 "Parity (net)": oos(books["parity"]["pnl"])["net"]},
                os.path.join(OUT_DIR, "equity_curves.png"))
    plot_weights(weights, os.path.join(OUT_DIR, "ridge_weights.png"))
    hdr.to_csv(os.path.join(OUT_DIR, "headline.csv"))
    ic_df.round(4).to_csv(os.path.join(OUT_DIR, "ic.csv"))
    fattr.round(4).to_csv(os.path.join(OUT_DIR, "factor_attribution.csv"))
    cattr.round(4).to_csv(os.path.join(OUT_DIR, "country_attribution.csv"))
    corr.round(3).to_csv(os.path.join(OUT_DIR, "factor_correlation.csv"))
    weights.round(4).to_csv(os.path.join(OUT_DIR, "ridge_weights.csv"))
    books["ridge"]["pnl"].round(5).to_csv(os.path.join(OUT_DIR, "pnl_monthly.csv"))
    rob_s.round(4).to_csv(os.path.join(OUT_DIR, "robustness.csv"))
    if not dq.empty:
        dq.round(4).to_csv(os.path.join(OUT_DIR, "data_quality.csv"))
    summary = {
        "generated_utc": str(pd.Timestamp.now("UTC")),
        "oos_window": [str(first_ridge), str(last_real)],
        "universe": CCYS,
        "ridge_net": _fmt(head["ridge net"]), "ridge_gross": _fmt(head["ridge gross"]),
        "parity_net": _fmt(head["parity net"]),
        "beta_ridge_net": _fmt(beta["ridge"]),
        "ic_composite_ridge": _fmt(ic["Composite (ridge)"]),
        "costs_pct_per_year": {"trade": round(cost_py[0], 3), "roll": round(cost_py[1], 3)},
        "robustness_net_sharpe": {k: round(float(v), 3) for k, v in rob.items()},
        "cost_model_bp": {c: v["bid_offer_bp"] for c, v in INSTRUMENTS.items()},
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
