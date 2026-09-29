"""
run.py -- end-to-end pipeline.

  python run.py            # full run, writes charts + tables to ./output
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import data
import factors as F
import backtest as B
import learn as L

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

INK = "#1c1c1c"
GRID = "#d8d8d8"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


# --------------------------------------------------------------------------
def live_scorecard() -> pd.DataFrame:
    """
    Current factor scores from the REAL Sep-2026 snapshot.

    Uses the same scaling rules as the panel factors but computes scores
    cross-sectionally on one date, which is what a desk scorecard is.
    """
    s = data.snapshot_frame().copy()
    eff_t = 0.5 * s["target"] + 0.5 * s["cpi_yoy"].clip(lower=0)
    scale = np.maximum(eff_t, F.TARGET_FLOOR)

    raw = pd.DataFrame(index=s.index)
    # 1. inflation pressure (negative = bearish duration)
    raw["inflation_pressure"] = -((s["cpi_yoy"] - eff_t) / scale)
    # 2. real yield & carry
    raw["real_carry"] = (s["policy_rate"] - s["cpi_yoy"]) / scale
    # 3. FX valuation proxy: consensus disinflation slope x openness
    raw["fx_valuation"] = (-s["cpi_slope"]) * np.sqrt(s["openness"].clip(lower=0.1))
    # 4. terms of trade: commodity beta net of energy dependence
    raw["terms_of_trade"] = s["comdty_beta"] - 0.5 * s["energy_dep"]
    # 5. fiscal thrust
    raw["fiscal_thrust"] = 0.6 * (s["fiscal_bal"] / 5.0) + 0.4 * s["fiscal_chg"]
    # 6. credit shortfall
    raw["credit_gap_shortfall"] = -s["credit_gap"]

    z = raw.sub(raw.mean()).div(raw.std().replace(0, np.nan)).clip(-3, 3)
    z["COMPOSITE"] = z[F.FACTORS].mean(axis=1)
    z["COMPOSITE"] = (z["COMPOSITE"] - z["COMPOSITE"].mean()) / z["COMPOSITE"].std()
    z.insert(0, "country", s["name"])
    return z.sort_values("COMPOSITE", ascending=False)


def plot_scorecard(z: pd.DataFrame, path: str):
    cols = F.FACTORS + ["COMPOSITE"]
    labels = [F.FACTOR_LABELS[c] for c in F.FACTORS] + ["COMPOSITE"]
    m = z[cols].to_numpy(dtype=float)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rv", ["#8c2d2d", "#c86a5a", "#f2efe9", "#6f9e8a", "#245c48"])

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(m, cmap=cmap, vmin=-2.2, vmax=2.2, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([f"{c}  {n}" for c, n in zip(z.index, z["country"])], fontsize=8.5)

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7.4, weight="bold" if j == len(cols) - 1 else "normal",
                        color="white" if abs(v) > 1.25 else INK)

    ax.axvline(len(F.FACTORS) - 0.5, color=INK, lw=1.4)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(z), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)

    ax.set_title("EM local rates relative-value scorecard\n"
                 "Cross-sectional z-scores, Sep 2026. Green = long 5y receiver vs basket",
                 fontsize=10.5, loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("z-score", fontsize=8)
    cb.outline.set_edgecolor(GRID)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_equity(curves: dict, path: str):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    colors = {"Conceptual parity (gross)": "#245c48",
              "Conceptual parity (net)": "#6f9e8a",
              "Threshold variant (net)": "#c07a2e",
              "Sequential ML (gross)": "#3a5f8a"}
    for k, v in curves.items():
        ax.plot(v.index.to_timestamp(), v.cumsum(), lw=1.6,
                color=colors.get(k, "#888"), label=k)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_ylabel("cumulative return, % of risk capital")
    ax.set_title("Cross-country EM duration RV -- cumulative PnL (10% vol target)\n"
                 "CALIBRATED SYNTHETIC PANEL -- not a live track record",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_weights(w: pd.DataFrame, path: str):
    if w.empty:
        return
    cols = [c for c in w.columns if c != "alpha"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.stackplot(w.index.to_timestamp(), [w[c].to_numpy() for c in cols],
                 labels=cols, colors=["#245c48", "#6f9e8a", "#c9b37e",
                                      "#c07a2e", "#8c2d2d", "#3a5f8a"], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("weight")
    ax.set_title("Sequentially learned factor weights (non-negative ridge, expanding window)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("EM LOCAL RATES -- SIX-FACTOR CROSS-COUNTRY RELATIVE VALUE")
    print("=" * 74)

    # ---- live scorecard from real data
    z = live_scorecard()
    plot_scorecard(z, os.path.join(OUT, "scorecard.png"))
    z.round(2).to_csv(os.path.join(OUT, "scorecard.csv"))
    print("\n[1] LIVE SCORECARD (real data, Sep 2026)\n")
    print(z[["country"] + F.FACTORS + ["COMPOSITE"]].round(2).to_string())

    # ---- panel + factors
    panel = data.load_panel()
    fac = F.build_factors(panel)

    ret_wide = fac.pivot(index="date", columns="ccy", values="ret_fwd_1m")
    trad = fac.pivot(index="date", columns="ccy", values="tradable").fillna(False)
    comp = fac.pivot(index="date", columns="ccy", values="composite")

    corr = F.factor_correlation(fac)
    corr.round(2).to_csv(os.path.join(OUT, "factor_correlation.csv"))
    print("\n[2] FACTOR CROSS-CORRELATION (pooled)\n")
    print(corr.round(2).to_string())

    # ---- conceptual parity book
    pos = B.signal_to_positions(comp, trad)
    book = B.run_book(pos, ret_wide)
    st_gross = B.stats(book["gross"])
    st_net = B.stats(book["net"])

    # ---- threshold variant
    pos_t = B.threshold_positions(comp, trad)
    book_t = B.run_book(pos_t, ret_wide)
    st_thr = B.stats(book_t["net"])

    # ---- sequential ML
    ml_sig, w_hist = L.sequential_signal(fac)
    curves = {"Conceptual parity (gross)": book["gross"],
              "Conceptual parity (net)": book["net"],
              "Threshold variant (net)": book_t["net"]}
    st_ml = {}
    if not ml_sig.empty:
        ml_wide = ml_sig.pivot(index="date", columns="ccy", values="ml_signal")
        ml_wide = ml_wide.reindex(index=comp.index, columns=comp.columns)
        pos_ml = B.signal_to_positions(ml_wide, trad)
        book_ml = B.run_book(pos_ml, ret_wide)
        st_ml = B.stats(book_ml["gross"])
        curves["Sequential ML (gross)"] = book_ml["gross"]
        plot_weights(w_hist, os.path.join(OUT, "ml_weights.png"))
        w_hist.round(3).to_csv(os.path.join(OUT, "ml_weights.csv"))

    plot_equity(curves, os.path.join(OUT, "equity_curves.png"))

    # ---- diagnostics
    ic = B.information_coefficient(fac)
    beta = B.beta_decomposition(book["gross"], ret_wide)
    fattr = B.factor_attribution(fac, trad, ret_wide)
    cattr = B.country_attribution(book["positions"], ret_wide, book["leverage"])

    fattr.round(3).to_csv(os.path.join(OUT, "factor_attribution.csv"))
    cattr.round(3).to_csv(os.path.join(OUT, "country_attribution.csv"))

    print("\n[3] HEADLINE PERFORMANCE (synthetic panel -- pipeline validation only)\n")
    hdr = pd.DataFrame({
        "Parity gross": st_gross, "Parity net": st_net,
        "Threshold net": st_thr, "Sequential ML gross": st_ml,
    })
    print(hdr.round(3).to_string())

    print("\n[4] PREDICTIVE DIAGNOSTICS\n")
    print(pd.Series(ic).round(4).to_string())
    print("\n  beta decomposition vs equal-weighted EM basket:")
    print(pd.Series(beta).round(4).to_string())

    print("\n[5] FACTOR ATTRIBUTION (each factor traded standalone)\n")
    print(fattr.round(3).to_string())

    print("\n[6] COUNTRY ATTRIBUTION (top and bottom 5)\n")
    print(pd.concat([cattr.head(5), cattr.tail(5)]).round(3).to_string())

    if not w_hist.empty:
        print("\n[7] FINAL LEARNED WEIGHTS\n")
        print(w_hist.drop(columns=["alpha"]).iloc[-1].round(3).to_string())

    summary = {
        "generated": str(pd.Timestamp.now("UTC")),
        "universe": data.UNIVERSE,
        "panel_range": [str(panel["date"].min()), str(panel["date"].max())],
        "parity_gross": st_gross, "parity_net": st_net,
        "threshold_net": st_thr, "ml_gross": st_ml,
        "ic": ic, "beta": beta,
        "WARNING": ("Performance statistics are computed on a CALIBRATED "
                    "SYNTHETIC panel. They validate the pipeline, not the "
                    "strategy. The live scorecard uses real data."),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
