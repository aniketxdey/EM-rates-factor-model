# Cross-Country EM Local Rates Relative Value

A six-factor, point-in-time macro model that ranks emerging-market local
currency rates markets against each other and trades the dispersion as a
market-neutral, volatility-targeted book of 5-year fixed-receiver positions.

The deliverable is three things: a **live scorecard** built from real macro
data, a **backtest engine** with full PnL attribution, and an honest account
of **what the results do and do not establish**.

---

## 1. The thesis

Returns on EM duration are dominated by a global factor. Monthly return
correlations between EM rates markets sit broadly in the 0.3–0.7 range, which
means a directional long-duration book is mostly a bet on US rates wearing a
costume. Stripping out the common factor and trading only the cross-section
isolates whatever country-level macro information the market has not yet
priced.

The premise is rational inattention: markets do not fully and promptly price
slow-moving divergences in inflation, real yields, external balances and
fiscal stance across seventeen countries, because doing so is tedious and
nobody is paid to watch Romania closely.

## 2. Universe

Seventeen EM currency areas with liquid local rates markets and floating
exchange rates:

`BRL CLP COP CZK HUF IDR INR KRW MXN MYR PEN PHP PLN RON THB TRY ZAR`

Turkey is deliberately **retained**, not dropped. Excluding markets that blew
up is the single most common source of survivorship bias in EM backtests. It
is instead handled by a tradability blacklist that suspends positions during
windows of policy discontinuity or capital controls, which is what a desk
would actually have experienced.

## 3. The six factors

Each factor occupies a distinct economic channel. Signs are imposed from
theory and never fitted. Positive = expect this market's 5y receiver to
**outperform** the basket.

| Factor | Channel | Construction |
|---|---|---|
| Inflation pressure | Policy reaction function | Excess CPI vs *effective* target, averaged over y/y, 6m/6m and 3m/3m lookbacks, ratio-scaled, blended with decayed CPI surprises. **Sign: negative.** |
| Real yield & carry | Risk premium / valuation | Ex-ante 5y real yield plus vol-targeted carry |
| FX valuation | External pass-through | Real effective appreciation scaled by trade openness |
| Terms of trade | Commodity shock | Commodity-basket-weighted ToT change, net of energy import dependence |
| Fiscal thrust | Fiscal | Government balance level plus 12m change |
| Credit shortfall | Domestic demand | Private credit growth below nominal trend |

### Why six, and why these six

Twelve-factor frameworks in the literature collapse to roughly five dominant
factors under regularisation. Six is enough to demonstrate conceptual breadth
and to make a regularisation step meaningful, without fitting noise across a
sample containing only a handful of genuine macro regimes.

Critically, the three factors an inflation-focused model would naturally pick
— surprise, momentum, real rate — are **not** three independent things.
Surprise and momentum are two inputs to one concept (pressure); the real rate
is a valuation factor, not an inflation factor. The added three (FX, terms of
trade, fiscal) were chosen because they occupy channels the original three
leave empty, and because post-2020 evidence indicates energy price shocks and
inflation history explain more cross-country inflation variance than the
conventional fundamentals do.

## 4. Methodology decisions that actually matter

These are the choices that separate a working backtest from a contaminated
one. Each is implemented in `factors.py` and labelled `R1`–`R6`.

**R1 — Ratio scaling.** A 2pp target miss means something different in
Czechia than in Turkey. Every inflation quantity is divided by
`max(effective_target, 2.0)` before normalisation. Without this, Turkey
single-handedly determines the book.

**R2 — Effective, not official, targets.** The effective target blends the
stated target with trailing delivered inflation. Turkey's official target has
been 5% through a period of 30%+ inflation; treating that as a 25-standard-
deviation signal is not information, it is a broken denominator.

**R3 — Sequential normalisation.** At each date, scaling uses only the panel
observed up to that date. Full-sample z-scoring leaks the future through the
denominator. This is subtle, common, and fatal.

**R4 — Winsorisation** at ±3 standard deviations.

**R5 — Relative, not absolute.** Every factor is expressed versus the
concurrently-tradable basket mean.

**R6 — Theory-imposed signs.** Enforced again in the learner via
non-negativity constraints, which prevents the model from "discovering" that
high inflation predicts bond rallies.

**Timing.** Signals at month-end *t*, held over *[t, t+1]*. Returns are
strictly forward. Positions are vol-targeted at 10% annualised per unit of
signal, which makes Malaysia and Turkey directly comparable without a
separate DV01 overlay.

## 5. Data provenance — read this before trusting any number

The repository keeps two data sources strictly separate, and conflating them
would be the easiest way to mislead yourself.

### Real (drives the live scorecard)

| Series | Source | Status |
|---|---|---|
| Headline CPI %y/y, Aug 2026 print | Trading Economics | **Real, cited** |
| Consensus CPI path Q3/26–Q2/27 | Trading Economics | **Real, cited** |
| Policy rates, Sep 2026 | Central bank decision pages via Wikipedia compilation | **Real, cited** |
| Official inflation targets | Central bank published targets | **Real** |
| Trade openness, energy dependence, commodity beta, fiscal balances, foreign ownership, yield vol | — | **Analyst estimates, not transcribed** |

That last row is the weakest link and is flagged as such in `data.py`.
Replace it with World Bank WDI, IMF WEO and national debt-office data before
the scorecard is used for anything real.

### Synthetic (drives the backtest)

The monthly panel 2015-01 to 2026-08 is **generated, not historical**. It is
calibrated so its statistical properties match published estimates: country
yield volatilities, the timing of the 2021–22 inflation surge and 2023–24
disinflation, cross-country return correlations, and an embedded
signal-to-return information coefficient of 0.045. Each country's terminal
CPI is anchored to its real Aug-2026 print.

**Therefore: the backtest statistics validate the pipeline, not the
strategy.** They demonstrate that the code computes IC, attribution, turnover
and beta correctly on data with known properties. They are not evidence that
this strategy makes money. Swapping `load_panel()` for a real data loader is
the only change needed; nothing downstream moves.

## 6. Results

### Headline (synthetic panel, 140 months)

| | Parity gross | Parity net | Threshold net | Sequential ML |
|---|---|---|---|---|
| Ann. return % | 6.03 | 5.04 | 4.45 | 3.08 |
| Ann. vol % | 10.0 | 10.0 | 10.0 | 10.0 |
| **Sharpe** | **0.60** | **0.50** | **0.45** | **0.31** |
| Sortino | 1.16 | 0.96 | 0.83 | 0.49 |
| Max drawdown % | −17.4 | −18.2 | −16.8 | −22.4 |
| Hit rate | 0.56 | 0.54 | 0.54 | 0.30 |

Monthly IC 0.034, t-stat 1.66. Cost model charges 4bp of risk capital per
unit of gross turnover, which is a placeholder — public EM bid-offer data is
poor, and this is the least defensible number in the project.

### Market neutrality

Beta to the equal-weighted EM basket is **0.062**, R² **0.001**. The book is
not a disguised long-duration position. This is the diagnostic that should be
run first on any relative-value claim, and the one most often skipped.

### Factor attribution

| Factor | Standalone Sharpe | Mean IC | IC t-stat | Ann. turnover |
|---|---|---|---|---|
| FX valuation | 0.33 | +0.026 | 1.26 | 6.8× |
| Terms of trade | 0.29 | +0.030 | 1.33 | 6.6× |
| Real yield & carry | 0.26 | −0.001 | −0.06 | 0.6× |
| Fiscal thrust | 0.24 | +0.013 | 0.56 | 2.7× |
| Credit shortfall | 0.23 | −0.005 | −0.23 | 1.1× |
| **Inflation pressure** | **−0.04** | **−0.014** | **−0.65** | **8.5×** |

## 7. Three findings worth defending in an interview

**The inflation factor is the weakest of the six, and the most expensive.**
It has the worst standalone Sharpe and by far the highest turnover — 8.5×
annually against 0.6× for real carry. That result is not an artefact: it
matches published evidence that relative inflation metrics are less precise
predictors of market effects than directional ones, because countries differ
in how inflation maps into policy. An inflation-only three-factor model would
have been built on its weakest leg. Adding terms of trade and FX valuation is
what makes the book work.

**Machine learning does not beat equal weighting here.** The sequentially
fitted, sign-constrained, parity-shrunk ridge produces Sharpe 0.31 against
0.60 for conceptual parity. Diagnostically, the *terminal* learned weights
applied statically give IC 0.041 versus 0.034 for equal weights — so the
weights are informative. The damage comes entirely from early-sample
instability, when too few macro regimes have been observed to estimate
anything stable. This is the steep bias-variance trade-off of macro panels,
and it reproduces the published finding that ML failed to outperform
conceptual parity in an analogous FX application. Shrinking toward parity
improved IC from −0.002 to +0.018 but did not close the gap.

**Two factors are substantially redundant.** Real carry and credit shortfall
correlate at −0.57, by far the highest pair. They are both picking up
monetary stance. A disciplined next version would merge them or drop one,
which is precisely what regularisation does to twelve-factor frameworks.

## 8. Known limitations

1. **The backtest is on synthetic data.** Stated three times because it is
   the thing most likely to be skimmed past.
2. **Structural data is estimated,** not sourced. Trade openness, fiscal
   balances and foreign ownership need real inputs.
3. **Surprises are modelled, not measured.** A proper implementation fits an
   ARMA(1,1) on expanding-window monthly CPI increments per country, using
   only pre-release information. Benchmarking against consensus is the
   alternative, but EM analyst coverage is thin and uneven, so surprise may
   partly measure coverage quality rather than inflation news.
4. **Transaction costs are a guess.** The threshold variant is the honest
   response: it cuts Sharpe from 0.50 to 0.45 while materially reducing
   turnover, and that trade-off is the real question for capacity.
5. **No curve dimension.** Every position is outright 5y duration. Expressing
   signals as steepeners would isolate the policy view and remove residual
   global duration beta.
6. **Revision handling is not implemented.** Real point-in-time work must
   treat revisions on non-release dates as a separate surprise event class.

## 9. Repository layout

```
data.py       real snapshot + calibrated panel generator + blacklist
factors.py    six factors, R1-R6 methodology rules
backtest.py   positions, PnL, stats, IC, attribution, beta decomposition
learn.py      sequential sign-constrained ridge with parity shrinkage
run.py        pipeline; writes charts and tables to ./output
```

```bash
python run.py
```

Outputs: `scorecard.png`, `equity_curves.png`, `ml_weights.png`,
`scorecard.csv`, `factor_attribution.csv`, `country_attribution.csv`,
`factor_correlation.csv`, `summary.json`.

## 10. Moving to real data

Replace `data.load_panel()` with a loader returning the same long-format
schema. Required series per country per month, all point-in-time:

```
cpi_yoy, cpi_3m3m, cpi_6m6m, cpi_surprise, target,
yld_5y, real_yld, carry, reer_chg, tot_chg,
fiscal_bal, fiscal_chg, credit_gap, foreign_own, ret_fwd_1m, tradable
```

Free-ish sources: IMF IFS and WEO (CPI, fiscal, external), World Bank WDI
(openness, energy dependence), BIS (policy rates, credit-to-GDP gaps, REER),
national debt offices (foreign ownership), UN Comtrade (export baskets).
The genuinely hard input without a terminal is the local yield curve history,
and that constraint will likely determine universe size more than anything
else.

**The single most important thing to get right is aligning every macro series
on its release timestamp rather than its reference month.** Doing it wrong
produces a beautiful backtest and a worthless one.
