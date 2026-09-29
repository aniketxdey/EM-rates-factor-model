# Handoff: what you need to do before trading real money

Everything in the repo runs today on free public data, and `live.py`
produces a real trade list from it. The items below are the gaps between
"works on public data" and "safe to run capital through". They are ordered
by how much they can change the answer.

Current state as of 29-Sep-2026:

* **Backtest:** net Sharpe 0.63 (t = 2.5), beta 0.03 to the EM basket, OOS
  window Feb-2011 to Jul-2026.
* **Costs:** the strategy breaks even at about 2.9× the assumed bid-offer.
* **Factor mix:** about 95% of PnL comes from real yield & carry.

---

## 1. Licensed rates data (blocker for production)

**Why.** The model prices on TradingView `TVC:{cc}05Y` government benchmark
yields, pulled from an unauthenticated websocket. The data is real and
cross-checks against OECD/FRED at a 0.93–0.998 correlation of changes. But:

* TradingView's terms don't permit automated retrieval for commercial use;
* the feed has gaps and benchmark switches (see COP and PEN);
* **you will trade swaps, not these bonds.**

**What to get (any one of these):**

* **Bloomberg** (BLPAPI or Data License), 5y swap tickers. Confirm each with
  the desk. PLN `PZSW5`, CZK `CKSW5`, HUF `HFSW5`, ZAR `SASW5`, MXN `MPSW5`,
  KRW `KWSWO5`, INR `IRSWNI5`, THB `TBSWO5`, MYR `MRSWQ5`, CLP `CHSWP5`,
  COP `CLSWIB5`, BRL `ODF31`/DI1 generics, IDR `IHSW5`/INDOGB, PHP `PPSWN5`,
  TRY `TYSW5`. You'll also need the generic govt curves for the QA
  cross-check.
* **LSEG/Refinitiv** DataScope: the same RICs TradingView re-publishes, plus
  swap curves.
* **JPMaQS (Macrosynergy):** point-in-time EM rates returns and macro
  indicators, built for exactly this model.

**How to plug in.** Write a `pipeline/sources.py` function that returns the
same long frame as `tvc_yields()` (`ccy, tenor, date, value`), and point
`fetch_all.fetch_yields()` at it. Nothing downstream changes. Re-run
`python run.py` and compare against the numbers above. **Expect the Sharpe to
move**: swap spreads and swap floating indices are not in the current
backtest.

**Free partial fixes** if you want to extend history before buying data:

* **MXN pre-Dec-2024:** register for a Banxico SIE API token (free, instant,
  <https://www.banxico.org.mx/SieAPIRest/service/v1/token>). Series `SF43886`
  is the M-bono 5y *auction* yield only; daily secondary yields need
  Valmer/PiP or Bloomberg.
* **BRL pre-2020:** B3 publishes daily DI1 settlement prices (free historical
  files at <https://www.b3.com.br>, "Market Data > Historical Data"). A
  loader would convert the ~5y contract settle to a yield.
* **COP:** Banco de la República publishes the TES zero-coupon curve daily
  (<https://www.banrep.gov.co/en/estadisticas/tes>).
* **PEN:** BCRP's statistics API (<https://estadisticas.bcrp.gob.pe>) needs no
  key. Find the 5y soberano yield series code there.

## 2. Executable transaction costs (the number most likely to be wrong)

**Why.** `config.INSTRUMENTS[*]["bid_offer_bp"]` is set at or above the top of
published ranges: ADB Asia Bond Monitor (Mar-2026), BIS Papers 67, and dealer
conventions. At 2× those costs net Sharpe drops to 0.28; at 3× it's −0.07.

**What to do.**

* Collect 2–4 weeks of **executable dealer runs** for a 5y clip at your
  intended size in each of the 17 markets. Record the bid-offer in bp and the
  maximum size quoted at that spread.
* Put the median into `bid_offer_bp` and re-run `run.py`.
* After go-live, record fills against mid, and do this TCA monthly.
* **Clip sizes:** at $10mm of risk capital the book trades up to ~$7k DV01
  per market (THB and MYR are the largest legs). That's normal clip size in
  THB and MYR, but check TRY, PHP, RON, PEN and IDR liquidity.

## 3. Market access and documentation (blocker)

This isn't something code can do. The instrument per market is in
`config.INSTRUMENTS`.

* **ISDA/CSA** with at least two dealers covering the NDIRS/NDS markets:
  KRW, INR, MYR, IDR, PHP, CLP, COP, BRL (offshore).
* **Clearing:** confirm LCH SwapClear/CME eligibility with your FCM for PLN,
  CZK, HUF, ZAR, MXN, THB, INR, KRW, CLP, COP and BRL. Uncleared trades need
  an initial-margin (UMR) assessment.
* **Futures:** B3 access for DI1 (BRL), direct or through an FCM.
* **Onshore bonds** (IDR, PHP, RON, PEN, TRY fallback): custody, tax
  registration and FX hedging lines. Withholding tax isn't modelled (e.g.
  IndoGB coupons).
* **Floating-index transitions:** confirm with dealers which index each
  "5y IRS" references today. WIBOR→WIRON (PLN), JIBAR→ZARONIA (ZAR) and
  TIIE 28d→TIIE de Fondeo (MXN) all change carry versus the policy-rate
  funding assumed in the backtest.

## 4. Point-in-time macro vintages (improves backtest honesty)

**Why.** Macro inputs use the latest revised vintage. Fiscal is the most
exposed, because WEO outturns get revised.

**What to do.**

* Download historical WEO vintages in a browser. `imf.org` returns
  403 "Access Denied" to scripts. The files are at
  <https://www.imf.org/en/Publications/WEO/weo-database>: pick each April and
  October edition from 2010 onward and choose "Entire dataset, by countries".
* Put them in `data/manual/weo_vintages/`. I can then add a loader that picks
  the vintage in force at each cutoff; it's a small change to
  `panel.build_panel()`.
* Optional: JPMaQS already provides point-in-time versions of most inputs.

## 5. Operations

* **Schedule.** Run on the last business day of each month, after local
  closes, and execute at the next session. Example `crontab` (UTC, weekdays
  at 22:30; `--month-end-only` exits unless today is the last business day):
  `30 22 26-31 * 1-5 cd /path/rates-factor-model && python3 live.py --capital <USD> --refresh --month-end-only --holdings positions/current.csv`
* **Holdings file.** After execution, write actual DV01 per market (USD per
  1bp fall in the 5y rate, positive = receive) to `positions/current.csv`
  with columns `ccy,dv01_usd`. `live.py` writes a suggested
  `holdings_after_<date>.csv`; reconcile it against dealer confirms before
  overwriting. Without a holdings file the book is treated as flat, and the
  first month trades 50% of target by design.
* **Hard stops.** `live.py` exits with code 2 if any pre-trade check fails
  (fewer than 6 markets, stale prints, vol or leverage limits, ridge weights
  not available). Don't override it.
* **Monitoring.** Alert if `python -m pipeline.fetch_all` returns non-zero.
  Also watch for TradingView symbol errors and for a manifest `retrieved_utc`
  older than 2 days.

## 6. Risk sign-off (your decision)

* The historical max drawdown is −18% and the worst month is −13%, at a 10%
  vol target. Worst episodes: Sep-2008, Aug-2018 (TRY) and Mar-2022 (MYR).
  Decide on a drawdown stop and a capital allocation; neither is modelled.
* **Concentration.** Real yield & carry is 55% of the weights and about 95%
  of PnL. The book is effectively risk-adjusted EM carry with macro tilts,
  and will suffer in broad EM carry unwinds even though its basket beta is
  near zero.
* The model has **no discretionary overlay** for elections, capital controls
  or sanctions. The data rules exclude stale or broken markets, but they
  don't anticipate events.

## 7. Nice-to-have extensions

* A timelier terms-of-trade factor: rebuild CTOT from daily commodity prices
  (IMF PCPS or World Bank Pink Sheet) using IMF CTOT weights, which would
  remove the 4-month lag.
* Roll-down, using the 2y/10y TVC tenors or swap curves once licensed.
* CPI surprise versus consensus. This needs a consensus source: Bloomberg
  ECO, or the Trading Economics API (paid key).
