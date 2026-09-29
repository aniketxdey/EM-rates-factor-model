"""
Minimal TradingView websocket client for daily bars of TVC government-bond
yield symbols (e.g. TVC:PL05Y).  Unauthenticated access; data is the
Refinitiv-sourced benchmark yield TradingView displays.  Up to 5000 daily
bars per request.
"""

from __future__ import annotations

import json
import random
import re
import string
import time

import pandas as pd
import websocket

WS_URL = "wss://data.tradingview.com/socket.io/websocket"
HEADERS = ["Origin: https://www.tradingview.com"]


def _sid(prefix: str) -> str:
    return prefix + "".join(random.choice(string.ascii_lowercase) for _ in range(12))


def _frame(func: str, params: list) -> str:
    body = json.dumps({"m": func, "p": params}, separators=(",", ":"))
    return f"~m~{len(body)}~m~{body}"


def _messages(raw: str) -> list[str]:
    out, i = [], 0
    while i < len(raw):
        m = re.match(r"~m~(\d+)~m~", raw[i:])
        if not m:
            break
        n = int(m.group(1))
        start = i + m.end()
        out.append(raw[start:start + n])
        i = start + n
    return out


def daily_bars(symbol: str, n_bars: int = 5000, timeout: float = 45.0) -> pd.DataFrame:
    """Return DataFrame[date, close] of daily closes for e.g. 'TVC:PL05Y'."""
    ws = websocket.create_connection(WS_URL, header=HEADERS, timeout=15)
    cs = _sid("cs_")
    try:
        ws.send(_frame("set_auth_token", ["unauthorized_user_token"]))
        ws.send(_frame("chart_create_session", [cs, ""]))
        ws.send(_frame("resolve_symbol",
                       [cs, "sym_1", "=" + json.dumps({"symbol": symbol, "adjustment": "splits"})]))
        ws.send(_frame("create_series", [cs, "s1", "s1", "sym_1", "1D", n_bars]))
        bars, t0 = [], time.time()
        while time.time() - t0 < timeout:
            raw = ws.recv()
            for msg in _messages(raw):
                if msg.startswith("~h~"):
                    ws.send(f"~m~{len(msg)}~m~{msg}")
                    continue
                try:
                    obj = json.loads(msg)
                except ValueError:
                    continue
                m = obj.get("m")
                if m in ("timescale_update", "du"):
                    ser = obj["p"][1].get("s1", {}).get("s", [])
                    bars.extend(b["v"] for b in ser)
                elif m == "symbol_error":
                    raise ValueError(f"TradingView symbol error: {symbol}")
                elif m == "series_completed":
                    return _to_frame(bars)
                elif m == "critical_error":
                    raise RuntimeError(f"TradingView critical error: {obj}")
        raise TimeoutError(f"TradingView timeout for {symbol}")
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _to_frame(bars: list) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame(bars).iloc[:, [0, 4]]
    df.columns = ["ts", "close"]
    # TVC stamps bars at the session start in UTC: 03:30/06:00 for Asia and
    # Europe (same calendar day) but 21:00 of the PREVIOUS day for markets on
    # a UTC+3 session clock (verified against KR 2025-06-03 and ZA 2025-04-28
    # holidays).  Shifting by +6h maps every stamp to its session date.
    ts = pd.to_datetime(df["ts"], unit="s") + pd.Timedelta(hours=6)
    df["date"] = ts.dt.normalize()
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    return df[["date", "close"]].reset_index(drop=True)


def daily_bars_retry(symbol: str, tries: int = 5) -> pd.DataFrame:
    last = None
    for i in range(tries):
        try:
            df = daily_bars(symbol)
            if len(df):
                return df
            last = RuntimeError(f"empty series {symbol}")
        except ValueError:
            raise
        except Exception as e:
            last = e
        time.sleep(1.5 * (i + 1))
    raise last
