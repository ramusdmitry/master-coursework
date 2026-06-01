"""
demo/feed.py — источники баров для демо. Асинхронные генераторы (ts, close).

- replay_feed: проигрывает историю из CSV ускоренно (для теста/демо без ожидания).
- live_feed: подключается к Binance kline WebSocket (через aiohttp), отдаёт бар на
  каждое ЗАКРЫТИЕ свечи. НЕ торгует — только читает рынок.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

BINANCE_WS = "wss://stream.binance.com:9443/ws/{sym}@kline_{tf}"


_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h"}


async def replay_feed(csv_1m: str, tf: str = "1h", since: str | None = None,
                      delay: float = 0.3, limit: int | None = None):
    """Проигрывает бары OHLCV из {asset}_1m.csv (ресемпл в tf). Отдаёт (ts, bar-dict)."""
    import build_processed_data as bp
    df = bp._read_single_path(Path(csv_1m))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    d = df.resample(_FREQ.get(tf, "1h")).agg(agg).dropna(subset=["close"])
    if since:
        d = d[d.index >= since]
    if limit:
        d = d.iloc[-limit:]
    for ts, row in d.iterrows():
        yield ts.to_pydatetime(), {"o": float(row["open"]), "h": float(row["high"]),
                                   "l": float(row["low"]), "c": float(row["close"]),
                                   "v": float(row["volume"])}
        await asyncio.sleep(delay)


async def live_feed(symbol: str = "btcusdt", tf: str = "1m"):
    """Binance kline WS: отдаёт (ts, bar-dict) при закрытии свечи (k['x']==True)."""
    import aiohttp
    url = BINANCE_WS.format(sym=symbol.lower(), tf=tf)
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(url, heartbeat=30) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                k = json.loads(msg.data).get("k", {})
                if k.get("x"):   # свеча закрылась
                    ts = pd.to_datetime(k["T"], unit="ms", utc=True).to_pydatetime()
                    yield ts, {"o": float(k["o"]), "h": float(k["h"]), "l": float(k["l"]),
                               "c": float(k["c"]), "v": float(k["v"])}


def _bar(row):
    return {"o": float(row["open"]), "h": float(row["high"]), "l": float(row["low"]),
            "c": float(row["close"]), "v": float(row["volume"])}


async def replay_feed_multi(sym_csv: dict, tf: str = "1m", since: str | None = None,
                            delay: float = 0.25):
    """Выровненный по времени replay нескольких пар. Отдаёт (ts, sym, bar)."""
    import build_processed_data as bp
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    series = {}
    for sym, csv in sym_csv.items():
        d = bp._read_single_path(Path(csv)).resample(_FREQ.get(tf, "1h")).agg(agg).dropna(subset=["close"])
        if since:
            d = d[d.index >= since]
        series[sym] = d
    idx = sorted(set().union(*[s.index for s in series.values()]))
    for ts in idx:
        for sym, d in series.items():
            if ts in d.index:
                yield ts.to_pydatetime(), sym, _bar(d.loc[ts])
        await asyncio.sleep(delay)


async def live_feed_multi(symbols: list[str], tf: str = "1m"):
    """Combined Binance WS по нескольким парам. Отдаёт (ts, sym, bar) на закрытие свечи."""
    import aiohttp
    streams = "/".join(f"{s.lower()}@kline_{tf}" for s in symbols)
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(url, heartbeat=30) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                k = m.get("data", {}).get("k", {})
                if k.get("x"):
                    sym = k["s"].lower()
                    ts = pd.to_datetime(k["T"], unit="ms", utc=True).to_pydatetime()
                    yield ts, sym, {"o": float(k["o"]), "h": float(k["h"]), "l": float(k["l"]),
                                    "c": float(k["c"]), "v": float(k["v"])}
