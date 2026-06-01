"""
demo/server.py — FastAPI live-демо: сравнение стратегий на веб-странице (НЕ торгует).

Фоновый фид (replay истории ИЛИ live Binance) → Book.step → снимок → SSE push в
браузер. Стратегии печатают прогноз+PnL%/hit%/Sharpe, лидер сверху. Запуск:

    DEMO_MODE=replay DEMO_TF=1h ~/.envs/ds/bin/uvicorn demo.server:app --port 8000
    DEMO_MODE=live   DEMO_TF=1m ~/.envs/ds/bin/uvicorn demo.server:app --port 8000

Открыть http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from demo.engine import Book, default_strategies
from demo.model_strategy import LSTMStrategy
from demo import feed as feedmod

MODE = os.environ.get("DEMO_MODE", "replay")
TF = os.environ.get("DEMO_TF", "1m")   # модель обучена на 1m
SYMBOLS = [s.strip().lower() for s in os.environ.get(
    "DEMO_SYMBOLS", "btcusdt,ethusdt,bnbusdt,solusdt,xrpusdt").split(",") if s.strip()]
RAW = os.environ.get("DEMO_RAW", "/home/dmitriy/magistracy/master-coursework/data/raw")
SINCE = os.environ.get("DEMO_SINCE", "2025-09-25")
PER_YEAR = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760}.get(TF, 8760)
STATIC = Path(__file__).parent / "static"


def _csv(sym):   # btcusdt -> .../BTC_1m.csv
    return f"{RAW}/{sym[:-4].upper()}_1m.csv"


def _make_strategies(sym):
    strats = default_strategies()
    if sym == "btcusdt":          # LSTM обучена на BTC — только для неё
        try:
            strats.append(LSTMStrategy())
        except Exception as e:
            print(f"[demo] LSTM не загружен: {e}")
    return strats


subscribers: set[asyncio.Queue] = set()
books = {sym: Book(_make_strategies(sym), per_year=PER_YEAR) for sym in SYMBOLS}
history: dict[str, list[dict]] = {sym: [] for sym in SYMBOLS}   # снимки по парам


async def broadcast(snap: dict):
    sym = snap.get("symbol")
    if sym in history:
        history[sym].append(snap)
        if len(history[sym]) > 4000:
            del history[sym][:1000]
    for q in list(subscribers):
        try:
            q.put_nowait(snap)
        except asyncio.QueueFull:
            subscribers.discard(q)


async def seed_live_history(limit=1500):
    """REST-подкачка истории по каждой паре (прогрев фич LSTM), тихо, без push."""
    import aiohttp
    async with aiohttp.ClientSession() as s:
        for sym in SYMBOLS:
            url = (f"https://api.binance.com/api/v3/klines?symbol={sym.upper()}"
                   f"&interval={TF}&limit={limit}")
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                kl = await r.json()
            for k in kl[:-1]:
                books[sym].step({"o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                                 "c": float(k[4]), "v": float(k[5])})
    print(f"[demo] прогрето {len(SYMBOLS)} пар по ~{limit} баров")


async def run_feed():
    try:
        if MODE == "replay":
            gen = feedmod.replay_feed_multi({s: _csv(s) for s in SYMBOLS},
                                            tf=TF, since=SINCE, delay=0.2)
        else:
            await seed_live_history()
            gen = feedmod.live_feed_multi(SYMBOLS, tf=TF)
        async for ts, sym, bar in gen:
            if sym not in books:
                continue
            snap = books[sym].step(bar)
            snap.update({"ts": ts.isoformat(), "mode": MODE, "tf": TF, "symbol": sym})
            await broadcast(snap)
    except Exception as e:
        await broadcast({"error": str(e)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_feed())
    yield
    task.cancel()


app = FastAPI(title="Live Forecast Demo", lifespan=lifespan)


@app.get("/stream")
async def stream():
    q: asyncio.Queue = asyncio.Queue(maxsize=4000)
    # отдать новому клиенту накопленную историю по всем парам, затем живой поток
    for sym in SYMBOLS:
        for s in history[sym][-200:]:
            q.put_nowait(s)
    subscribers.add(q)

    async def gen():
        try:
            while True:
                snap = await q.get()
                yield f"data: {json.dumps(snap)}\n\n"
        finally:
            subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
