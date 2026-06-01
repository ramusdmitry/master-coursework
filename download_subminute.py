#!/usr/bin/env python3
"""
download_subminute.py — потоковая (stream) докачка субминутных данных.

Проблема прямого подхода: 1s за весь период = ~150M баров/актив ≈ 46 ГБ RAM, если
грузить всё в память перед сохранением (как binance_spot_downloader накапливает frames).

Решение: обрабатываем ПО ОДНОМУ месяцу. Скачали 1s-ZIP месяца → ресемпл сразу в
{1s,5s,15s,30s} → дописали в CSV на диск → выбросили месяц из памяти.
Пиковая RAM = один месяц 1s (~2.6M баров ≈ 0.3–0.5 ГБ), независимо от длины периода.
То есть в 28 ГБ влезает любой период; реальный лимит — диск и время, не RAM.

Сохраняет и order-flow поля (quote_volume, trades, taker_buy_base/quote).

Usage:
  # 4 недостающих актива за 2025 (5s/15s/30s ресемплятся из 1s):
  python download_subminute.py --symbols ETH,BNB,SOL,XRP --start 2025-01 --end 2025-09
  # дыра BTC:
  python download_subminute.py --symbols BTC --start 2025-02 --end 2025-03 --tfs 1s,5s,15s,30s --append
  # весь период (RAM не лимит, ~15 ГБ диска/актив на 1s):
  python download_subminute.py --symbols BTC --start 2021-01 --end 2025-09
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

VISION_BASE = "https://data.binance.vision"
TMPL = "data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{yyyy_mm}.zip"

OUT_COLS = ["timestamp", "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
       "volume": "sum", "quote_volume": "sum", "trades": "sum",
       "taker_buy_base": "sum", "taker_buy_quote": "sum"}
TF_FREQ = {"1s": None, "5s": "5s", "15s": "15s", "30s": "30s"}


def month_range(start, end):
    s = datetime.strptime(start, "%Y-%m"); e = datetime.strptime(end, "%Y-%m")
    cur, out = datetime(s.year, s.month, 1), []
    while cur <= e:
        out.append(cur.strftime("%Y-%m"))
        y, m = cur.year + (cur.month // 12), 1 if cur.month == 12 else cur.month + 1
        cur = datetime(y, m, 1)
    return out


def fetch(symbol, interval, mm, retries=4):
    url = f"{VISION_BASE}/{TMPL.format(symbol=symbol, interval=interval, yyyy_mm=mm)}"
    for i in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return b""
            r.raise_for_status()
            return r.content
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"    retry {i+1}/{retries} ({e})"); time.sleep(2 * (i + 1))
    return b""


def parse(zbytes):
    """1s-ZIP -> DataFrame с order-flow, индекс=timestamp(UTC)."""
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        names = zf.namelist()
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as f:
            df = pd.read_csv(f, header=None, dtype=str, on_bad_lines="skip")
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                  "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"][:df.shape[1]]
    ts = pd.to_numeric(df["open_time"], errors="coerce")
    med = ts.dropna().median()
    unit = "ns" if med > 1e17 else "us" if med > 1e14 else "ms" if med > 1e11 else "s"
    out = pd.DataFrame({"timestamp": pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")})
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "trades", "taker_buy_base", "taker_buy_quote"]:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else 0.0
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out[~out["timestamp"].duplicated(keep="first")]
    return out.set_index("timestamp")


def resample_tf(month_df, tf):
    if tf == "1s":
        return month_df.reset_index()[OUT_COLS]
    d = month_df.resample(TF_FREQ[tf]).agg(AGG).dropna(subset=["close"])
    return d.reset_index()[OUT_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--tfs", default="1s,5s,15s,30s")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--append", action="store_true",
                    help="дописывать в существующие CSV (для дыр), иначе перезаписать")
    args = ap.parse_args()

    raw = Path(args.raw_dir); raw.mkdir(parents=True, exist_ok=True)
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    bases = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    months = month_range(args.start, args.end)
    print(f"Активы={bases} TF={tfs} месяцы={months[0]}..{months[-1]} ({len(months)} шт.)")

    for base in bases:
        symbol = f"{base}USDT"
        paths = {tf: raw / f"{base}_{tf}.csv" for tf in tfs}
        wrote_header = {tf: (args.append and paths[tf].exists()) for tf in tfs}
        if not args.append:
            for tf in tfs:
                if paths[tf].exists():
                    paths[tf].unlink()
        total = {tf: 0 for tf in tfs}
        for mm in months:
            print(f"[{base} {mm}] скачиваю 1s ZIP…", flush=True)
            try:
                z = fetch(symbol, "1s", mm)
            except Exception as e:
                print(f"  ОШИБКА скачивания: {e}"); continue
            if not z:
                print("  -> 404 (нет архива)"); continue
            md = parse(z)
            if md.empty:
                print("  -> пусто"); continue
            for tf in tfs:
                rs = resample_tf(md, tf)
                rs.to_csv(paths[tf], mode="a", header=not wrote_header[tf], index=False)
                wrote_header[tf] = True
                total[tf] += len(rs)
            del md
            time.sleep(0.2)
        # При --append дописанные месяцы попадают в конец файла -> порядок ломается.
        # Пере-сортируем через POLARS (pandas read_csv+sort на больших файлах = OOM:
        # 86M строк = ~30 ГБ в pandas). polars scan->sort->sink в temp, затем rename.
        # ВНИМАНИЕ: для БОЛЬШИХ исторических расширений (годы) лучше качать в отдельную
        # папку без --append (месяцы уже по порядку) и склеить cat-ом — без сортировки.
        if args.append:
            import polars as _pl
            for tf in tfs:
                if not paths[tf].exists():
                    continue
                tmp = paths[tf].with_suffix(".sorted.tmp")
                (_pl.scan_csv(str(paths[tf]))
                    .unique(subset=["timestamp"], keep="first")
                    .sort("timestamp")
                    .sink_csv(str(tmp)))
                tmp.replace(paths[tf])
                print(f"  [sort polars] {base} {tf}")
        for tf in tfs:
            print(f"[OK] {base} {tf}: {total[tf]:,} баров -> {paths[tf]}")


if __name__ == "__main__":
    main()
