"""
bookdepth_features.py — L2 order book imbalance из binance vision bookDepth -> 1h фичи.

bookDepth (futures/um/daily, с 2023): снапшоты ~каждые 30с, уровни ±1..5% от mid,
колонки [timestamp, percentage, depth, notional]. percentage<0 = bid, >0 = ask.

Считаем имбаланс стакана (микроструктура, которой НЕТ в свечах):
  imb_top = (bidN_1 - askN_1)/(bidN_1 + askN_1)      — у касания (±1%)
  imb_all = (sumBidN - sumAskN)/(sumBidN + sumAskN)  — вся глубина ±5%
Агрегируем к 1h: mean(imb_top), mean(imb_all), std(imb_all)=книжная волатильность,
mean(total_notional)=ликвидность. Качаем по дню (мало памяти), дописываем.

    python bookdepth_features.py --symbol BTCUSDT --start 2023-01-01 --end 2025-09-30 \
        --out /abs/data/raw/BTC_bookdepth_1h.csv
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth/{sym}/{sym}-bookDepth-{d}.zip"


def day_range(s, e):
    d0 = datetime.strptime(s, "%Y-%m-%d"); d1 = datetime.strptime(e, "%Y-%m-%d")
    cur, out = d0, []
    while cur <= d1:
        out.append(cur.strftime("%Y-%m-%d")); cur += timedelta(days=1)
    return out


def fetch(sym, d, retries=4):
    url = BASE.format(sym=sym, d=d)
    for i in range(retries):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 404:
                return b""
            r.raise_for_status(); return r.content
        except Exception as e:
            if i == retries - 1:
                print(f"    {d} ОШИБКА: {e}"); return b""
            time.sleep(2 * (i + 1))
    return b""


def day_to_hourly(zbytes):
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        names = zf.namelist()
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as f:
            df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
    bid = df[df["percentage"] < 0]; ask = df[df["percentage"] > 0]
    # суммы по снапшоту (timestamp)
    gb_b = bid.groupby("timestamp")["notional"].sum()
    gb_a = ask.groupby("timestamp")["notional"].sum()
    n1_b = bid[bid["percentage"] == -1].set_index("timestamp")["notional"]
    n1_a = ask[ask["percentage"] == 1].set_index("timestamp")["notional"]
    snap = pd.DataFrame({"bidN": gb_b, "askN": gb_a, "bN1": n1_b, "aN1": n1_a}).dropna()
    snap["imb_all"] = (snap["bidN"] - snap["askN"]) / (snap["bidN"] + snap["askN"])
    snap["imb_top"] = (snap["bN1"] - snap["aN1"]) / (snap["bN1"] + snap["aN1"])
    snap["tot_notional"] = snap["bidN"] + snap["askN"]
    # агрегат к 1h
    h = snap.resample("1h").agg(
        ob_imb_all=("imb_all", "mean"), ob_imb_top=("imb_top", "mean"),
        ob_imb_std=("imb_all", "std"), ob_liq=("tot_notional", "mean"))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        outp.unlink()
    days = day_range(args.start, args.end)
    print(f"{args.symbol} bookDepth: {days[0]}..{days[-1]} ({len(days)} дней)")
    wrote = False
    n404 = 0
    for d in days:
        z = fetch(args.symbol, d)
        if not z:
            n404 += 1; continue
        h = day_to_hourly(z)
        if h.empty:
            continue
        h.to_csv(outp, mode="a", header=not wrote)
        wrote = True
        if days.index(d) % 60 == 0:
            print(f"  {d}: {len(h)} часов | пропущено {n404}", flush=True)
        time.sleep(0.1)
    print(f"[OK] -> {outp} (пропущено дней: {n404})")


if __name__ == "__main__":
    main()
