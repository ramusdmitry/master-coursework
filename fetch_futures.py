"""
fetch_futures.py — загрузка Binance USDⓈ-M futures метрик позиционирования (BTC).

Источник: data.binance.vision/data/futures/um. Бесплатно, исторически.
- metrics (daily, 5-мин): open interest, top-trader long/short ratio, taker l/s vol ratio;
- fundingRate (monthly, 8ч): ставка финансирования.

Сохраняет два CSV для последующего merge на 1m-грид (со сдвигом против look-ahead).

    python fetch_futures.py --start 2024-01 --end 2025-09 --out-dir data/futures
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"
SYM = "BTCUSDT"


_SESSION = requests.Session()


def _get_zip_csv(url, retries=4):
    """GET с ретраями и backoff (Binance vision иногда рвёт соединение)."""
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and len(r.content) >= 50:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                return pd.read_csv(z.open(z.namelist()[0]))
            return None
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_metrics(start_ym, end_ym, out_dir):
    d0 = date(int(start_ym[:4]), int(start_ym[5:7]), 1)
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    d1 = date(ey + (em // 12), 1 if em == 12 else em + 1, 1) - timedelta(days=1)
    frames, d, miss = [], d0, 0
    while d <= d1:
        url = f"{BASE}/daily/metrics/{SYM}/{SYM}-metrics-{d.isoformat()}.zip"
        df = _get_zip_csv(url)
        if df is not None:
            frames.append(df)
        else:
            miss += 1
        d += timedelta(days=1)
        time.sleep(0.15)
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_dir / "btc_futures_metrics.csv", index=False)
    print(f"metrics: {len(full)} строк (5-мин), пропущено дней: {miss}, "
          f"колонки: {list(full.columns)}")


def fetch_funding(start_ym, end_ym, out_dir):
    months = pd.period_range(start_ym, end_ym, freq="M").astype(str)
    frames = []
    for mm in months:
        url = f"{BASE}/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{mm}.zip"
        df = _get_zip_csv(url)
        if df is not None:
            frames.append(df)
        time.sleep(0.05)
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_dir / "btc_funding.csv", index=False)
    print(f"funding: {len(full)} строк (8ч), колонки: {list(full.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2025-09")
    ap.add_argument("--out-dir", default="data/futures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("=== metrics (daily 5-мин) ===")
    fetch_metrics(args.start, args.end, out)
    print("=== fundingRate (monthly 8ч) ===")
    fetch_funding(args.start, args.end, out)
    print("Готово:", out.resolve())


if __name__ == "__main__":
    main()
