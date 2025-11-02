#!/usr/bin/env python3
"""
Binance Spot Archive Downloader (data.binance.vision)
- Picks BTC, ETH + top-3 liquid USDT pairs by 24h quote volume (excl. stablecoins).
- Downloads monthly OHLCV klines ZIPs for a given interval (default: 1h).
- Concatenates into clean CSVs per asset in data/raw/
Usage examples:
  python binance_spot_downloader.py --interval 1h --start 2024-01 --end 2025-10
  python binance_spot_downloader.py --symbols BTC,ETH,SOL,BNB,XRP --interval 1h --start 2024-01 --end 2025-10
"""
import argparse
import csv
import io
import sys
import time
import zipfile
from datetime import datetime
from typing import List, Tuple
from pathlib import Path

import requests
import pandas as pd

VISION_BASE = "https://data.binance.vision"
MONTHLY_PATH_TMPL = "data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{yyyy_mm}.zip"
SPOT_TICKER_24H = "https://api.binance.com/api/v3/ticker/24hr"

STABLES = {"USDT","BUSD","USDC","FDUSD","TUSD","DAI","EUR","TRY","BRL","BIDR","RUB","PLN","ARS","NGN","ZAR","HKD","AUD"}

def month_range(start_yyyy_mm: str, end_yyyy_mm: str):
    s = datetime.strptime(start_yyyy_mm, "%Y-%m")
    e = datetime.strptime(end_yyyy_mm, "%Y-%m")
    months = []
    cur = datetime(s.year, s.month, 1)
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))

        y = cur.year + (cur.month // 12)
        m = 1 if cur.month == 12 else cur.month + 1
        cur = datetime(y, m, 1)
    return months

def pick_top_usdt_symbols(k: int = 5, exclude_bases=None):
    """Return top-k USDT symbols by 24h quoteVolume (spot)."""
    exclude_bases = set(exclude_bases or [])
    r = requests.get(SPOT_TICKER_24H, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data:
        sym = d.get("symbol","")
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        # skip stables as base and delisted weirds
        if base in STABLES:
            continue
        try:
            qv = float(d.get("quoteVolume","0"))
        except:
            qv = 0.0
        rows.append((sym, base, qv))
    # sort by quote volume desc
    rows.sort(key=lambda x: x[2], reverse=True)
    out = []
    for sym, base, _ in rows:
        if base in exclude_bases:
            continue
        out.append(sym)
        if len(out) >= k:
            break
    return out

def ensure_dirs():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/tmp").mkdir(parents=True, exist_ok=True)

def fetch_month_zip(symbol: str, interval: str, yyyy_mm: str) -> bytes:
    rel = MONTHLY_PATH_TMPL.format(symbol=symbol, interval=interval, yyyy_mm=yyyy_mm)
    url = f"{VISION_BASE}/{rel}"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return b""
    resp.raise_for_status()
    return resp.content

def parse_month_zip(zbytes: bytes) -> pd.DataFrame:
    import io, zipfile
    import pandas as pd
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        names = zf.namelist()
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as f:
            # читаем текстом, чтобы не словить переполнения/наезды типов
            df = pd.read_csv(f, header=None, dtype=str, on_bad_lines="skip")
            # колонки CSV у Binance kline (12 штук)
            df.columns = [
                "open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","trades_count",
                "taker_buy_base","taker_buy_quote","ignore"
            ]

            ts = pd.to_numeric(df["open_time"], errors="coerce")

            # авто-детект единиц времени
            # ~2025 год: seconds~1e9, ms~1e12, us~1e15, ns~1e18
            med = ts.dropna().median()
            if med > 1e17:
                unit = "ns"
            elif med > 1e14:
                unit = "us"
            elif med > 1e11:
                unit = "ms"
            else:
                unit = "s"
                
            print(f"[parse] detected unit={unit}, median(open_time)={med:.3e}")

            # отбраковка невозможных дат
            ts_dt = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
            out = pd.DataFrame({
                "timestamp": ts_dt,
                "open":  pd.to_numeric(df["open"],  errors="coerce"),
                "high":  pd.to_numeric(df["high"],  errors="coerce"),
                "low":   pd.to_numeric(df["low"],   errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume":pd.to_numeric(df["volume"],errors="coerce"),
            })
            out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
            # иногда попадаются дубли меток времени — уберём
            out = out[~out["timestamp"].duplicated(keep="first")]
            return out


def concat_and_save(frames, out_csv: Path):
    import pandas as pd
    if not frames:
        print(f"[WARN] No data to save for {out_csv.name}", file=sys.stderr)
        return
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    
    # drop duplicates by timestamp
    df = df[~df["timestamp"].duplicated(keep="first")]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {len(df):,} rows -> {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Download Binance spot OHLCV archives (monthly klines)")
    parser.add_argument("--interval", default="1h", help="kline interval (1m, 5m, 1h, 1d, ...)")
    parser.add_argument("--start", required=True, help="start month YYYY-MM (inclusive)")
    parser.add_argument("--end", required=True, help="end month YYYY-MM (inclusive)")
    parser.add_argument("--symbols", default="", help="comma-separated bases, e.g. BTC,ETH,SOL,BNB,XRP; if empty, auto-pick top-3 plus BTC/ETH")
    parser.add_argument("--dry", action="store_true", help="only print planned downloads")
    args = parser.parse_args()

    ensure_dirs()

    if args.symbols.strip():
        bases = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        top = pick_top_usdt_symbols(k=6, exclude_bases=["BTC","ETH"])  # get a bit more, we'll trim
        # Convert to bases
        bases_auto = [s[:-4] for s in top]
        bases = ["BTC","ETH"] + bases_auto[:3]
        print("[Auto-picked]", bases)

    months = month_range(args.start, args.end)
    print("Months:", months)

    for base in bases:
        symbol = f"{base}USDT"
        frames = []
        for mm in months:
            rel = MONTHLY_PATH_TMPL.format(symbol=symbol, interval=args.interval, yyyy_mm=mm)
            url = f"{VISION_BASE}/{rel}"
            print(f"Fetch {url}")
            if args.dry:
                continue
            try:
                z = fetch_month_zip(symbol, args.interval, mm)
                if not z:
                    print(f"  -> 404 (no archive for {mm})")
                    continue
                df = parse_month_zip(z)
                if df is not None and not df.empty:
                    frames.append(df)
                    time.sleep(0.25)
            except Exception as e:
                print(f"  -> ERROR: {e}")
        if not args.dry:
            out_csv = Path(f"data/raw/{base}_{args.interval}.csv")
            concat_and_save(frames, out_csv)

if __name__ == "__main__":
    main()
