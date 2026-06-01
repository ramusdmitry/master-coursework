"""
fetch_onchain.py — загрузка BTC on-chain метрик (пункт 3в плана).

Источник: Blockchain.com Charts API (бесплатно, без ключа, годы истории).
Метрики дневные → сохраняются как CSV (date, metric...). Дальше встраиваются
в признаки forward-fill на 1m-грид (медленный контекст режима сети).

    python fetch_onchain.py --out data/onchain/btc_onchain_daily.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# метрика -> короткое имя признака
METRICS = {
    "n-transactions": "tx_count",                       # число транзакций/день
    "estimated-transaction-volume-usd": "tx_volume_usd",  # объём переводов USD (как в плане)
    "transaction-fees": "fees_btc",                     # суммарные комиссии (BTC)
    "hash-rate": "hash_rate",                            # хешрейт сети
    "n-unique-addresses": "active_addresses",            # активные адреса
    "miners-revenue": "miners_revenue_usd",              # доход майнеров USD
}

BASE = "https://api.blockchain.info/charts/{}?timespan={}&format=json&sampled=false"


def fetch_metric(chart: str, timespan: str) -> pd.Series:
    r = requests.get(BASE.format(chart, timespan), timeout=60)
    r.raise_for_status()
    vals = r.json().get("values", [])
    if not vals:
        return pd.Series(dtype=float)
    s = pd.Series(
        {pd.Timestamp(v["x"], unit="s", tz="UTC").normalize(): float(v["y"]) for v in vals}
    )
    return s[~s.index.duplicated(keep="first")].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/onchain/btc_onchain_daily.csv")
    ap.add_argument("--timespan", default="3years")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cols = {}
    for chart, name in METRICS.items():
        s = fetch_metric(chart, args.timespan)
        cols[name] = s
        print(f"{name:22s} ({chart}): {len(s)} дней "
              f"[{s.index.min().date()}..{s.index.max().date()}]" if len(s) else f"{name}: пусто")
        time.sleep(0.5)

    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df.to_csv(out)
    print(f"\nСохранено: {out.resolve()}  shape={df.shape}")
    print(df.tail(3))


if __name__ == "__main__":
    main()
