"""
poltest_resample.py — сравнение пика RAM: polars (streaming) vs pandas
на ресемпле сырого 1s -> 5s/15s/30s. Запускать ОТДЕЛЬНЫМ процессом на движок
(ru_maxrss = high-water mark процесса).

    python poltest_resample.py --engine polars --path .../BTC_1s.csv
    python poltest_resample.py --engine pandas --path .../BTC_1s.csv
"""
from __future__ import annotations

import argparse
import resource
import time

COLS = ["open", "high", "low", "close", "volume", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote"]
TFS = ["5s", "15s", "30s"]


def ram_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run_polars(path):
    import polars as pl
    agg = [pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
           pl.col("close").last(), pl.col("volume").sum(), pl.col("quote_volume").sum(),
           pl.col("trades").sum(), pl.col("taker_buy_base").sum(),
           pl.col("taker_buy_quote").sum()]
    t0 = time.time()
    lf = pl.scan_csv(path).with_columns(
        pl.col("timestamp").str.to_datetime(
            format="%Y-%m-%d %H:%M:%S%z", time_unit="us").set_sorted())
    for tf in TFS:
        out = (lf.group_by_dynamic("timestamp", every=tf, closed="left")
               .agg(agg).collect(engine="streaming"))
        r = out.row(0)
        print(f"polars {tf}: {out.height:,} строк | пик RAM {ram_gb():.2f} ГБ | "
              f"t={time.time()-t0:.0f}s | первая [ts={r[0]} o={r[1]} c={r[4]} v={r[5]:.4f}]",
              flush=True)


def run_pandas(path):
    import pandas as pd
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "quote_volume": "sum", "trades": "sum",
           "taker_buy_base": "sum", "taker_buy_quote": "sum"}
    t0 = time.time()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    print(f"pandas read: {len(df):,} строк | пик RAM {ram_gb():.2f} ГБ | "
          f"t={time.time()-t0:.0f}s", flush=True)
    for tf in TFS:
        out = df.resample(tf).agg(agg).dropna(subset=["close"])
        r = out.iloc[0]
        print(f"pandas {tf}: {len(out):,} строк | пик RAM {ram_gb():.2f} ГБ | "
              f"t={time.time()-t0:.0f}s | первая [ts={out.index[0]} o={r['open']} "
              f"c={r['close']} v={r['volume']:.4f}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["polars", "pandas"])
    ap.add_argument("--path", required=True)
    args = ap.parse_args()
    (run_polars if args.engine == "polars" else run_pandas)(args.path)
    print(f"=== {args.engine}: ИТОГОВЫЙ пик RAM {ram_gb():.2f} ГБ ===", flush=True)


if __name__ == "__main__":
    main()
