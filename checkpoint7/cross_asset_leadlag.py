"""
cross_asset_leadlag.py — ведёт ли BTC альты на 1h? (лаг-корреляция BTC_ret[t-k]->alt).

Вывод: НЕТ. k=0 (одновременно) высокий (0.61-0.84 — co-movement, не предсказание),
k>=1 ≈ 0 (-0.02..0.00) — exploitable лид-лага на 1h нет (рынок эффективен внутри часа).

    python cross_asset_leadlag.py --raw-dir /abs/data/raw
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_processed_data as bp

ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP"]


def hourly_ret(raw, a):
    d = bp._read_single_path(Path(raw) / f"{a}_1m.csv")
    c = d["close"].resample("1h").last().dropna()
    return np.log(c / c.shift(1)).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()
    rets = {a: hourly_ret(args.raw_dir, a) for a in ASSETS}
    btc = rets["BTC"]
    print("Лаг-корреляция BTC_ret[t-k] -> ALT_ret[t] (1h):")
    print(f"{'alt':<5}{'k=0':>9}{'k=1':>9}{'k=2':>9}{'k=3':>9}")
    for a in ASSETS[1:]:
        j = pd.concat({"btc": btc, "alt": rets[a]}, axis=1).dropna()
        row = [j["btc"].shift(k).corr(j["alt"]) for k in range(4)]
        print(f"{a:<5}" + "".join(f"{c:>9.3f}" for c in row))
    print("\nk=0 высокий = одновременное co-movement (не торгуемо). k>=1≈0 = лид-лага нет.")


if __name__ == "__main__":
    main()
