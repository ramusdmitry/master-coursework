"""
sub_assets.py — ФАКТИЧЕСКИЙ замер субминутных, ОДИН актив за вызов (memory-safe).

Раньше один процесс гонял все 5 активов подряд → RAM не освобождалась между
активами → OOM, WSL exit 1. Теперь: один --asset за запуск, на выходе процесс
умирает и освобождает ВСЮ память. Драйвер (shell-цикл) зовёт скрипт 5 раз и
дописывает строки в общий CSV. Пик RAM = один актив (BTC 1s в одиночку влезает).

Метод тот же, что давал числа BTC (full_spectrum.run_one): ratio-сплит 70/15/15,
ROC + Sharpe maker 1bps vs BH, аннуализация под tf. Плюс gc и лог RAM.

    python sub_assets.py --asset ETH --raw-dir /abs/data/raw --out sub_assets_results.csv
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import pandas as pd

try:
    import resource
    def _ram_gb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
except Exception:
    def _ram_gb():
        return float("nan")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_processed_data as bp
from full_spectrum import run_one, SUB
from src.common import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="sub_assets_results.csv")
    args = ap.parse_args()
    set_seed(42)
    p = Path(args.raw_dir) / f"{args.asset}_1s.csv"
    if not p.exists():
        print(f"[skip] нет {p}"); return
    print(f"=== {args.asset}: читаю 1s ({p.stat().st_size/1e9:.1f} ГБ) ===", flush=True)
    df1s = bp._read_single_path(p)
    rows = []
    for tf, (freq, bd) in SUB.items():
        r = run_one(df1s, args.asset, tf, freq, bd, "ratio")
        if r:
            rows.append(r)
            print(f"{args.asset} {tf}: ROC={r['ROC-AUC']} Sh maker={r['Sh maker']} "
                  f"BH={r['BH']} (test={r['test']}) RAM={_ram_gb():.1f}ГБ", flush=True)
        gc.collect()
    del df1s; gc.collect()
    if rows:
        df = pd.DataFrame(rows)
        outp = Path(args.out)
        df.to_csv(outp, mode="a", header=not outp.exists(), index=False, encoding="utf-8")
        print(f"[OK] {args.asset}: дописано {len(rows)} строк -> {outp} (пик RAM {_ram_gb():.1f}ГБ)",
              flush=True)


if __name__ == "__main__":
    main()
