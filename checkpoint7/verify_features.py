"""
verify_features.py — сверка polars-фич с pandas build_features (шаг 1 порта).

Берёт ОДИН ресемпленный OHLCV (через polars), считает фичи двумя путями:
pandas bp.build_features (fallback, ta=None) и features_polars.add_features,
сверяет каждую колонку (np.allclose). Запуск на BTC 2025, tf 30s и 5s.

    python verify_features.py --path /abs/data/raw/BTC_1s.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_processed_data as bp
import features_polars as fp

CASES = [("30s", 2880), ("5s", 17280)]


def check(path, asset, freq, bd):
    rs = fp.resample_lazy(path, freq).collect(engine="streaming")          # polars OHLCV
    # --- pandas путь ---
    bp.DAY_N = bd
    pdf = rs.to_pandas().set_index("timestamp")
    all_df = pd.DataFrame(pdf.values, index=pd.to_datetime(pdf.index, utc=True),
                          columns=pd.MultiIndex.from_product([[asset], pdf.columns]))
    feat_pd = bp.build_features(all_df).dropna()
    # --- polars путь ---
    feat_pl = fp.add_features(rs, bd).to_pandas().set_index("timestamp")
    feat_pl.index = pd.to_datetime(feat_pl.index, utc=True)
    feat_pl = feat_pl.dropna(subset=fp.FEATURE_COLS)

    common = feat_pd.index.intersection(feat_pl.index)
    print(f"\n=== {asset} {freq} (bd={bd}) | pandas {len(feat_pd):,} / polars {len(feat_pl):,} "
          f"/ общих {len(common):,} ===")
    ok = True
    for col in fp.FEATURE_COLS:
        a = feat_pd.loc[common, f"{asset}__{col}"].to_numpy()
        b = feat_pl.loc[common, col].to_numpy()
        close = np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
        maxd = np.nanmax(np.abs(a - b)) if len(a) else 0.0
        flag = "OK " if close else "РАСХОЖДЕНИЕ"
        if not close:
            ok = False
        print(f"  {flag} {col:<12} max|Δ|={maxd:.2e}")
    print(f"  ИТОГ {freq}: {'ВСЕ СОВПАЛИ' if ok else 'ЕСТЬ РАСХОЖДЕНИЯ'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--asset", default="BTC")
    args = ap.parse_args()
    allok = all(check(args.path, args.asset, f, bd) for f, bd in CASES)
    print(f"\n{'='*50}\nПОРТ КОРРЕКТЕН' if allok else 'ПОРТ ТРЕБУЕТ ПРАВОК'\n{'='*50}" if False else
          f"\n{'='*50}\n{'ПОРТ КОРРЕКТЕН ✅' if allok else 'ПОРТ ТРЕБУЕТ ПРАВОК ❌'}\n{'='*50}")


if __name__ == "__main__":
    main()
