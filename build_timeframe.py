"""
build_timeframe.py — пункт 2 плана: таймфреймы 15s/30s из 1s-данных.

Binance отдаёт klines только до 1s (5s/15s/30s — нет), поэтому 15s/30s получаем
ресемплом из BTC_1s.csv. Один актив BTC (кросс-активных фич нет). Формат выхода —
тот же, что у build_processed_data (BTC__* фичи, (BTC,y_bin/y_reg) таргеты), чтобы
eval_primary.py работал без изменений.

    python build_timeframe.py --csv data/raw/BTC_1s.csv --tf 30s --out-dir data/processed_30s
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import build_processed_data as bp

# баров в сутках по таймфрейму (для DAY_N в vol_roll)
DAY_BARS = {"1s": 86400, "5s": 17280, "15s": 5760, "30s": 2880, "1m": 1440,
            "15min": 96, "30min": 48, "1h": 24}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="30s")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--order-flow", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DAY_N под таймфрейм (vol_roll считается за сутки баров)
    bp.DAY_N = DAY_BARS.get(args.tf, 1440)

    print(f"=== Загрузка {args.csv} (1s) ===")
    df = bp._read_single_path(Path(args.csv))   # OHLCV(+order-flow), индекс UTC
    print(f"1s баров: {len(df):,}")

    print(f"=== Ресемпл 1s -> {args.tf} ===")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in bp.OF_COLS:
        if c in df.columns:
            agg[c] = "sum"
    df_tf = df.resample(args.tf).agg(agg).dropna(subset=["close"])
    print(f"{args.tf} баров: {len(df_tf):,}  [{df_tf.index.min()} .. {df_tf.index.max()}]")

    frames = {"BTC": df_tf}
    all_df = bp.build_all_df(frames)
    all_df_i = bp.interpolate(all_df)
    features = bp.build_features(all_df_i, order_flow=args.order_flow)
    targets = bp.build_targets(all_df_i, bp.HORIZON)
    print(f"features: {features.shape}, targets: {targets.shape}")

    dataset = pd.concat([features, targets], axis=1).dropna()
    print(f"после dropna: {len(dataset):,}")

    # сплит по тем же датам, что и 1m (train≤2024-08-31 …). Для 1s данных 2025
    # train будет пустым — поэтому для субминутных tf используем доли 70/15/15.
    n = len(dataset)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    Xc, yc = features.columns, targets.columns
    X_train, X_val, X_test = dataset[Xc].iloc[:i_tr], dataset[Xc].iloc[i_tr:i_va], dataset[Xc].iloc[i_va:]
    y_train, y_val, y_test = dataset[yc].iloc[:i_tr], dataset[yc].iloc[i_tr:i_va], dataset[yc].iloc[i_va:]
    print(f"Train {X_train.shape} Val {X_val.shape} Test {X_test.shape}")

    for name, obj in [("X_train", X_train), ("y_train", y_train), ("X_val", X_val),
                      ("y_val", y_val), ("X_test", X_test), ("y_test", y_test)]:
        obj.to_parquet(out_dir / f"{name}.parquet")
    print("Сохранено в", out_dir.resolve())


if __name__ == "__main__":
    main()
