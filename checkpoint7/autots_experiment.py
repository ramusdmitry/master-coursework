"""
autots_experiment.py — пункт 1 плана куратора: «посмотреть, что даёт auto-ts».

Прогноз ЦЕНЫ BTC (close) на 1h-ряду: AutoTS против наивного baseline (random walk,
прогноз = последнее значение). Метрики RMSE/MAE на отложенном горизонте.

Запуск (venv с установленным autots):
    python autots_experiment.py --csv /abs/data/raw/BTC_1h.csv --horizon 168
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--horizon", type=int, default=168)  # 1 неделя 1h-баров
    ap.add_argument("--context", type=int, default=4000)  # хвост для скорости AutoTS
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    H = args.horizon
    train = df.iloc[-(args.context + H):-H].copy()
    test = df.iloc[-H:].copy()
    y_true = test["close"].to_numpy(float)
    print(f"Контекст train: {len(train)} баров, тест: {H} баров (1h)")

    # --- naive baseline (random walk): прогноз = последнее значение train ---
    last_val = float(train["close"].iloc[-1])
    y_naive = np.full(H, last_val)

    # --- AutoTS ---
    from autots import AutoTS
    long_df = pd.DataFrame({
        "timestamp": train["timestamp"].values,
        "value": train["close"].to_numpy(float),
        "series_id": "BTC_close",
    })
    model = AutoTS(forecast_length=H, frequency="infer", ensemble="simple",
                   model_list="superfast", max_generations=3, num_validations=2,
                   no_negatives=True, verbose=0)
    model = model.fit(long_df, date_col="timestamp", value_col="value", id_col="series_id")
    pred = model.predict()
    y_autots = np.asarray(pred.forecast.iloc[:, 0].to_numpy(float))[:H]
    if len(y_autots) < H:
        y_autots = np.concatenate([y_autots, np.full(H - len(y_autots), y_autots[-1])])

    print("\n" + "=" * 56)
    print("AUTO-TS vs NAIVE (прогноз цены BTC close, 1h):")
    print("=" * 56)
    print(f"{'модель':<16}{'RMSE':>12}{'MAE':>12}")
    print(f"{'naive (RW)':<16}{rmse(y_true, y_naive):>12.2f}{mae(y_true, y_naive):>12.2f}")
    print(f"{'AutoTS':<16}{rmse(y_true, y_autots):>12.2f}{mae(y_true, y_autots):>12.2f}")
    try:
        print(f"\nЛучшая модель AutoTS: {model.best_model_name}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
