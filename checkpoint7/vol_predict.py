"""
vol_predict.py — предсказуема ли реализованная волатильность 1h? (vs направление).

Реализованная вол за час = sqrt(sum 1m-доходностей^2). Предсказываем log-RV(t)
из лагов log-RV (вол кластеризуется). Out-of-sample R²/corr, тривиальный Ridge.

Вывод: волатильность ПРЕДСКАЗУЕМА (R² 0.58-0.73, corr ~0.8) у всех 5 — в отличие
от НАПРАВЛЕНИЯ (ROC ~0.51). Знак непредсказуем, амплитуда — да. Но бот зарабатывает
на знаке; vol-прогноз годен лишь как vol-targeting (надстройка на BH), не направл. edge.

    python vol_predict.py --raw-dir /abs/data/raw
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_processed_data as bp

ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()
    print(f"{'актив':<6}{'R2 logRV':>10}{'corr':>8}  (направление: ROC~0.51)")
    for a in ASSETS:
        d = bp._read_single_path(Path(args.raw_dir) / f"{a}_1m.csv")
        r1 = np.log(d["close"] / d["close"].shift(1))
        rv = np.sqrt((r1 ** 2).resample("1h").sum()).dropna()
        lrv = np.log(rv.replace(0, np.nan)).dropna()
        X = pd.concat({f"l{k}": lrv.shift(k) for k in range(1, 7)}, axis=1)
        df = pd.concat([X, lrv.rename("y")], axis=1).dropna()
        n = len(df); s = int(n * 0.7)
        m = Ridge().fit(df.iloc[:s, :-1], df.iloc[:s, -1])
        p = m.predict(df.iloc[s:, :-1])
        yte = df.iloc[s:, -1].to_numpy()
        print(f"{a:<6}{r2_score(yte, p):>10.3f}{np.corrcoef(yte, p)[0, 1]:>8.3f}")
    print("\nВол предсказуема (R²>0.5), направление — нет. Предсказуемая часть (амплитуда)")
    print("≠ та, что даёт прибыль (знак). Применение вола: vol-targeting, не направл. edge.")


if __name__ == "__main__":
    main()
