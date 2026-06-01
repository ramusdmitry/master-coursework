"""
sweep_all.py — рычаги B (maker-комиссия) + A (min_holding) ко ВСЕМУ зоопарку CP5+CP6.

Обучает зоопарк CP5 (SimpleLSTM, 1D CNN, Transformer, RandomForest, Ensemble) на
честных данных и для КАЖДОЙ модели делает sweep net Sharpe (1-бар) по
cost_bps ∈ {7,1,0} × min_holding ∈ {15,60,240}. Buy&Hold для сравнения.

Отвечает: какие модели (включая 1D CNN) и при каких издержках/удержании обгоняют BH.
DL на GPU. ~20 мин.

    python sweep_all.py --data-dir /abs/data/processed
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from src.common import (PER_YEAR_1M, SimpleLSTM, WindowDataset, apply_min_holding,
                        sharpe_ratio, set_seed, _predict_labels_probs)
from src.zoo import TransformerClassifier
from src.train import _prepare_data
from benchmark_full_test import _build_cfg
from cp5_fixed import Conv1dClassifier, aggregate_windows, train_dl, DEVICE, WINDOW

COSTS = {"7bps": 7e-4, "1bps(maker)": 1e-4, "0bps": 0.0}
HOLDS = [15, 60, 240]


def net_sharpe(pos, r1, cpt):
    gross = pos * r1
    turn = pos.diff().abs().fillna(pos.abs())
    return sharpe_ratio(gross - turn * cpt, PER_YEAR_1M), int(turn.sum())


def sweep_rows(name, p1, r1, idx):
    n = min(len(p1), len(idx))
    base = pd.Series((np.asarray(p1)[:n] >= 0.5).astype(float), index=idx[:n])
    r1n = pd.Series(np.asarray(r1)[:n], index=idx[:n])
    rows = []
    for mh in HOLDS:
        pos = apply_min_holding(base, mh)
        rec = {"Модель": name, "min_hold": mh}
        ntr = None
        for ck, cv in COSTS.items():
            sh, ntr = net_sharpe(pos, r1n, cv)
            rec[ck] = round(sh, 3)
        rec["сделок"] = ntr
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    set_seed(42)
    data = _prepare_data(_build_cfg(args.data_dir))
    Xt, Xv, Xte = data["Xt_s"], data["Xv_s"], data["Xte_s"]
    yt, yv, yte = data["yt_tb"], data["yv_tb"], data["yte_tb"]
    r1, idx, nfeat = data["r1_te"], data["idx_te_w"], data["n_features"]

    ds_tr, ds_va, ds_te = WindowDataset(Xt, yt, WINDOW), WindowDataset(Xv, yv, WINDOW), WindowDataset(Xte, yte, WINDOW)
    ld_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=512)
    ld_te = DataLoader(ds_te, batch_size=512)
    ct = np.bincount(ds_tr.labels, minlength=2)
    wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
    yva_w = ds_va.labels
    print(f"device={DEVICE}, фич={nfeat}, train={len(Xt)}, test={len(idx)}")

    all_rows, p1_store = [], {}
    for name, ctor in [("SimpleLSTM", lambda: SimpleLSTM(nfeat, 64, 1, 0.2)),
                       ("1D CNN", lambda: Conv1dClassifier(nfeat, 64, 0.2)),
                       ("Transformer", lambda: TransformerClassifier(nfeat, hidden=64, num_layers=2, dropout=0.2))]:
        print("Обучение", name)
        torch.manual_seed(42)
        m = train_dl(ctor(), ld_tr, ld_va, yva_w, wt, epochs=args.epochs, tag=name)
        _, _, p1 = _predict_labels_probs(m, ld_te, DEVICE)
        p1_store[name] = p1
        all_rows += sweep_rows(name, p1, r1, idx)

    print("Обучение RandomForest")
    Xt_agg, yt_agg = aggregate_windows(Xt, yt, WINDOW)
    Xte_agg, _ = aggregate_windows(Xte, yte, WINDOW)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(Xt_agg, yt_agg)
    rf_p1 = rf.predict_proba(Xte_agg)[:, 1]
    p1_store["RandomForest"] = rf_p1
    all_rows += sweep_rows("RandomForest", rf_p1, r1, idx)

    n = min(len(v) for v in p1_store.values())
    ens = np.mean([v[:n] for v in p1_store.values()], axis=0)
    all_rows += sweep_rows("Ensemble", ens, r1, idx)

    # Buy & Hold
    bh = pd.Series(1.0, index=idx)
    r1n = pd.Series(np.asarray(r1), index=idx)
    sh0, _ = net_sharpe(bh, r1n, 0.0)
    all_rows.append({"Модель": "Buy & Hold", "min_hold": "-", "7bps": round(sh0, 3),
                     "1bps(maker)": round(sh0, 3), "0bps": round(sh0, 3), "сделок": 1})

    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 88)
    print("SWEEP ВСЕГО ЗООПАРКА: net Sharpe (1-бар), B (комиссия) × A (min_holding), тест 2025")
    print("=" * 88)
    print(df.to_string(index=False))
    df.to_csv("sweep_all_results.csv", index=False, encoding="utf-8")
    # лучшие конфигурации, обгоняющие BH 0.63
    print("\n--- Конфигурации с maker (1bps), обгоняющие BH 0.63 ---")
    beat = df[(df["1bps(maker)"] > sh0)][["Модель", "min_hold", "1bps(maker)"]]
    print(beat.to_string(index=False) if len(beat) else "нет")
    print(f"\nСохранено: sweep_all_results.csv (BH={sh0:.3f})")


if __name__ == "__main__":
    main()
