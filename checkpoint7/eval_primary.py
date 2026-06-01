"""
eval_primary.py — обучает primary SimpleLSTM на одном наборе данных и выводит
метрики ПРОГНОЗА на полном тесте (ROC-AUC, accuracy) + торговую (net-Sharpe).

Назначение: сравнить baseline (15 OHLCV-фич) и order-flow (15 + order-flow)
по качеству прогноза направления. Запускается по разу на каждый набор:

    python eval_primary.py --data-dir /abs/data/processed     --tag baseline
    python eval_primary.py --data-dir /abs/data/processed_of  --tag order_flow

Обучение без теста в памяти (ленивый WindowDataset), тест — чанками. Seed фикс.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from src.common import (
    PER_YEAR_FWD,
    SimpleLSTM,
    WindowDataset,
    apply_min_holding,
    evaluate_on_forward,
    fit_classifier_sharpe,
    set_seed,
    _predict_labels_probs,
)
from src.train import _prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg(data_dir, seed=42, epochs=30):
    return OmegaConf.create({
        "seed": seed,
        "data": {"active_asset": "BTC", "data_dir": data_dir, "synthetic": False,
                 "splits": {"X_train": "X_train.parquet", "X_val": "X_val.parquet",
                            "X_test": "X_test.parquet", "y_train": "y_train.parquet",
                            "y_val": "y_val.parquet", "y_test": "y_test.parquet"}},
        "model": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "lr": 1e-3,
                  "batch_size": 256, "epochs": epochs, "patience": 5, "window": 60,
                  "horizon": 15, "min_holding": 15, "cost_bps": 7},
    })


@torch.no_grad()
def _infer_p1_chunked(model, Xte_s, window, chunk=80000):
    model.eval()
    N = len(Xte_s)
    n = N - window
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ds = WindowDataset(Xte_s[s:window + e], np.zeros(window + (e - s), dtype=np.int64), window)
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        _, _, p1 = _predict_labels_probs(model, loader, DEVICE)
        out[s:e] = p1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    set_seed(42)
    cfg = _cfg(args.data_dir, epochs=args.epochs)
    data = _prepare_data(cfg)
    window = cfg.model.window
    nfeat = data["n_features"]
    print(f"[{args.tag}] фич: {nfeat}, train={len(data['Xt_s'])}, test_окон={len(data['idx_te_w'])}")

    # обучение primary SimpleLSTM (early stopping по val net-Sharpe, как в проекте)
    ds_tr = WindowDataset(data["Xt_s"], data["yt_tb"], window)
    ds_va = WindowDataset(data["Xv_s"], data["yv_tb"], window)
    ct = np.bincount(ds_tr.labels, minlength=2)
    cw = ct.sum() / (2.0 * np.maximum(ct, 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))
    ld_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=512, shuffle=False)
    print(f"[{args.tag}] устройство обучения: {DEVICE}")

    torch.manual_seed(42)
    model = SimpleLSTM(input_size=nfeat, hidden_size=64, num_layers=1, dropout=0.2).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    fit_classifier_sharpe(model, ld_tr, ld_va, crit, opt,
                          fwd_val=data["fwd_va"], idx_val_w=data["idx_va_w"],
                          epochs=args.epochs, patience=5, tag=f"{args.tag}-primary",
                          periods_per_year=PER_YEAR_FWD, min_hold=15,
                          device=DEVICE)

    # тест чанками
    p1 = _infer_p1_chunked(model, data["Xte_s"], window)
    n = len(p1)
    yte = data["yte_tb"][window: window + n]
    idx = data["idx_te_w"][:n]

    roc = roc_auc_score(yte, p1) if len(np.unique(yte)) > 1 else float("nan")
    acc = accuracy_score(yte, (p1 >= 0.5).astype(int))
    pos = apply_min_holding(pd.Series((p1 >= 0.5).astype(float), index=idx), 15)
    ev = evaluate_on_forward(pos, data["fwd_te"], name=args.tag, prob=p1, true_label=yte)

    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТ [{args.tag}] на полном тесте 2025:")
    print(f"  ROC-AUC:        {roc:.4f}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Sharpe net:     {ev.get('Sharpe net')}")
    print(f"  Hit-rate входов:{ev.get('Hit-rate входов')}")
    print(f"  % в long:       {ev.get('% времени в long')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
