"""
edge_horizon.py — рычаг C: повысить edge на сделку увеличением горизонта.

Гипотеза: на коротком горизонте (15m) движение за сделку < издержек. На большем
горизонте (1h=60, 4h=240) движение крупнее → edge на сделку может покрыть комиссию.

Для H ∈ {15, 60, 240}: обучаем SimpleLSTM на triple-barrier с вертикальным барьером H,
держим H баров, оцениваем net Sharpe (1-бар) при taker(7bps) и maker(1bps) + edge на
сделку (PnL/сделок) против Buy&Hold.

    python edge_horizon.py --data-dir /abs/data/processed
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from src.common import (
    PER_YEAR_1M, SimpleLSTM, WindowDataset, apply_min_holding, sharpe_ratio,
    set_seed, _predict_labels_probs, _train_epoch,
)
from src.train import _prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60


def _cfg(data_dir, horizon):
    return OmegaConf.create({
        "seed": 42,
        "data": {"active_asset": "BTC", "data_dir": data_dir, "synthetic": False,
                 "splits": {"X_train": "X_train.parquet", "X_val": "X_val.parquet",
                            "X_test": "X_test.parquet", "y_train": "y_train.parquet",
                            "y_val": "y_val.parquet", "y_test": "y_test.parquet"}},
        "model": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "lr": 1e-3,
                  "batch_size": 512, "epochs": 12, "patience": 4, "window": WINDOW,
                  "horizon": horizon, "min_holding": horizon, "cost_bps": 7},
    })


def train_lstm(ld_tr, ld_va, yva_w, wt, nfeat, epochs=12, patience=4):
    torch.manual_seed(42)
    model = SimpleLSTM(nfeat, 64, 1, 0.2).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=wt.to(DEVICE)); opt = optim.Adam(model.parameters(), lr=1e-3)
    best, state, stale = -1, None, 0
    for _ in range(epochs):
        _train_epoch(model, ld_tr, crit, opt, DEVICE)
        _, _, p1 = _predict_labels_probs(model, ld_va, DEVICE)
        m = min(len(p1), len(yva_w))
        a = roc_auc_score(yva_w[:m], p1[:m]) if len(np.unique(yva_w[:m])) > 1 else 0.5
        if a > best: best, state, stale = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience: break
    if state: model.load_state_dict(state)
    return model, best


def net_metrics(pos, r1, cpt):
    gross = pos * r1
    turn = pos.diff().abs().fillna(pos.abs())
    net = gross - turn * cpt
    ntr = int(turn.sum())
    edge = float(net.sum()) / max(ntr, 1)   # edge на сделку (1-барный PnL / сделок)
    return sharpe_ratio(net, PER_YEAR_1M), ntr, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    set_seed(42)
    rows = []
    bh_done = False
    for H in [15, 60, 240]:
        print(f"=== HORIZON={H} ===")
        data = _prepare_data(_cfg(args.data_dir, H))
        nfeat = data["n_features"]
        ds_tr = WindowDataset(data["Xt_s"], data["yt_tb"], WINDOW)
        ds_va = WindowDataset(data["Xv_s"], data["yv_tb"], WINDOW)
        ds_te = WindowDataset(data["Xte_s"], data["yte_tb"], WINDOW)
        ct = np.bincount(ds_tr.labels, minlength=2)
        wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
        m, vauc = train_lstm(DataLoader(ds_tr, batch_size=512, shuffle=True),
                             DataLoader(ds_va, batch_size=512), ds_va.labels, wt, nfeat)
        _, yte_w, p1 = _predict_labels_probs(m, DataLoader(ds_te, batch_size=512), DEVICE)
        idx = data["idx_te_w"]; n = min(len(p1), len(idx))
        r1 = pd.Series(np.asarray(data["r1_te"])[:n], index=idx[:n])
        roc = roc_auc_score(yte_w[:n], p1[:n]) if len(np.unique(yte_w[:n])) > 1 else float("nan")
        pos = apply_min_holding(pd.Series((p1[:n] >= 0.5).astype(float), index=idx[:n]), H)
        sh7, ntr, edge7 = net_metrics(pos, r1, 7e-4)
        sh1, _, _ = net_metrics(pos, r1, 1e-4)
        rows.append({"HORIZON": H, "ROC-AUC": round(roc, 4), "Sharpe net taker(7bps)": round(sh7, 3),
                     "Sharpe net maker(1bps)": round(sh1, 3), "edge/сделку(bps)": round(edge7 * 1e4, 3),
                     "сделок": ntr})
        if not bh_done:
            bh = pd.Series(1.0, index=idx[:n])
            shb, _, _ = net_metrics(bh, r1, 0.0)
            bh_sharpe = round(shb, 3); bh_done = True
        print(f"  ROC={roc:.4f} netSharpe taker={sh7:.2f} maker={sh1:.2f} edge/сделку={edge7*1e4:.2f}bps сделок={ntr}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(f"РЫЧАГ C: edge на сделку по горизонту (Buy&Hold net Sharpe = {bh_sharpe})")
    print("=" * 90)
    print(df.to_string(index=False))
    print("\nИздержки на сделку: taker=7bps, maker=1bps. edge>издержек ⇒ прибыльно.")
    df.to_csv("edge_horizon_results.csv", index=False, encoding="utf-8")
    print("Сохранено: edge_horizon_results.csv")


if __name__ == "__main__":
    main()
