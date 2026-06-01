"""
eval_assets.py — оценка прогноза/торговли по ВСЕМ 5 активам (BTC/ETH/BNB/SOL/XRP).

Прошлые замеры были только по BTC. Здесь для каждого актива обучаем SimpleLSTM
(triple-barrier) и считаем ROC-AUC + Sharpe net (1-бар) при taker(7bps) и maker(1bps),
min_holding 15 и 60. Гипотеза: на менее ликвидных альтах рынок менее эффективен →
больше предсказуемости (ROC>0.515) и реальный edge.

    python eval_assets.py --data-dir /abs/data/processed
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

from src.common import (PER_YEAR_1M, SimpleLSTM, WindowDataset, apply_min_holding,
                        sharpe_ratio, set_seed, _predict_labels_probs, _train_epoch)
from src.train import _prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60
ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP"]


def _cfg(data_dir, asset, horizon=15):
    return OmegaConf.create({
        "seed": 42,
        "data": {"active_asset": asset, "data_dir": data_dir, "synthetic": False,
                 "splits": {"X_train": "X_train.parquet", "X_val": "X_val.parquet",
                            "X_test": "X_test.parquet", "y_train": "y_train.parquet",
                            "y_val": "y_val.parquet", "y_test": "y_test.parquet"}},
        "model": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "lr": 1e-3,
                  "batch_size": 512, "epochs": 15, "patience": 4, "window": WINDOW,
                  "horizon": horizon, "min_holding": horizon, "cost_bps": 7},
    })


def train_lstm(ld_tr, ld_va, yva_w, wt, nfeat, epochs=15, patience=4):
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
    return model


def net_sh(pos, r1, cpt):
    gross = pos * r1
    turn = pos.diff().abs().fillna(pos.abs())
    return sharpe_ratio(gross - turn * cpt, PER_YEAR_1M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    set_seed(42)
    rows = []
    for asset in ASSETS:
        print(f"=== {asset} ===")
        try:
            data = _prepare_data(_cfg(args.data_dir, asset))
        except Exception as e:
            print(f"  {asset} пропущен: {e}"); continue
        nfeat = data["n_features"]
        ds_tr = WindowDataset(data["Xt_s"], data["yt_tb"], WINDOW)
        ds_va = WindowDataset(data["Xv_s"], data["yv_tb"], WINDOW)
        ds_te = WindowDataset(data["Xte_s"], data["yte_tb"], WINDOW)
        ct = np.bincount(ds_tr.labels, minlength=2)
        wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
        m = train_lstm(DataLoader(ds_tr, batch_size=512, shuffle=True),
                       DataLoader(ds_va, batch_size=512), ds_va.labels, wt, nfeat)
        _, yte_w, p1 = _predict_labels_probs(m, DataLoader(ds_te, batch_size=512), DEVICE)
        idx = data["idx_te_w"]; n = min(len(p1), len(idx))
        r1 = pd.Series(np.asarray(data["r1_te"])[:n], index=idx[:n])
        roc = roc_auc_score(yte_w[:n], p1[:n]) if len(np.unique(yte_w[:n])) > 1 else float("nan")
        base = pd.Series((p1[:n] >= 0.5).astype(float), index=idx[:n])
        pos15 = apply_min_holding(base, 15); pos60 = apply_min_holding(base, 60)
        bh = pd.Series(1.0, index=idx[:n])
        rec = {
            "Актив": asset, "ROC-AUC": round(roc, 4),
            "Sh taker mh15": round(net_sh(pos15, r1, 7e-4), 2),
            "Sh maker mh15": round(net_sh(pos15, r1, 1e-4), 2),
            "Sh maker mh60": round(net_sh(pos60, r1, 1e-4), 2),
            "Sh Buy&Hold": round(net_sh(bh, r1, 0.0), 2),
        }
        rows.append(rec)
        print(f"  ROC={roc:.4f} maker_mh60={rec['Sh maker mh60']} BH={rec['Sh Buy&Hold']}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("ПРОГНОЗ/ТОРГОВЛЯ ПО ВСЕМ АКТИВАМ (SimpleLSTM, полный тест 2025, 1-бар Sharpe)")
    print("=" * 80)
    print(df.to_string(index=False))
    df.to_csv("eval_assets_results.csv", index=False, encoding="utf-8")
    print("\nСохранено: eval_assets_results.csv")


if __name__ == "__main__":
    main()
