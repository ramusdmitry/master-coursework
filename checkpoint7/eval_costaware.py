"""
eval_costaware.py — проверка рычага №5: согласовать таргет прогноза с торговлей.

Вместо triple-barrier метки модель учится предсказывать ЗНАК net forward-return
за горизонт удержания: y = 1, если forward_return(H) > cost_thr. Тогда ROC-AUC
меряет ровно то, что приносит Sharpe — и они должны сойтись (если edge > издержек).

Сравнивается с baseline (eval_primary на triple-barrier): тот же primary SimpleLSTM,
те же данные, отличается только ТАРГЕТ обучения/оценки.

    python eval_costaware.py --data-dir /abs/data/processed_of --cost-thr 0 --tag ca0
    python eval_costaware.py --data-dir /abs/data/processed_of --cost-thr 0.0007 --tag ca_cost
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
    PER_YEAR_1M,
    PER_YEAR_FWD,
    SimpleLSTM,
    WindowDataset,
    apply_min_holding,
    evaluate_on_forward,
    evaluate_strategy,
    fit_classifier_sharpe,
    set_seed,
    _predict_labels_probs,
)
from src.train import _prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg(data_dir, epochs=30):
    return OmegaConf.create({
        "seed": 42,
        "data": {"active_asset": "BTC", "data_dir": data_dir, "synthetic": False,
                 "splits": {"X_train": "X_train.parquet", "X_val": "X_val.parquet",
                            "X_test": "X_test.parquet", "y_train": "y_train.parquet",
                            "y_val": "y_val.parquet", "y_test": "y_test.parquet"}},
        "model": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "lr": 1e-3,
                  "batch_size": 512, "epochs": epochs, "patience": 5, "window": 60,
                  "horizon": 15, "min_holding": 15, "cost_bps": 7},
    })


def _costaware_labels(n_total, fwd, window, cost_thr):
    """y[window+k] = 1 если fwd[k] > cost_thr (знак net-return за горизонт)."""
    y = np.zeros(n_total, dtype=np.int64)
    v = np.asarray(fwd.values, dtype=float)
    m = min(len(v), n_total - window)
    y[window:window + m] = (v[:m] > cost_thr).astype(np.int64)
    return y


@torch.no_grad()
def _infer_p1(model, Xte_s, window, chunk=80000):
    model.eval()
    N = len(Xte_s); n = N - window
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ds = WindowDataset(Xte_s[s:window + e], np.zeros(window + (e - s), np.int64), window)
        _, _, p1 = _predict_labels_probs(model, DataLoader(ds, batch_size=512), DEVICE)
        out[s:e] = p1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cost-thr", type=float, default=0.0)
    ap.add_argument("--tag", default="costaware")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    set_seed(42)
    data = _prepare_data(_cfg(args.data_dir, args.epochs))
    window = 60
    nfeat = data["n_features"]

    # costs-aware таргеты (знак net forward-return) вместо triple-barrier
    y_tr = _costaware_labels(len(data["Xt_s"]), data["fwd_tr"], window, args.cost_thr)
    y_va = _costaware_labels(len(data["Xv_s"]), data["fwd_va"], window, args.cost_thr)

    ds_tr = WindowDataset(data["Xt_s"], y_tr, window)
    ds_va = WindowDataset(data["Xv_s"], y_va, window)
    print(f"[{args.tag}] фич: {nfeat}, cost_thr={args.cost_thr}, "
          f"баланс train y: {np.bincount(ds_tr.labels) }, device={DEVICE}")

    ct = np.bincount(ds_tr.labels, minlength=2)
    cw = ct.sum() / (2.0 * np.maximum(ct, 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))
    ld_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=512, shuffle=False)

    torch.manual_seed(42)
    model = SimpleLSTM(input_size=nfeat, hidden_size=64, num_layers=1, dropout=0.2).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    fit_classifier_sharpe(model, ld_tr, ld_va, crit, opt,
                          fwd_val=data["fwd_va"], idx_val_w=data["idx_va_w"],
                          epochs=args.epochs, patience=5, tag=f"{args.tag}",
                          periods_per_year=PER_YEAR_FWD, min_hold=15, device=DEVICE)

    p1 = _infer_p1(model, data["Xte_s"], window)
    n = len(p1)
    idx = data["idx_te_w"][:n]
    # ROC по costs-aware таргету (то, что и торгуем)
    y_te = _costaware_labels(len(data["Xte_s"]), data["fwd_te"], window, args.cost_thr)[window:window + n]
    roc = roc_auc_score(y_te, p1) if len(np.unique(y_te)) > 1 else float("nan")
    acc = accuracy_score(y_te, (p1 >= 0.5).astype(int))
    pos = apply_min_holding(pd.Series((p1 >= 0.5).astype(float), index=idx), 15)
    ev = evaluate_on_forward(pos, data["fwd_te"], name=args.tag, prob=p1, true_label=y_te)
    ev1 = evaluate_strategy(pos, data["r1_te"], PER_YEAR_1M, name=args.tag, prob=p1, true_label=y_te)

    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТ [{args.tag}] costs-aware таргет (cost_thr={args.cost_thr}):")
    print(f"  ROC-AUC (net-return target):   {roc:.4f}")
    print(f"  Accuracy:                      {acc:.4f}")
    print(f"  Sharpe net (1бар, ГЛАВНАЯ):    {ev1.get('Sharpe net')}")
    print(f"  Sharpe net (fwd, старый):      {ev.get('Sharpe net')}")
    print(f"  PnL (1бар):                    {ev1.get('PnL net (sum logret)')}")
    print(f"  % в long:                      {ev1.get('% времени в long')}")
    print(f"  Сделок:                        {ev1.get('Сделок (смен позиции)')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
