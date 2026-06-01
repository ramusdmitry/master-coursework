"""
cp5_fixed.py — воспроизведение зоопарка CP5 на честных данных с gross И net Sharpe.

Обучает модели CP5 (SimpleLSTM, 1D CNN, Transformer, RandomForest, Ensemble) на
triple-barrier таргете (полный тест 2025) и оценивает КАЖДУЮ двумя способами:
- Sharpe GROSS: argmax-позиция каждый бар, БЕЗ издержек (как считал CP5);
- Sharpe NET:  + min_holding=15 + издержки 7bps (честно, 1-барная доходность).

Цель — показать на ЕДИНОЙ метрике, что «лучшая» по gross (1D CNN) на net проваливается.
DL на GPU (~/.envs/ds). Основа артефакта checkpoint-5-fixed.ipynb.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from src.common import (
    PER_YEAR_1M,
    SimpleLSTM,
    WindowDataset,
    apply_min_holding,
    evaluate_strategy,
    set_seed,
    _predict_labels_probs,
    _train_epoch,
)
from src.zoo import TransformerClassifier
from src.train import _prepare_data
from benchmark_full_test import _build_cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60
MIN_HOLD = 15


class Conv1dClassifier(nn.Module):
    """1D CNN из CP5 (Часть E.1): 2× Conv1d + Global Max Pool."""
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, 3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_size * 2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.global_pool(self.relu(self.conv2(x)))
        return self.fc(self.dropout(x.squeeze(-1)))


def aggregate_windows(X, y, window):
    """Агрегаты окна для RF: mean|std|min|max|last (как в CP5)."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(X) - window
    sw = sliding_window_view(X, window, axis=0)[:n]  # (n, F, w)
    agg = np.concatenate([sw.mean(2), sw.std(2), sw.min(2), sw.max(2), sw[:, :, -1]], axis=1)
    return agg.astype(np.float32), y[window:window + n]


def train_dl(model, ld_tr, ld_va, yva_w, w_tensor, epochs=15, patience=4, tag=""):
    """Обучение DL с early stopping по val ROC-AUC (как эталон CP5)."""
    model = model.to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w_tensor.to(DEVICE))
    opt = optim.Adam(model.parameters(), lr=1e-3)
    best_auc, best_state, stale = -1, None, 0
    for ep in range(1, epochs + 1):
        _train_epoch(model, ld_tr, crit, opt, DEVICE)
        _, _, p1 = _predict_labels_probs(model, ld_va, DEVICE)
        m = min(len(p1), len(yva_w))
        auc = roc_auc_score(yva_w[:m], p1[:m]) if len(np.unique(yva_w[:m])) > 1 else 0.5
        if auc > best_auc:
            best_auc, best_state, stale = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    print(f"  [{tag}] best val ROC-AUC={best_auc:.4f}")
    return model


def evaluate_both(p1, r1, idx, yte_w):
    """Возвращает (ROC-AUC, Sharpe gross argmax, Sharpe net min_hold+costs)."""
    n = min(len(p1), len(idx), len(r1))
    p1, idxn, r1n = p1[:n], idx[:n], pd.Series(np.asarray(r1)[:n], index=idx[:n])
    roc = roc_auc_score(yte_w[:n], p1) if len(np.unique(yte_w[:n])) > 1 else float("nan")
    pos_arg = pd.Series((p1 >= 0.5).astype(float), index=idxn)
    ev_g = evaluate_strategy(pos_arg, r1n, PER_YEAR_1M)              # gross argmax
    pos_mh = apply_min_holding(pos_arg, MIN_HOLD)
    ev_n = evaluate_strategy(pos_mh, r1n, PER_YEAR_1M)              # net + min_hold
    return roc, ev_g["Sharpe gross"], ev_n["Sharpe net"], ev_n["Сделок (смен позиции)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    set_seed(42)

    data = _prepare_data(_build_cfg(args.data_dir))
    Xt, Xv, Xte = data["Xt_s"], data["Xv_s"], data["Xte_s"]
    yt, yv, yte = data["yt_tb"], data["yv_tb"], data["yte_tb"]
    r1_te, idx_te = data["r1_te"], data["idx_te_w"]
    nfeat = data["n_features"]
    print(f"device={DEVICE}, фич={nfeat}, train={len(Xt)}, test_окон={len(idx_te)}")

    ds_tr, ds_va = WindowDataset(Xt, yt, WINDOW), WindowDataset(Xv, yv, WINDOW)
    ds_te = WindowDataset(Xte, yte, WINDOW)
    ld_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=512)
    ld_te = DataLoader(ds_te, batch_size=512)
    ct = np.bincount(ds_tr.labels, minlength=2)
    w_tensor = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
    yte_w = ds_te.labels
    yva_w = ds_va.labels

    results = {}
    p1_store = {}

    # --- DL модели ---
    for name, ctor in [
        ("SimpleLSTM", lambda: SimpleLSTM(nfeat, 64, 1, 0.2)),
        ("1D CNN", lambda: Conv1dClassifier(nfeat, 64, 0.2)),
        ("Transformer", lambda: TransformerClassifier(nfeat, hidden=64, num_layers=2, dropout=0.2)),
    ]:
        print(f"Обучение {name}...")
        torch.manual_seed(42)
        m = train_dl(ctor(), ld_tr, ld_va, yva_w, w_tensor, epochs=args.epochs, tag=name)
        _, _, p1 = _predict_labels_probs(m, ld_te, DEVICE)
        p1_store[name] = p1
        results[name] = evaluate_both(p1, r1_te, idx_te, yte_w)

    # --- RandomForest на агрегатах окна ---
    print("Обучение RandomForest...")
    Xt_agg, yt_agg = aggregate_windows(Xt, yt, WINDOW)
    Xte_agg, _ = aggregate_windows(Xte, yte, WINDOW)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(Xt_agg, yt_agg)
    rf_p1 = rf.predict_proba(Xte_agg)[:, 1]
    p1_store["RandomForest"] = rf_p1
    results["RandomForest"] = evaluate_both(rf_p1, r1_te, idx_te, yte_w)

    # --- Ensemble (avg p1 четырёх моделей) ---
    n = min(len(v) for v in p1_store.values())
    ens = np.mean([v[:n] for v in p1_store.values()], axis=0)
    results["Ensemble (avg)"] = evaluate_both(ens, r1_te, idx_te, yte_w)

    # --- Buy & Hold ---
    bh = pd.Series(1.0, index=idx_te)
    evb = evaluate_strategy(bh, pd.Series(np.asarray(r1_te), index=idx_te), PER_YEAR_1M)
    results["Buy & Hold"] = (float("nan"), evb["Sharpe gross"], evb["Sharpe net"], evb["Сделок (смен позиции)"])

    df = pd.DataFrame(
        [(k, v[0], v[1], v[2], v[3]) for k, v in results.items()],
        columns=["Модель", "ROC-AUC", "Sharpe GROSS (argmax)", "Sharpe NET (min_hold+costs)", "Сделок"],
    ).sort_values("Sharpe GROSS (argmax)", ascending=False)
    print("\n" + "=" * 92)
    print("CP5-ЗООПАРК НА ЧЕСТНЫХ ДАННЫХ: gross (как CP5) vs net (честно), полный тест 2025")
    print("=" * 92)
    print(df.to_string(index=False))
    df.to_csv("cp5_fixed_results.csv", index=False, encoding="utf-8")
    print("\nСохранено: cp5_fixed_results.csv")


if __name__ == "__main__":
    main()
