"""
deepen_1h.py — шаг 2: углубить BTC 1h (лучший кандидат, 5/9 walk-forward).

Добавляем (a) order-flow фичи на 1h (есть в данных 2021-2025), (b) ансамбль
архитектур (LSTM/CNN/GRU/Transformer). Сравниваем с baseline 1h (OHLCV LSTM).
Статический тест 2025 (ROC + Sharpe maker vs BH); лучшую -> walk-forward отдельно.

    python deepen_1h.py --raw-dir /abs/data/raw
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_processed_data as bp
from src.common import (SimpleLSTM, WindowDataset, apply_min_holding, sharpe_ratio,
                        set_seed, _predict_labels_probs, _train_epoch)
from src.zoo import GRUClassifier, TransformerClassifier
from cp5_fixed import Conv1dClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60
PER_YEAR_1H = 24 * 365
TR_END, VA_END = "2023-12-31", "2024-12-31"


def train(model, ld_tr, ld_va, yva, wt, epochs=20, patience=5):
    model = model.to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=wt.to(DEVICE)); opt = optim.Adam(model.parameters(), lr=1e-3)
    best, st, stale = -1, None, 0
    for _ in range(epochs):
        _train_epoch(model, ld_tr, crit, opt, DEVICE)
        _, _, p1 = _predict_labels_probs(model, ld_va, DEVICE)
        k = min(len(p1), len(yva))
        a = roc_auc_score(yva[:k], p1[:k]) if len(np.unique(yva[:k])) > 1 else 0.5
        if a > best: best, st, stale = a, {kk: v.cpu().clone() for kk, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience: break
    if st: model.load_state_dict(st)
    return model


def build_1h(raw_dir, order_flow):
    bp.DAY_N = 24
    df1m = bp._read_single_path(Path(raw_dir) / "BTC_1m.csv")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in bp.OF_COLS:
        if c in df1m.columns: agg[c] = "sum"
    d = df1m.resample("1h").agg(agg).dropna(subset=["close"])
    all_df = pd.DataFrame(d.values, index=d.index,
                          columns=pd.MultiIndex.from_product([["BTC"], d.columns]))
    feats = bp.build_features(all_df, order_flow=order_flow)
    close = all_df[("BTC", "close")]
    y_reg = np.log(close.shift(-1) / close)
    data = feats.join(y_reg.rename("y_reg")).replace([np.inf, -np.inf], np.nan).dropna()
    return data


def split_eval(data, models_specs, label):
    idx = data.index
    X = data[[c for c in data.columns if c != "y_reg"]].values.astype(np.float32)
    r1 = data["y_reg"].values.astype(float); yb = (r1 > 0).astype(np.int64)
    tr = idx <= pd.Timestamp(TR_END, tz="UTC")
    va = (idx > pd.Timestamp(TR_END, tz="UTC")) & (idx <= pd.Timestamp(VA_END, tz="UTC"))
    te = idx > pd.Timestamp(VA_END, tz="UTC")
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
    ds_tr = WindowDataset(Xtr, yb[tr], WINDOW); ds_va = WindowDataset(Xva, yb[va], WINDOW)
    ds_te = WindowDataset(Xte, yb[te], WINDOW)
    ct = np.bincount(ds_tr.labels, minlength=2)
    wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
    nfeat = X.shape[1]
    te_idx_all = idx[te]
    r1_te_full = r1[te]
    p1_store = {}
    rows = []
    for name, ctor in models_specs:
        torch.manual_seed(42)
        m = train(ctor(nfeat), DataLoader(ds_tr, batch_size=256, shuffle=True),
                  DataLoader(ds_va, batch_size=256), ds_va.labels, wt)
        _, yte_w, p1 = _predict_labels_probs(m, DataLoader(ds_te, batch_size=256), DEVICE)
        p1_store[name] = p1
        rows.append((f"{label}/{name}", p1, yte_w))
    # ансамбль
    n = min(len(v) for v in p1_store.values())
    ens = np.mean([v[:n] for v in p1_store.values()], axis=0)
    yte_w0 = rows[0][2][:n]
    rows.append((f"{label}/Ensemble", ens, yte_w0))

    out = []
    for nm, p1, yw in rows:
        n = min(len(p1), len(te_idx_all) - WINDOW)
        ti = te_idx_all[WINDOW:WINDOW + n]
        r1t = pd.Series(r1_te_full[WINDOW:WINDOW + n], index=ti)
        roc = roc_auc_score(yw[:n], p1[:n]) if len(np.unique(yw[:n])) > 1 else float("nan")
        pos = apply_min_holding(pd.Series((p1[:n] >= 0.5).astype(float), index=ti), 1)
        turn = pos.diff().abs().fillna(pos.abs())
        sh = sharpe_ratio(pos * r1t - turn * 1e-4, PER_YEAR_1H)
        bh = sharpe_ratio(pd.Series(1.0, index=ti) * r1t, PER_YEAR_1H)
        out.append({"Конфиг": nm, "ROC-AUC": round(roc, 4), "Sh maker": round(sh, 2), "BH": round(bh, 2)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()
    set_seed(42)
    specs = [
        ("LSTM", lambda nf: SimpleLSTM(nf, 64, 1, 0.2)),
        ("CNN", lambda nf: Conv1dClassifier(nf, 64, 0.2)),
        ("GRU", lambda nf: GRUClassifier(nf, hidden=64, num_layers=1, dropout=0.2)),
        ("Transformer", lambda nf: TransformerClassifier(nf, hidden=64, num_layers=2, dropout=0.2)),
    ]
    all_rows = []
    print("=== baseline (OHLCV) ===")
    all_rows += split_eval(build_1h(args.raw_dir, False), specs, "OHLCV")
    print("=== + order-flow ===")
    all_rows += split_eval(build_1h(args.raw_dir, True), specs, "OF")
    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 70)
    print("ШАГ 2: углубление BTC 1h (order-flow + ансамбль), тест 2025")
    print("=" * 70)
    print(df.to_string(index=False))
    df.to_csv("deepen_1h_results.csv", index=False, encoding="utf-8")
    print(f"\nBaseline (OHLCV/LSTM): ROC {df.iloc[0]['ROC-AUC']}, Sh {df.iloc[0]['Sh maker']} vs BH {df.iloc[0]['BH']}")


if __name__ == "__main__":
    main()
