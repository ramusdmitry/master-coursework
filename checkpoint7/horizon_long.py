"""
horizon_long.py — прогноз направления на горизонтах 1h/4h/1d, 5 активов, 2021-2025.

На длинной истории (4.7 года) проверяем гипотезу: на крупных барах сигнал чище →
ROC выше 0.515 и Sharpe положительный (мало сделок). Честный сплит по датам
(train 2021-2023, val 2024, test 2025), правильная аннуализация Sharpe под таймфрейм.

    python horizon_long.py --raw-dir /abs/data/raw
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # корень репо
import build_processed_data as bp
from src.common import (SimpleLSTM, WindowDataset, apply_min_holding, sharpe_ratio,
                        set_seed, _predict_labels_probs, _train_epoch)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60
ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP"]
TFS = {"1h": ("1h", 24, 8760), "4h": ("4h", 6, 2190), "1d": ("1D", 1, 365)}  # freq, bars/day, per_year
TR_END, VA_END = "2023-12-31", "2024-12-31"


def train_lstm(ld_tr, ld_va, yva_w, wt, nfeat, epochs=20, patience=5):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()
    set_seed(42)
    from pathlib import Path
    rows = []
    for asset in ASSETS:
        df1m = bp._read_single_path(Path(args.raw_dir) / f"{asset}_1m.csv")
        for tfname, (freq, bars_day, per_year) in TFS.items():
            bp.DAY_N = bars_day
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            d = df1m.resample(freq).agg(agg).dropna(subset=["close"])
            all_df = pd.DataFrame(d.values, index=d.index,
                                  columns=pd.MultiIndex.from_product([[asset], d.columns]))
            feats = bp.build_features(all_df)
            close = all_df[(asset, "close")]
            y_reg = np.log(close.shift(-1) / close)        # доходность след. бара
            y_bin = (y_reg > 0).astype(int)
            data = feats.join(y_reg.rename("y_reg")).join(y_bin.rename("y_bin")).dropna()
            idx = data.index
            X = data[[c for c in feats.columns]].values.astype(np.float32)
            yb = data["y_bin"].values.astype(np.int64)
            r1 = data["y_reg"].values.astype(float)
            tr = idx <= pd.Timestamp(TR_END, tz="UTC")
            va = (idx > pd.Timestamp(TR_END, tz="UTC")) & (idx <= pd.Timestamp(VA_END, tz="UTC"))
            te = idx > pd.Timestamp(VA_END, tz="UTC")
            if tr.sum() < WINDOW + 200 or te.sum() < WINDOW + 20:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
            ds_tr = WindowDataset(Xtr, yb[tr], WINDOW); ds_va = WindowDataset(Xva, yb[va], WINDOW)
            ds_te = WindowDataset(Xte, yb[te], WINDOW)
            if len(ds_va) < 30 or len(ds_te) < 20: continue
            ct = np.bincount(ds_tr.labels, minlength=2)
            wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
            m = train_lstm(DataLoader(ds_tr, batch_size=256, shuffle=True),
                           DataLoader(ds_va, batch_size=256), ds_va.labels, wt, X.shape[1])
            _, yte_w, p1 = _predict_labels_probs(m, DataLoader(ds_te, batch_size=256), DEVICE)
            n = len(p1)
            te_idx = idx[te][WINDOW:WINDOW + n]
            r1_te = pd.Series(r1[te][WINDOW:WINDOW + n], index=te_idx)
            roc = roc_auc_score(yte_w[:n], p1[:n]) if len(np.unique(yte_w[:n])) > 1 else float("nan")
            pos = apply_min_holding(pd.Series((p1[:n] >= 0.5).astype(float), index=te_idx), 1)

            def sh(p, cpt):
                turn = p.diff().abs().fillna(p.abs())
                return sharpe_ratio(p * r1_te - turn * cpt, per_year)
            bh = pd.Series(1.0, index=te_idx)
            rec = {"Актив": asset, "TF": tfname, "test_баров": n, "ROC-AUC": round(roc, 4),
                   "Sh maker": round(sh(pos, 1e-4), 2), "Sh taker": round(sh(pos, 7e-4), 2),
                   "BH": round(sh(bh, 0.0), 2), "%long": round(100 * (pos.values > 0).mean(), 1)}
            rows.append(rec)
            print(f"{asset} {tfname}: ROC={roc:.4f} Sh_maker={rec['Sh maker']} BH={rec['BH']} (test={n})")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("ГОРИЗОНТЫ 1h/4h/1d, 5 активов, 2021-2025 (правильная аннуализация, 1-бар Sharpe)")
    print("=" * 92)
    print(df.to_string(index=False))
    df.to_csv("horizon_long_results.csv", index=False, encoding="utf-8")
    print("\nИщем: ROC>0.515 И Sh maker>BH ⇒ реальный edge")
    win = df[(df["ROC-AUC"] > 0.515) & (df["Sh maker"] > df["BH"])]
    print(win.to_string(index=False) if len(win) else "нет конфигураций с ROC>0.515 и Sh>BH одновременно")


if __name__ == "__main__":
    main()
