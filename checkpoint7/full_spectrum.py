"""
full_spectrum.py — полный цикл: таймфреймы 1s..1h × активы, все данные 2021-2025.

- Крупные tf (1m/15m/30m/1h): 5 активов, ресемпл из {asset}_1m.csv (2021-2025),
  честный сплит по датам (train 2021-23, val 2024, test 2025).
- Субминутные (1s/5s/15s/30s): только BTC, из BTC_1s.csv (2025, что доступно),
  сплит по долям 70/15/15. (1s за 4.7 года × 5 активов недоступно — терабайты.)

Для каждого: ROC-AUC + Sharpe maker(1bps) vs Buy&Hold, правильная аннуализация под tf.

    python full_spectrum.py --raw-dir /abs/data/raw
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60
DAY = 86400  # секунд в сутках
# tf: (freq, bars/day, per_year)
BIG = {"1m": ("1min", 1440), "15m": ("15min", 96), "30m": ("30min", 48), "1h": ("1h", 24)}
SUB = {"1s": ("1s", 86400), "5s": ("5s", 17280), "15s": ("15s", 5760), "30s": ("30s", 2880)}
ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP"]
TR_END, VA_END = "2023-12-31", "2024-12-31"
MAX_TRAIN = 1_600_000  # ограничение train-баров (для субминутных, чтобы влезало по времени)


def train_lstm(ld_tr, ld_va, yva, wt, nfeat, epochs=12, patience=4):
    torch.manual_seed(42)
    m = SimpleLSTM(nfeat, 64, 1, 0.2).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=wt.to(DEVICE)); opt = optim.Adam(m.parameters(), lr=1e-3)
    best, st, stale = -1, None, 0
    for _ in range(epochs):
        _train_epoch(m, ld_tr, crit, opt, DEVICE)
        _, _, p1 = _predict_labels_probs(m, ld_va, DEVICE)
        k = min(len(p1), len(yva))
        a = roc_auc_score(yva[:k], p1[:k]) if len(np.unique(yva[:k])) > 1 else 0.5
        if a > best: best, st, stale = a, {kk: v.cpu().clone() for kk, v in m.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience: break
    if st: m.load_state_dict(st)
    return m


def run_one(src_df, asset, tf, freq, bars_day, split_mode):
    bp.DAY_N = bars_day
    per_year = bars_day * 365
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    d = src_df.resample(freq).agg(agg).dropna(subset=["close"])
    all_df = pd.DataFrame(d.values, index=d.index,
                          columns=pd.MultiIndex.from_product([[asset], d.columns]))
    feats = bp.build_features(all_df)
    close = all_df[(asset, "close")]
    y_reg = np.log(close.shift(-1) / close)
    data = feats.join(y_reg.rename("y_reg")).dropna()
    idx = data.index
    X = data[[c for c in feats.columns]].values.astype(np.float32)
    r1 = data["y_reg"].values.astype(float)
    yb = (r1 > 0).astype(np.int64)

    if split_mode == "date":
        tr = idx <= pd.Timestamp(TR_END, tz="UTC")
        va = (idx > pd.Timestamp(TR_END, tz="UTC")) & (idx <= pd.Timestamp(VA_END, tz="UTC"))
        te = idx > pd.Timestamp(VA_END, tz="UTC")
        tr_i, va_i, te_i = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
    else:  # доли
        n = len(idx); a, b = int(n * .70), int(n * .85)
        tr_i, va_i, te_i = np.arange(a), np.arange(a, b), np.arange(b, n)
    if len(tr_i) > MAX_TRAIN:
        tr_i = tr_i[-MAX_TRAIN:]
    if len(tr_i) < WINDOW + 200 or len(te_i) < WINDOW + 10 or len(va_i) < WINDOW + 5:
        return None
    sc = StandardScaler().fit(X[tr_i])
    Xtr, Xva, Xte = sc.transform(X[tr_i]), sc.transform(X[va_i]), sc.transform(X[te_i])
    ds_tr = WindowDataset(Xtr, yb[tr_i], WINDOW); ds_va = WindowDataset(Xva, yb[va_i], WINDOW)
    ds_te = WindowDataset(Xte, yb[te_i], WINDOW)
    if len(ds_va) < 20 or len(ds_te) < 10: return None
    ct = np.bincount(ds_tr.labels, minlength=2)
    wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
    m = train_lstm(DataLoader(ds_tr, batch_size=512, shuffle=True),
                   DataLoader(ds_va, batch_size=512), ds_va.labels, wt, X.shape[1])
    _, yte_w, p1 = _predict_labels_probs(m, DataLoader(ds_te, batch_size=512), DEVICE)
    n = len(p1)
    te_idx = idx[te_i][WINDOW:WINDOW + n]
    r1_te = pd.Series(r1[te_i][WINDOW:WINDOW + n], index=te_idx)
    roc = roc_auc_score(yte_w[:n], p1[:n]) if len(np.unique(yte_w[:n])) > 1 else float("nan")
    pos = apply_min_holding(pd.Series((p1[:n] >= 0.5).astype(float), index=te_idx), 1)
    turn = pos.diff().abs().fillna(pos.abs())
    sh_m = sharpe_ratio(pos * r1_te - turn * 1e-4, per_year)
    bh = sharpe_ratio(pd.Series(1.0, index=te_idx) * r1_te, per_year)
    return {"Актив": asset, "TF": tf, "ROC-AUC": round(roc, 4),
            "Sh maker": round(sh_m, 2), "BH": round(bh, 2), "test": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()
    set_seed(42)
    rd = Path(args.raw_dir)
    rows = []
    # крупные tf: все активы из {asset}_1m
    for asset in ASSETS:
        df1m = bp._read_single_path(rd / f"{asset}_1m.csv")
        for tf, (freq, bd) in BIG.items():
            r = run_one(df1m, asset, tf, freq, bd, "date")
            if r:
                rows.append(r); print(f"{asset} {tf}: ROC={r['ROC-AUC']} Sh={r['Sh maker']} BH={r['BH']} (test={r['test']})")
    # субминутные: только BTC из BTC_1s
    btc1s_path = rd / "BTC_1s.csv"
    if btc1s_path.exists():
        df1s = bp._read_single_path(btc1s_path)
        for tf, (freq, bd) in SUB.items():
            r = run_one(df1s, "BTC", tf, freq, bd, "ratio")
            if r:
                rows.append(r); print(f"BTC {tf}: ROC={r['ROC-AUC']} Sh={r['Sh maker']} BH={r['BH']} (test={r['test']})")

    df = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(["1s", "5s", "15s", "30s", "1m", "15m", "30m", "1h"])}
    df["_o"] = df["TF"].map(order)
    df = df.sort_values(["Актив", "_o"]).drop(columns="_o")
    print("\n" + "=" * 80)
    print("ПОЛНЫЙ ЦИКЛ: таймфреймы × активы (крупные на 2021-2025, субмин BTC на 2025)")
    print("=" * 80)
    print(df.to_string(index=False))
    df.to_csv("full_spectrum_results.csv", index=False, encoding="utf-8")
    print("\nROC>0.515 И Sh maker>BH:")
    w = df[(df["ROC-AUC"] > 0.515) & (df["Sh maker"] > df["BH"])]
    print(w.to_string(index=False) if len(w) else "нет")


if __name__ == "__main__":
    main()
