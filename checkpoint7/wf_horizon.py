"""
wf_horizon.py — walk-forward устойчивость BTC на горизонте 1h (или 4h).

Помесячное переобучение (expanding): для каждого месяца 2025 обучаем LSTM на 1h-барах
до месяца, предсказываем месяц, считаем Sharpe maker (1-бар, прав. аннуализация) vs BH.
Проверяет, устойчив ли edge (статически BTC 1h дал ROC 0.529, Sh 0.61>BH).

    python wf_horizon.py --raw-dir /abs/data/raw --tf 1h
"""

from __future__ import annotations

import argparse
import os
import sys
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_processed_data as bp
from src.common import (SimpleLSTM, WindowDataset, apply_min_holding, sharpe_ratio,
                        set_seed, _predict_labels_probs, _train_epoch)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60
TFS = {"15m": ("15min", 96, 96 * 365), "30m": ("30min", 48, 48 * 365),
       "1h": ("1h", 24, 8760), "4h": ("4h", 6, 2190)}


def _train(ld_tr, ld_va, yva, wt, nfeat, epochs=20, patience=5):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--order-flow", action="store_true")
    ap.add_argument("--wf-start", default="2025-01", help="первый месяц walk-forward")
    ap.add_argument("--wf-end", default="2025-09", help="последний месяц walk-forward")
    ap.add_argument("--extra-feat-csv", default="", help="CSV доп. 1h-фич (index=timestamp), напр. L2")
    args = ap.parse_args()
    set_seed(42)
    from pathlib import Path
    freq, bars_day, per_year = TFS[args.tf]
    bp.DAY_N = bars_day
    df1m = bp._read_single_path(Path(args.raw_dir) / f"{args.asset}_1m.csv")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in bp.OF_COLS:
        if c in df1m.columns: agg[c] = "sum"
    d = df1m.resample(freq).agg(agg).dropna(subset=["close"])
    all_df = pd.DataFrame(d.values, index=d.index,
                          columns=pd.MultiIndex.from_product([[args.asset], d.columns]))
    feats = bp.build_features(all_df, order_flow=args.order_flow).replace([np.inf, -np.inf], np.nan)
    if args.extra_feat_csv:   # подмешать внешние 1h-фичи (напр. L2-имбаланс стакана)
        ext = pd.read_csv(args.extra_feat_csv, index_col=0)
        ext.index = pd.to_datetime(ext.index, utc=True)
        feats = feats.join(ext, how="left").replace([np.inf, -np.inf], np.nan)
        print(f"[L2] добавлено фич: {list(ext.columns)}; строк с L2: {feats[ext.columns].notna().all(1).sum()}")
    close = all_df[(args.asset, "close")]
    y_reg = np.log(close.shift(-1) / close)
    data = feats.join(y_reg.rename("y_reg")).dropna()
    idx = data.index
    X = data[[c for c in feats.columns]].values.astype(np.float32)
    r1 = data["y_reg"].values.astype(float)
    yb = (data["y_reg"].values > 0).astype(np.int64)

    rows = []
    for M in pd.period_range(args.wf_start, args.wf_end, freq="M"):
        ms = pd.Timestamp(M.start_time, tz="UTC"); me = pd.Timestamp((M + 1).start_time, tz="UTC")
        te0 = int(np.searchsorted(idx.values, np.datetime64(ms)))
        te1 = int(np.searchsorted(idx.values, np.datetime64(me)))
        va0 = int(np.searchsorted(idx.values, np.datetime64(ms - pd.Timedelta(days=60))))
        if te1 <= te0 or va0 < WINDOW + 200 or (te1 - te0) < WINDOW + 10:
            continue
        sc = StandardScaler().fit(X[:va0])
        Xtr, Xva = sc.transform(X[:va0]), sc.transform(X[va0:te0])
        Xte = sc.transform(X[te0 - WINDOW:te1])
        ds_tr = WindowDataset(Xtr, yb[:va0], WINDOW); ds_va = WindowDataset(Xva, yb[va0:te0], WINDOW)
        if len(ds_va) < 30: continue
        ct = np.bincount(ds_tr.labels, minlength=2)
        wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
        m = _train(DataLoader(ds_tr, batch_size=256, shuffle=True),
                   DataLoader(ds_va, batch_size=256), ds_va.labels, wt, X.shape[1])
        ds_te = WindowDataset(Xte, np.zeros(len(Xte), np.int64), WINDOW)
        _, _, p1 = _predict_labels_probs(m, DataLoader(ds_te, batch_size=256), DEVICE)
        midx = idx[te0:te1]; n = min(len(p1), len(midx))
        r1m = pd.Series(r1[te0:te0 + n], index=midx[:n])
        pos = apply_min_holding(pd.Series((p1[:n] >= 0.5).astype(float), index=midx[:n]), 1)
        turn = pos.diff().abs().fillna(pos.abs())
        ss = sharpe_ratio(pos * r1m - turn * 1e-4, per_year)
        sb = sharpe_ratio(pd.Series(1.0, index=midx[:n]) * r1m, per_year)
        rows.append((str(M), ss, sb, n))
        print(f"[{M}] strat={ss:.2f} BH={sb:.2f} баров={n}")

    print("\n" + "=" * 56)
    print(f"WALK-FORWARD: {args.asset} {args.tf}, maker 1bps, 2025")
    print("=" * 56)
    wins = sum(1 for _, ss, sb, _ in rows if ss > sb)
    for mm, ss, sb, n in rows:
        print(f"{mm:<10}{ss:>8.2f}{sb:>8.2f}{'   ДА' if ss > sb else '   нет':>8}")
    print(f"Месяцев strat>BH: {wins}/{len(rows)}")


if __name__ == "__main__":
    main()
