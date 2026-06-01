"""
wf_best.py — walk-forward устойчивость лучшей конфигурации (SimpleLSTM, mh60, maker).

Помесячное переобучение (expanding window): для каждого месяца 2025 обучаем
SimpleLSTM на triple-barrier до месяца, предсказываем месяц, оцениваем net Sharpe
на 1-баре с maker-комиссией (1bps) и min_holding=60 — против Buy&Hold.

Проверяет: устойчиво ли преимущество над BH (sweep дал 1.24 на одном периоде)?

    python wf_best.py --data-dir /abs/data/processed --cost-bps 1 --min-hold 60
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from src.common import (
    PER_YEAR_1M, PT_MULT, SL_MULT, VOL_WINDOW_TB, SimpleLSTM, WindowDataset,
    apply_min_holding, forward_return, resolve_y_column, set_seed, sharpe_ratio,
    triple_barrier_labels, _predict_labels_probs, _train_epoch,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW, HORIZON = 60, 15


def _load_full(d):
    import os
    X = pd.concat([pd.read_parquet(os.path.join(d, f"X_{s}.parquet")) for s in ["train", "val", "test"]]).sort_index()
    y = pd.concat([pd.read_parquet(os.path.join(d, f"y_{s}.parquet")) for s in ["train", "val", "test"]]).sort_index()
    X = X[~X.index.duplicated(keep="first")]; y = y[~y.index.duplicated(keep="first")]
    feat = [c for c in X.columns if str(c).startswith("BTC__")]
    return X[feat], y[resolve_y_column(y, "BTC", "y_reg")].astype(float)


def _train(model, ld_tr, ld_va, yva_w, wt, epochs=12, patience=4):
    from sklearn.metrics import roc_auc_score
    model = model.to(DEVICE)
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
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--min-hold", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    set_seed(42)
    cpt = args.cost_bps / 1e4
    Xdf, yreg = _load_full(args.data_dir)
    idx = Xdf.index
    Xv = Xdf.values.astype(np.float32)
    r1_full = forward_return(yreg, 1).reindex(idx).values.astype(float)  # 1-бар доходность

    rows = []
    for M in pd.period_range("2025-01", "2025-09", freq="M"):
        ms = pd.Timestamp(M.start_time, tz="UTC"); me = pd.Timestamp((M + 1).start_time, tz="UTC")
        te0 = int(np.searchsorted(idx.values, np.datetime64(ms)))
        te1 = int(np.searchsorted(idx.values, np.datetime64(me)))
        va0 = int(np.searchsorted(idx.values, np.datetime64(ms - pd.Timedelta(days=30))))
        if te1 <= te0 or va0 < WINDOW + 1000:
            continue
        # triple-barrier таргет на train-части (до месяца)
        s_tr = yreg.iloc[:te0]
        tb = triple_barrier_labels(s_tr, VOL_WINDOW_TB, PT_MULT, SL_MULT, HORIZON)
        ytb = tb.fillna(0).astype(np.int64).values
        sc = StandardScaler().fit(Xv[:va0])
        Xtr_s = sc.transform(Xv[:va0]).astype(np.float32)
        Xva_s = sc.transform(Xv[va0:te0]).astype(np.float32)
        Xte_s = sc.transform(Xv[te0 - WINDOW:te1]).astype(np.float32)
        ds_tr = WindowDataset(Xtr_s, ytb[:va0], WINDOW)
        ds_va = WindowDataset(Xva_s, ytb[va0:te0], WINDOW)
        if len(ds_va) < 50: continue
        ct = np.bincount(ds_tr.labels, minlength=2)
        wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
        torch.manual_seed(42)
        model = _train(SimpleLSTM(Xv.shape[1], 64, 1, 0.2),
                       DataLoader(ds_tr, batch_size=512, shuffle=True),
                       DataLoader(ds_va, batch_size=512), ds_va.labels, wt, epochs=args.epochs)
        ds_te = WindowDataset(Xte_s, np.zeros(len(Xte_s), np.int64), WINDOW)
        _, _, p1 = _predict_labels_probs(model, DataLoader(ds_te, batch_size=512), DEVICE)
        midx = idx[te0:te1]; n = min(len(p1), len(midx))
        pos = apply_min_holding(pd.Series((p1[:n] >= 0.5).astype(float), index=midx[:n]), args.min_hold)
        r1 = pd.Series(r1_full[te0:te0 + n], index=midx[:n])
        turn = pos.diff().abs().fillna(pos.abs())
        net = pos * r1 - turn * cpt
        ss = sharpe_ratio(net, PER_YEAR_1M)
        sb = sharpe_ratio(pd.Series(1.0, index=midx[:n]) * r1, PER_YEAR_1M)
        rows.append((str(M), ss, sb, 100 * (pos.values > 0).mean(), int(turn.sum())))
        print(f"[{M}] strat={ss:.2f} BH={sb:.2f} %long={100*(pos.values>0).mean():.0f} сделок={int(turn.sum())}")

    print("\n" + "=" * 64)
    print(f"WALK-FORWARD: SimpleLSTM mh={args.min_hold}, cost={args.cost_bps}bps (1-бар), 2025")
    print("=" * 64)
    print(f"{'месяц':<10}{'strat':>9}{'BH':>9}{'%long':>8}{'выигр':>8}")
    wins = 0
    for m, ss, sb, fl, tr in rows:
        wins += int(ss > sb)
        print(f"{m:<10}{ss:>9.2f}{sb:>9.2f}{fl:>7.0f}%{'  ДА' if ss>sb else '  нет':>8}")
    print("-" * 64)
    print(f"Месяцев strat>BH: {wins}/{len(rows)}")


if __name__ == "__main__":
    main()
