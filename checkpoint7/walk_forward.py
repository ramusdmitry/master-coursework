"""
walk_forward.py — проверка устойчивости edge через помесячное переобучение.

Гипотеза: costs-aware edge затухает из-за concept drift (модель обучена один раз на
2024, к середине 2025 режим уходит). Walk-forward: для каждого тестового месяца M
обучаем costs-aware модель на ВСЕХ данных до M (expanding window) и предсказываем M.
Если адаптация удерживает edge — Sharpe положителен в большинстве месяцев.

Сравнивается со static-моделью (robustness_costaware.py: торгует ~3 мес, потом молчит).

    python walk_forward.py --data-dir /abs/data/processed_of --cost-thr 0.0007
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.common import (
    PER_YEAR_FWD,
    SimpleLSTM,
    WindowDataset,
    apply_min_holding,
    fit_classifier_sharpe,
    forward_return,
    net_returns,
    resolve_y_column,
    set_seed,
    sharpe_ratio,
    _predict_labels_probs,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60
HORIZON = 15
MIN_HOLD = 15
COST = 0.0007


def _load_full(data_dir):
    """Склеивает train/val/test в непрерывную серию (фичи + y_reg)."""
    import os
    parts_X, parts_y = [], []
    for split in ["train", "val", "test"]:
        parts_X.append(pd.read_parquet(os.path.join(data_dir, f"X_{split}.parquet")))
        parts_y.append(pd.read_parquet(os.path.join(data_dir, f"y_{split}.parquet")))
    X = pd.concat(parts_X).sort_index()
    y = pd.concat(parts_y).sort_index()
    X = X[~X.index.duplicated(keep="first")]
    y = y[~y.index.duplicated(keep="first")]
    feat = [c for c in X.columns if str(c).startswith("BTC__")]
    yreg = y[resolve_y_column(y, "BTC", "y_reg")].astype(float)
    return X[feat], yreg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cost-thr", type=float, default=COST)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    set_seed(42)
    Xdf, yreg = _load_full(args.data_dir)
    idx = Xdf.index
    Xvals = Xdf.values.astype(np.float32)
    fwd_full = forward_return(yreg, HORIZON).reindex(idx)
    fwd_arr = fwd_full.values.astype(float)
    print(f"Полная серия: {len(Xdf)} баров [{idx.min()} .. {idx.max()}], фич: {Xvals.shape[1]}")

    test_months = pd.period_range("2025-01", "2025-09", freq="M")
    rows = []

    for M in test_months:
        m_start = pd.Timestamp(M.start_time, tz="UTC")
        m_end = pd.Timestamp((M + 1).start_time, tz="UTC")
        te_start = int(np.searchsorted(idx.values, np.datetime64(m_start)))
        te_end = int(np.searchsorted(idx.values, np.datetime64(m_end)))
        val_start = int(np.searchsorted(idx.values, np.datetime64(m_start - pd.Timedelta(days=30))))
        if te_end <= te_start or val_start < WINDOW + 100:
            continue

        # train core [0:val_start), val [val_start:te_start)
        scaler = StandardScaler().fit(Xvals[:val_start])
        Xtr_s = scaler.transform(Xvals[:val_start]).astype(np.float32)
        Xva_s = scaler.transform(Xvals[val_start:te_start]).astype(np.float32)
        # тест с контекстом window
        Xte_s = scaler.transform(Xvals[te_start - WINDOW:te_end]).astype(np.float32)

        def ca_y(a, b):
            v = fwd_arr[a:b]
            return (np.nan_to_num(v) > args.cost_thr).astype(np.int64)
        y_tr = ca_y(0, val_start)
        y_va = ca_y(val_start, te_start)

        ds_tr = WindowDataset(Xtr_s, y_tr, WINDOW)
        ds_va = WindowDataset(Xva_s, y_va, WINDOW)
        if len(ds_va) < 50:
            continue
        ct = np.bincount(ds_tr.labels, minlength=2)
        cw = ct.sum() / (2.0 * np.maximum(ct, 1))
        crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))

        fwd_va = pd.Series(fwd_arr[val_start:te_start][WINDOW:], index=idx[val_start + WINDOW:te_start])
        idx_va_w = idx[val_start + WINDOW:te_start]

        torch.manual_seed(42)
        model = SimpleLSTM(input_size=Xvals.shape[1], hidden_size=64, num_layers=1, dropout=0.2).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        fit_classifier_sharpe(
            model, DataLoader(ds_tr, batch_size=512, shuffle=True),
            DataLoader(ds_va, batch_size=512), crit, opt,
            fwd_val=fwd_va.fillna(0.0), idx_val_w=idx_va_w,
            epochs=args.epochs, patience=4, tag=f"wf-{M}",
            periods_per_year=PER_YEAR_FWD, min_hold=MIN_HOLD, device=DEVICE)

        # предсказание месяца (окна теста дают позиции для idx[te_start:te_end])
        ds_te = WindowDataset(Xte_s, np.zeros(len(Xte_s), np.int64), WINDOW)
        _, _, p1 = _predict_labels_probs(model, DataLoader(ds_te, batch_size=512), DEVICE)
        m_idx = idx[te_start:te_end]
        nlen = min(len(p1), len(m_idx))
        pos = apply_min_holding(pd.Series((p1[:nlen] >= 0.5).astype(float), index=m_idx[:nlen]), MIN_HOLD)
        m_fwd = pd.Series(fwd_arr[te_start:te_start + nlen], index=m_idx[:nlen]).fillna(0.0)
        strat = net_returns(pos, m_fwd)
        bh = net_returns(pd.Series(1.0, index=m_idx[:nlen]), m_fwd)
        ss, sb = sharpe_ratio(strat, PER_YEAR_FWD), sharpe_ratio(bh, PER_YEAR_FWD)
        frac = 100 * (pos.values > 0).mean()
        rows.append((str(M), ss, sb, frac, float(strat.sum()), float(bh.sum())))
        print(f"[{M}] strat_sharpe={ss:.2f} bh_sharpe={sb:.2f} %long={frac:.1f} "
              f"pnl_strat={strat.sum():.4f} pnl_bh={bh.sum():.4f}")

    print("\n" + "=" * 70)
    print("WALK-FORWARD (помесячное переобучение, costs-aware):")
    print("=" * 70)
    print(f"{'месяц':<10}{'strat':>9}{'BuyHold':>9}{'% long':>9}{'выигр':>8}")
    wins = sum(1 for r in rows if r[1] > r[2])
    pnl_s = sum(r[4] for r in rows); pnl_b = sum(r[5] for r in rows)
    for m, ss, sb, frac, ps, pb in rows:
        print(f"{m:<10}{ss:>9.2f}{sb:>9.2f}{frac:>8.1f}%{'  ДА' if ss > sb else '  нет':>8}")
    print("-" * 70)
    print(f"Месяцев strat>BuyHold: {wins}/{len(rows)}")
    print(f"PnL net: strat {pnl_s:.4f} | Buy&Hold {pnl_b:.4f}")


if __name__ == "__main__":
    main()
