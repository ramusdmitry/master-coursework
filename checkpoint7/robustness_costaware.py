"""
robustness_costaware.py — устойчивость costs-aware модели по под-периодам.

Обучает costs-aware primary (таргет = net-return > cost_thr) и считает Sharpe net
ПОМЕСЯЧНО на тесте, рядом с Buy&Hold. Если Sharpe положителен в большинстве
месяцев — edge реальный; если весь вклад из 1-2 месяцев — артефакт.

    python robustness_costaware.py --data-dir /abs/data/processed_of --cost-thr 0.0007
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.common import (
    PER_YEAR_FWD,
    SimpleLSTM,
    WindowDataset,
    apply_min_holding,
    fit_classifier_sharpe,
    net_returns,
    set_seed,
    sharpe_ratio,
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


def _ca_labels(n_total, fwd, window, cost_thr):
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
    ap.add_argument("--cost-thr", type=float, default=0.0007)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    set_seed(42)
    data = _prepare_data(_cfg(args.data_dir, args.epochs))
    window = 60

    y_tr = _ca_labels(len(data["Xt_s"]), data["fwd_tr"], window, args.cost_thr)
    y_va = _ca_labels(len(data["Xv_s"]), data["fwd_va"], window, args.cost_thr)
    ds_tr = WindowDataset(data["Xt_s"], y_tr, window)
    ds_va = WindowDataset(data["Xv_s"], y_va, window)

    ct = np.bincount(ds_tr.labels, minlength=2)
    cw = ct.sum() / (2.0 * np.maximum(ct, 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))
    ld_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=512, shuffle=False)

    torch.manual_seed(42)
    model = SimpleLSTM(input_size=data["n_features"], hidden_size=64, num_layers=1, dropout=0.2).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Обучение costs-aware (cost_thr={args.cost_thr}), device={DEVICE}...")
    fit_classifier_sharpe(model, ld_tr, ld_va, crit, opt,
                          fwd_val=data["fwd_va"], idx_val_w=data["idx_va_w"],
                          epochs=args.epochs, patience=5, tag="ca",
                          periods_per_year=PER_YEAR_FWD, min_hold=15, device=DEVICE)

    p1 = _infer_p1(model, data["Xte_s"], window)
    n = len(p1)
    idx = data["idx_te_w"][:n]
    pos = apply_min_holding(pd.Series((p1 >= 0.5).astype(float), index=idx), 15)
    fwd = pd.Series(data["fwd_te"].values[:n], index=idx)

    strat_net = net_returns(pos, fwd)
    bh_net = net_returns(pd.Series(1.0, index=idx), fwd)

    # помесячный Sharpe
    months = pd.Series(idx).dt.tz_convert(None).dt.to_period("M").values
    df = pd.DataFrame({"strat": strat_net.values, "bh": bh_net.values,
                       "pos": pos.values, "month": months})
    print("\n" + "=" * 64)
    print("ПОМЕСЯЧНЫЙ Sharpe net (costs-aware strat vs Buy&Hold):")
    print("=" * 64)
    print(f"{'месяц':<10}{'strat':>10}{'BuyHold':>10}{'% long':>9}{'выигр':>8}")
    wins = 0
    n_months = 0
    for m, g in df.groupby("month"):
        ss = sharpe_ratio(g["strat"], PER_YEAR_FWD)
        sb = sharpe_ratio(g["bh"], PER_YEAR_FWD)
        frac = 100 * (g["pos"] > 0).mean()
        win = ss > sb
        wins += int(win); n_months += 1
        print(f"{str(m):<10}{ss:>10.2f}{sb:>10.2f}{frac:>8.1f}%{'  ДА' if win else '  нет':>8}")
    full_s = sharpe_ratio(df["strat"], PER_YEAR_FWD)
    full_b = sharpe_ratio(df["bh"], PER_YEAR_FWD)
    print("-" * 64)
    print(f"{'ВЕСЬ ТЕСТ':<10}{full_s:>10.2f}{full_b:>10.2f}")
    print(f"\nМесяцев strat>BuyHold: {wins}/{n_months}")
    print(f"PnL net strat: {df['strat'].sum():.4f} | Buy&Hold: {df['bh'].sum():.4f}")


if __name__ == "__main__":
    main()
