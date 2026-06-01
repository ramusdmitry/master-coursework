"""
sweep_costs.py — как побороть отрицательный Sharpe: sweep по комиссии (B) и min_holding (A).

Для нескольких моделей (веса коллеги) и Buy&Hold считает net Sharpe (1-бар) при
разных издержках cost_bps ∈ {7 taker, 1 maker, 0} и min_holding ∈ {15, 60, 240}.
Показывает, при каких условиях стратегия выходит из минуса / обгоняет BH.

    python sweep_costs.py --data-dir /abs/data/processed --weights-dir /abs/teammate_cp6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from src.common import PER_YEAR_1M, apply_min_holding, sharpe_ratio, set_seed
from src.train import _prepare_data
from src.zoo import reconstruct_model
from benchmark_full_test import _build_cfg, _infer_signal_chunked

COSTS = {"7bps(taker)": 7e-4, "1bps(maker)": 1e-4, "0bps": 0.0}
HOLDS = [15, 60, 240]
# модель: (файл, threshold по умолчанию)
MODELS = [
    ("SimpleLSTM(primary)", "metalabel_primary_lstm.pt", 0.5),
    ("Contrastive", "contrastive_linear_probe_classifier.pt", 0.44),
    ("SSL masked", "ssl_masked_finetune_classifier.pt", 0.40),
]


def net_sharpe(pos, r1, cost_per_turn):
    pos = pos.astype(float)
    gross = pos * r1
    turn = pos.diff().abs().fillna(pos.abs())
    net = gross - turn * cost_per_turn
    return sharpe_ratio(net, PER_YEAR_1M), int(turn.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weights-dir", required=True)
    args = ap.parse_args()
    set_seed(42)
    wdir = Path(args.weights_dir)
    data = _prepare_data(_build_cfg(args.data_dir))
    window = 60
    idx = data["idx_te_w"]
    with open(wdir / "scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    scaler = sc["model"] if isinstance(sc, dict) and "model" in sc else sc
    Xte_s = scaler.transform(data["Xte"]).astype(np.float32)

    rows = []
    # модели
    for name, fn, thr in MODELS:
        obj = torch.load(wdir / fn, map_location="cpu", weights_only=False)
        thr = float(obj.get("threshold", thr))
        model, kind = reconstruct_model(obj["class_name"], Xte_s.shape[1],
                                        obj.get("hp", {}) or {"hidden": obj.get("hidden_size", 64)})
        model.load_state_dict(obj["state_dict"], strict=True)
        sig = _infer_signal_chunked(model, kind, Xte_s, window)
        n = len(sig)
        r1 = pd.Series(np.asarray(data["r1_te"])[:n], index=idx[:n])
        base = pd.Series((sig >= thr).astype(float), index=idx[:n])
        for mh in HOLDS:
            pos = apply_min_holding(base, mh)
            rec = {"Модель": name, "min_hold": mh}
            ntr = None
            for ck, cv in COSTS.items():
                sh, ntr = net_sharpe(pos, r1, cv)
                rec[ck] = round(sh, 3)
            rec["сделок"] = ntr
            rows.append(rec)
        print(f"{name} готов (thr={thr})")

    # Buy & Hold (не зависит от cost/min_hold — 1 сделка)
    n = len(idx)
    r1 = pd.Series(np.asarray(data["r1_te"])[:n], index=idx[:n])
    bh = pd.Series(1.0, index=idx[:n])
    sh0, _ = net_sharpe(bh, r1, 0.0)
    rows.append({"Модель": "Buy & Hold", "min_hold": "-",
                 "7bps(taker)": round(sh0, 3), "1bps(maker)": round(sh0, 3),
                 "0bps": round(sh0, 3), "сделок": 1})

    df = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print("SWEEP: net Sharpe (1-бар) по комиссии (B) и min_holding (A), полный тест 2025")
    print("=" * 84)
    print(df.to_string(index=False))
    df.to_csv("sweep_costs_results.csv", index=False, encoding="utf-8")
    print("\nСохранено: sweep_costs_results.csv")


if __name__ == "__main__":
    main()
