"""
recompute_1bar.py — честная переоценка моделей по 1-БАРНОЙ доходности (главная: Sharpe).

Исправляет баг overlapping forward-return: оценка стратегии считается как
pos[t] * r1[t] (доходность одного следующего бара), а не pos[t] * forward_H[t]
(перекрытие → каждая минута считается до H раз, PnL завышен в ~H раз).

Главная метрика — Sharpe net (1-бар, PER_YEAR_1M). Рядом — старый forward-Sharpe
(для сравнения) и ROC-AUC (второстепенная). Веса моделей из teammate_cp6/.

    python recompute_1bar.py --data-dir /abs/data/processed --weights-dir /abs/teammate_cp6
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.common import (
    PER_YEAR_1M,
    apply_min_holding,
    evaluate_on_forward,
    evaluate_strategy,
    set_seed,
)
from src.train import _prepare_data
from src.zoo import reconstruct_model
from benchmark_full_test import (
    MODEL_SPECS, _build_cfg, _infer_signal_chunked,
    _eval_meta_labeling,  # переиспользуем для primary+secondary
)


def _eval_1bar(pos, r1_te, fwd_te, name, prob=None, yte=None):
    """Оценка: главная — Sharpe net на 1-баре (PER_YEAR_1M); рядом — forward (старый)."""
    ev1 = evaluate_strategy(pos, r1_te, PER_YEAR_1M, name=name, prob=prob, true_label=yte)
    evf = evaluate_on_forward(pos, fwd_te, name=name)
    return {
        "Стратегия": name,
        "Sharpe net (1бар)": ev1.get("Sharpe net"),
        "Sharpe net (fwd, старый)": evf.get("Sharpe net"),
        "PnL (1бар)": ev1.get("PnL net (sum logret)"),
        "% в long": ev1.get("% времени в long"),
        "Сделок": ev1.get("Сделок (смен позиции)"),
        "ROC-AUC": ev1.get("ROC-AUC"),
    }


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
    r1_te = data["r1_te"]
    fwd_te = data["fwd_te"]
    yte_full = data["yte_tb"]

    # scaler коллеги
    with open(wdir / "scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    scaler = sc["model"] if isinstance(sc, dict) and "model" in sc else sc
    Xte_s = scaler.transform(data["Xte"]).astype(np.float32)
    Xtr_s = scaler.transform(data["Xt"]).astype(np.float32)
    print(f"Тест: {len(idx)} окон, период {idx.min()}..{idx.max()}\n")

    rows = []

    # Meta-Labeling (primary+secondary) — берём позиции через переиспользуемую функцию,
    # но оценим 1-баром: восстановим позиции тем же путём, что в benchmark.
    try:
        # _eval_meta_labeling возвращает forward-оценку; нам нужны позиции — повторим инференс
        import torch as _t
        primary_obj = _t.load(wdir / "metalabel_primary_lstm.pt", map_location="cpu", weights_only=False)
        model, _ = reconstruct_model("SimpleLSTM", Xte_s.shape[1], {"hidden": primary_obj.get("hidden_size", 64)})
        model.load_state_dict(primary_obj["state_dict"], strict=True)
        from benchmark_full_test import _retrain_secondary
        from src.common import window_aggregates, WindowDataset, _predict_labels_probs
        from torch.utils.data import DataLoader
        secondary = _retrain_secondary(model, Xtr_s, data["yt_tb"], window)
        thr, mh = 0.57, 15
        N = len(Xte_s); n = N - window
        sizes = np.empty(n); p1s = np.empty(n)
        for s in range(0, n, 80000):
            e = min(s + 80000, n)
            ds = WindowDataset(Xte_s[s:window + e], np.zeros(window + (e - s), np.int64), window)
            _, _, p1 = _predict_labels_probs(model, DataLoader(ds, batch_size=512), torch.device("cpu"))
            agg = window_aggregates(ds)
            mp = secondary.predict_proba(np.column_stack([agg, p1]))[:, 1]
            size = np.where(p1 >= 0.5, mp, 0.0); size = np.where(mp >= thr, size, 0.0)
            sizes[s:e] = size; p1s[s:e] = p1
        pos = apply_min_holding(pd.Series(sizes, index=idx[:n]), mh)
        rows.append(_eval_1bar(pos, r1_te, fwd_te, "Meta-Labeling", p1s, yte_full[window:window + n]))
        print("Meta-Labeling готов")
    except Exception as ex:
        print("Meta-Labeling упал:", ex)

    # Зоопарк
    for fn, name, thr_def, mh_def in MODEL_SPECS:
        p = wdir / fn
        if not p.exists():
            continue
        obj = torch.load(p, map_location="cpu", weights_only=False)
        cls = obj["class_name"]; hp = obj.get("hp", {}) or {"hidden": obj.get("hidden_size", 64)}
        thr = float(obj.get("threshold", thr_def)); mh = int(obj.get("min_hold", mh_def))
        model, kind = reconstruct_model(cls, Xte_s.shape[1], hp)
        model.load_state_dict(obj["state_dict"], strict=True)
        sig = _infer_signal_chunked(model, kind, Xte_s, window)
        nn_ = len(sig)
        if kind == "logits":
            p1 = sig; raw = (p1 >= thr).astype(float)
        elif kind == "policy":
            p1 = None; raw = (sig >= thr).astype(float)
        else:
            p1 = None; raw = sig
        pos = apply_min_holding(pd.Series(raw, index=idx[:nn_]), mh)
        rows.append(_eval_1bar(pos, r1_te, fwd_te, name, p1,
                               yte_full[window:window + nn_] if p1 is not None else None))
        print(f"{name} готов")

    # Buy & Hold
    bh = pd.Series(1.0, index=idx)
    rows.append(_eval_1bar(bh, r1_te, fwd_te, "Buy & Hold"))

    res = pd.DataFrame(rows).sort_values("Sharpe net (1бар)", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 100)
    print("ПЕРЕОЦЕНКА ПО 1-БАРНОЙ ДОХОДНОСТИ (главная: Sharpe net 1бар), полный тест 2025:")
    print("=" * 100)
    print(res.to_string(index=False))
    res.to_csv("recompute_1bar_results.csv", index=False, encoding="utf-8")
    print("\nСохранено: recompute_1bar_results.csv")


if __name__ == "__main__":
    main()
