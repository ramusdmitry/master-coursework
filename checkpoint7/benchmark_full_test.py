"""
benchmark_full_test.py — переоценка моделей CP6 коллеги на ПОЛНОМ тесте 2025.

Грузит обученные веса (teammate_cp6/*.pt|*.pkl) и переоценивает каждую модель
на честном непрерывном тесте 2025-01..09 — БЕЗ переобучения (train коллеги
побайтово идентичен нашему, поэтому веса валидны). Память O(chunk): прогон
теста чанками с перекрытием в window (обход OOM от материализации окон).

Для каждой модели берётся её тип выхода (классификатор/политика/Q-сеть) и
сохранённые threshold/min_hold из метаданных .pt — то же, что давало исходные
числа CP6. Итог — таблица «было (тест 1 день) → стало (полный тест)».

Запуск из checkpoint7/:
    python benchmark_full_test.py \
        --data-dir /abs/data/processed \
        --weights-dir /abs/teammate_cp6
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from src.common import (
    WindowDataset,
    apply_min_holding,
    evaluate_on_forward,
    set_seed,
    window_aggregates,
)
from src.train import _prepare_data
from src.zoo import reconstruct_model

# Маппинг файла весов -> (строка в cp6_results_summary.csv, дефолтные thr/min_hold)
MODEL_SPECS = [
    # filename, имя в таблице, дефолтный threshold, дефолтный min_hold
    ("best_transformer_tuned.pt", "BestTuned transformer", 0.44, 15),
    ("ssl_masked_finetune_classifier.pt", "SSL masked", 0.40, 15),
    ("contrastive_linear_probe_classifier.pt", "Contrastive", 0.44, 15),
    ("policynet_differentiable_sharpe.pt", "SharpeLoss PolicyNet", 0.7713, 1),
    ("dqn_agent_long_flat.pt", "DQN agent", 0.5, 15),
]


def _build_cfg(data_dir: str, seed: int = 42):
    return OmegaConf.create({
        "seed": seed,
        "data": {
            "active_asset": "BTC", "data_dir": data_dir, "synthetic": False,
            "splits": {"X_train": "X_train.parquet", "X_val": "X_val.parquet",
                       "X_test": "X_test.parquet", "y_train": "y_train.parquet",
                       "y_val": "y_val.parquet", "y_test": "y_test.parquet"},
        },
        "model": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "lr": 1e-3,
                  "batch_size": 256, "epochs": 1, "patience": 5, "window": 60,
                  "horizon": 15, "min_holding": 15, "cost_bps": 7,
                  "gbdt": {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05}},
    })


@torch.no_grad()
def _infer_signal_chunked(model, kind, Xte_s, window, chunk=80000):
    """Чанковый инференс: возвращает сигнал длины N-window.

    kind='logits' -> p1 (softmax); 'policy' -> pos[0,1]; 'qvalues' -> argmax-действие.
    """
    model.eval()
    N = len(Xte_s)
    n_pos = N - window
    out = np.empty(n_pos, dtype=np.float64)
    for s in range(0, n_pos, chunk):
        e = min(s + chunk, n_pos)
        sub = Xte_s[s:window + e]
        ds = WindowDataset(sub, np.zeros(len(sub), dtype=np.int64), window)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        vals = []
        for xb, _ in loader:
            o = model(xb)
            if kind == "logits":
                v = torch.softmax(o, 1)[:, 1]
            elif kind == "policy":
                v = o
            elif kind == "qvalues":
                v = o.argmax(1).float()
            vals.append(v.cpu().numpy())
        out[s:e] = np.concatenate(vals)
    return out


def _eval_single_model(path, name, thr_def, mh_def, Xte_s, fwd_te, idx_te_w, yte, window):
    """Грузит одну модель, переоценивает на полном тесте, возвращает dict метрик."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    cls = obj["class_name"]
    hp = obj.get("hp", {}) or {"hidden": obj.get("hidden_size", 64)}
    thr = float(obj.get("threshold", thr_def))
    mh = int(obj.get("min_hold", mh_def))

    model, kind = reconstruct_model(cls, Xte_s.shape[1], hp)
    model.load_state_dict(obj["state_dict"], strict=True)

    sig = _infer_signal_chunked(model, kind, Xte_s, window)
    n = len(sig)
    if kind == "logits":
        p1 = sig
        raw = (p1 >= thr).astype(float)
    elif kind == "policy":
        p1 = None
        raw = (sig >= thr).astype(float)
    else:  # qvalues
        p1 = None
        raw = sig  # уже 0/1

    pos = apply_min_holding(pd.Series(raw, index=idx_te_w[:n]), mh)
    ev = evaluate_on_forward(pos, fwd_te, name=name,
                             prob=p1, true_label=(yte[:n] if p1 is not None else None))
    return ev


def _retrain_secondary(model, Xtr_s, ytr_tb, window):
    """Переобучает secondary GBDT на primary коллеги (если pickle несовместим)."""
    from sklearn.ensemble import GradientBoostingClassifier
    from src.common import _predict_labels_probs, meta_labels
    ds = WindowDataset(Xtr_s, ytr_tb, window)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    _, ytr_p, p1_tr = _predict_labels_probs(model, loader, torch.device("cpu"))
    acted, meta_y = meta_labels(p1_tr, ytr_p)
    agg = window_aggregates(ds)
    meta_X = np.column_stack([agg, p1_tr])
    gbdt = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                      learning_rate=0.05, random_state=42)
    mask = acted == 1
    gbdt.fit(meta_X[mask], meta_y[mask]) if mask.sum() >= 10 else gbdt.fit(meta_X, meta_y)
    return gbdt


def _eval_meta_labeling(weights_dir, Xte_s, fwd_te, idx_te_w, yte, window,
                        Xtr_s=None, ytr_tb=None):
    """Особый случай: primary LSTM + secondary GBDT (Meta-Labeling)."""
    primary_obj = torch.load(weights_dir / "metalabel_primary_lstm.pt",
                             map_location="cpu", weights_only=False)
    model, _ = reconstruct_model("SimpleLSTM", Xte_s.shape[1],
                                 {"hidden": primary_obj.get("hidden_size", 64)})
    model.load_state_dict(primary_obj["state_dict"], strict=True)

    thr, mh = 0.57, 15
    try:
        with open(weights_dir / "metalabel_secondary_gbdt.pkl", "rb") as f:
            sec = pickle.load(f)
        secondary = sec["model"]
        smeta = sec.get("meta", {})
        thr = float(smeta.get("threshold", smeta.get("meta_threshold", 0.57)))
        mh = int(smeta.get("min_hold", smeta.get("best_min_hold", 15)))
    except Exception as e:
        print(f"  secondary pickle несовместим ({str(e)[:40]}) — переобучаю GBDT на primary коллеги")
        secondary = _retrain_secondary(model, Xtr_s, ytr_tb, window)

    # чанковый прогон primary + secondary
    model.eval()
    N = len(Xte_s)
    n_pos = N - window
    sizes = np.empty(n_pos)
    p1s = np.empty(n_pos)
    for s in range(0, n_pos, 80000):
        e = min(s + 80000, n_pos)
        sub = Xte_s[s:window + e]
        ds = WindowDataset(sub, np.zeros(len(sub), dtype=np.int64), window)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        from src.common import _predict_labels_probs
        _, _, p1 = _predict_labels_probs(model, loader, torch.device("cpu"))
        agg = window_aggregates(ds)
        meta_X = np.column_stack([agg, p1])
        meta_prob = secondary.predict_proba(meta_X)[:, 1]
        size = np.where(p1 >= 0.5, meta_prob, 0.0)
        size = np.where(meta_prob >= thr, size, 0.0)
        sizes[s:e] = size
        p1s[s:e] = p1
    pos = apply_min_holding(pd.Series(sizes, index=idx_te_w[:n_pos]), mh)
    return evaluate_on_forward(pos, fwd_te, name=f"Meta-Labeling (t={thr:.2f}, mh={mh})",
                               prob=p1s, true_label=yte[:n_pos])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights-dir", required=True)
    args = parser.parse_args()

    set_seed(42)
    weights_dir = Path(args.weights_dir)
    cfg = _build_cfg(args.data_dir)

    print("=== Подготовка честных данных (полный тест 2025) ===")
    data = _prepare_data(cfg)
    window = cfg.model.window
    idx_te_w = data["idx_te_w"]
    fwd_te = data["fwd_te"]
    yte = data["yte_tb"][window: window + len(idx_te_w)]

    # scaler коллеги (модели обучены с ним); фолбэк — наш из _prepare_data
    scaler_path = weights_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            sc = pickle.load(f)
        scaler = sc["model"] if isinstance(sc, dict) and "model" in sc else sc
        Xte_s = scaler.transform(data["Xte"]).astype(np.float32)
        Xtr_s = scaler.transform(data["Xt"]).astype(np.float32)
        print(f"Использую scaler коллеги (n_features={getattr(scaler,'n_features_in_','?')})")
    else:
        Xte_s = data["Xte_s"]
        Xtr_s = data["Xt_s"]
        print("scaler коллеги не найден — использую пересчитанный")

    print(f"Тест: {len(idx_te_w)} окон, период {idx_te_w.min()} .. {idx_te_w.max()}\n")

    rows = []
    # Meta-Labeling
    try:
        print("Переоценка: Meta-Labeling ...")
        rows.append(_eval_meta_labeling(weights_dir, Xte_s, fwd_te, idx_te_w, yte, window,
                                        Xtr_s=Xtr_s, ytr_tb=data["yt_tb"]))
    except Exception as e:
        print(f"  Meta-Labeling упал: {e}")

    for fn, name, thr_def, mh_def in MODEL_SPECS:
        p = weights_dir / fn
        if not p.exists():
            print(f"  {fn} не найден — пропуск")
            continue
        try:
            print(f"Переоценка: {name} ({fn}) ...")
            rows.append(_eval_single_model(p, name, thr_def, mh_def,
                                           Xte_s, fwd_te, idx_te_w, yte, window))
        except Exception as e:
            print(f"  {name} упал: {e}")

    # Buy & Hold
    bh = pd.Series(1.0, index=idx_te_w[:len(idx_te_w)])
    rows.append(evaluate_on_forward(bh, fwd_te, name="Buy & Hold (baseline)"))

    res = pd.DataFrame(rows)
    cols = [c for c in ["Стратегия", "Sharpe gross", "Sharpe net", "PnL net (sum logret)",
                        "% времени в long", "Сделок (смен позиции)", "ROC-AUC", "Hit-rate входов"]
            if c in res.columns]
    res = res[cols].sort_values("Sharpe net", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("ПЕРЕОЦЕНКА CP6 НА ПОЛНОМ ТЕСТЕ 2025-01..09 (forward-return H=15, издержки 7bps):")
    print("=" * 90)
    print(res.to_string(index=False))

    out = Path("benchmark_full_test_results.csv")
    res.to_csv(out, index=False, encoding="utf-8")
    print(f"\nСохранено: {out.resolve()}")


if __name__ == "__main__":
    main()
