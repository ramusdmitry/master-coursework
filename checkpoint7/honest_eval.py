"""
honest_eval.py — честная оценка Meta-Labeling на ПОЛНОМ тесте 2025-01..09.

Зачем отдельный скрипт, а не train.py:
- MetaLabelingModel.fit() материализует окна train+val+test одновременно
  (WindowDataset.__init__ делает np.stack всех окон). На полном тесте (393k окон,
  ~7.3 ГБ) это вызывает OOM. Здесь модель обучается БЕЗ теста (X_te=None), а тест
  прогоняется чанками с перекрытием в window баров — математически тот же результат,
  но память O(chunk).

Контролируемое сравнение с CP6: train/val побайтово идентичны данным коллеги, на
которых учился CP6; отличается ТОЛЬКО длина теста (1 день в CP6 → 9 месяцев здесь).

Запуск из checkpoint7/:
    python honest_eval.py --data-dir /abs/path/to/data/processed --epochs 30
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.common import (
    PER_YEAR_FWD,
    WindowDataset,
    apply_min_holding,
    evaluate_on_forward,
    set_seed,
    window_aggregates,
    _predict_labels_probs,
)
from src.model import MetaLabelingModel
from src.train import _prepare_data


def _build_cfg(data_dir: str, epochs: int, seed: int) -> OmegaConf:
    """Минимальный cfg для _prepare_data и модели (значения по умолчанию CP6)."""
    return OmegaConf.create(
        {
            "seed": seed,
            "data": {
                "active_asset": "BTC",
                "data_dir": data_dir,
                "synthetic": False,
                "splits": {
                    "X_train": "X_train.parquet",
                    "X_val": "X_val.parquet",
                    "X_test": "X_test.parquet",
                    "y_train": "y_train.parquet",
                    "y_val": "y_val.parquet",
                    "y_test": "y_test.parquet",
                },
            },
            "model": {
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.2,
                "lr": 1e-3,
                "batch_size": 256,
                "epochs": epochs,
                "patience": 5,
                "window": 60,
                "horizon": 15,
                "min_holding": 15,
                "cost_bps": 7,
                "gbdt": {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05},
            },
        }
    )


def _predict_test_chunked(model: MetaLabelingModel, Xte_s: np.ndarray, chunk_pos: int = 80000):
    """Прогон полного теста чанками. Возвращает (size, p1, y_dummy_ignored).

    Для глобальных позиций p в [window, N) нарезаем перекрывающиеся срезы Xte_s,
    чтобы каждый чанк давал ровно нужные позиции без потерь на границах.
    Повторяет логику MetaLabelingModel.predict_positions, но без min_holding
    (его применяем один раз на полном ряду — иначе границы чанков рвут удержание).
    """
    window = model.window
    N = len(Xte_s)
    n_pos = N - window
    sizes = np.empty(n_pos, dtype=np.float64)
    p1s = np.empty(n_pos, dtype=np.float64)

    for s in range(0, n_pos, chunk_pos):
        e = min(s + chunk_pos, n_pos)
        x0, x1 = s, window + e  # срез даёт позиции [s, e)
        sub = Xte_s[x0:x1]
        dummy_y = np.zeros(len(sub), dtype=np.int64)
        ds = WindowDataset(sub, dummy_y, window)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        _, _, p1 = _predict_labels_probs(model.primary, loader, model.device)
        agg = window_aggregates(ds)
        meta_X = np.column_stack([agg, p1])
        meta_prob = model.secondary.predict_proba(meta_X)[:, 1]
        size = np.where(p1 >= 0.5, meta_prob, 0.0)
        size = np.where(meta_prob >= model.meta_threshold, size, 0.0)
        sizes[s:e] = size
        p1s[s:e] = p1
        print(f"  чанк позиций [{s}:{e}] готов ({e}/{n_pos})")

    return sizes, p1s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = _build_cfg(args.data_dir, args.epochs, args.seed)

    print("=== Подготовка данных (triple-barrier, scaler, forward-return) ===")
    data = _prepare_data(cfg)
    print(f"train={len(data['Xt_s'])}, val={len(data['Xv_s'])}, test={len(data['Xte_s'])}")
    print(f"test окон: {len(data['idx_te_w'])}, период fwd_te: "
          f"{data['idx_te_w'].min()} .. {data['idx_te_w'].max()}")

    print("\n=== Обучение Meta-Labeling (БЕЗ теста — экономия памяти) ===")
    model = MetaLabelingModel(
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        lr=cfg.model.lr,
        batch_size=cfg.model.batch_size,
        epochs=cfg.model.epochs,
        patience=cfg.model.patience,
        window=cfg.model.window,
        horizon=cfg.model.horizon,
        min_holding=cfg.model.min_holding,
        cost_bps=cfg.model.cost_bps,
        gbdt_params={
            "n_estimators": cfg.model.gbdt.n_estimators,
            "max_depth": cfg.model.gbdt.max_depth,
            "learning_rate": cfg.model.gbdt.learning_rate,
            "random_state": cfg.seed,
        },
    )
    metrics = model.fit(
        X_tr=data["Xt_s"], y_tr=data["yt_tb"],
        X_va=data["Xv_s"], y_va=data["yv_tb"],
        fwd_va=data["fwd_va"], idx_va_w=data["idx_va_w"],
        X_te=None,  # ключ: не материализуем тест в fit()
        seed=cfg.seed,
    )
    print("Метрики train/val:", {k: round(v, 4) for k, v in metrics.items()})
    print(f"meta_threshold={model.meta_threshold:.3f}, min_hold={model.best_min_hold}")

    print("\n=== Прогон ПОЛНОГО теста чанками ===")
    sizes, p1_te = _predict_test_chunked(model, data["Xte_s"])

    idx_te_w = data["idx_te_w"]
    n_pos = len(sizes)
    raw_pos = pd.Series(sizes, index=idx_te_w[:n_pos])
    pos = apply_min_holding(raw_pos, model.best_min_hold)
    fwd_te = data["fwd_te"]

    # истинные метки теста для ROC-AUC/hit-rate (выровнены по окнам)
    yte = data["yte_tb"][model.window: model.window + n_pos]

    meta_eval = evaluate_on_forward(
        pos, fwd_te, name="Meta-Labeling (ПОЛНЫЙ тест 2025)",
        prob=p1_te, true_label=yte,
    )
    bh_pos = pd.Series(1.0, index=idx_te_w[:n_pos])
    bh_eval = evaluate_on_forward(bh_pos, fwd_te, name="Buy & Hold (baseline)")

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ НА ПОЛНОМ ТЕСТЕ (2025-01..09, forward-return H=15):")
    print("=" * 70)
    res = pd.DataFrame([meta_eval, bh_eval])
    cols = [c for c in ["Стратегия", "Sharpe gross", "Sharpe net",
                        "PnL net (sum logret)", "% времени в long",
                        "Сделок (смен позиции)", "ROC-AUC", "Hit-rate входов"]
            if c in res.columns]
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
