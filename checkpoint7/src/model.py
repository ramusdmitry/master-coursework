"""
Модуль model.py — Meta-Labeling модель для чекпоинта 7.

Содержит:
- MetaLabelingModel: обучение primary SimpleLSTM + secondary GradientBoostingClassifier,
  предсказание позиций, сохранение/загрузка.
- MetaLabelingPyfunc: обёртка mlflow.pyfunc.PythonModel для логирования
  всей связки в MLflow Model Registry.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.common import (
    COST_PER_TURN,
    HORIZON,
    MIN_HOLDING,
    PER_YEAR_FWD,
    SEED,
    VOL_WINDOW_TB,
    PT_MULT,
    SL_MULT,
    VERTICAL_TB,
    SimpleLSTM,
    WindowDataset,
    apply_min_holding,
    evaluate_on_forward,
    fit_classifier_sharpe,
    forward_return,
    meta_labels,
    net_returns,
    save_sklearn_model,
    save_torch_model,
    sharpe_ratio,
    triple_barrier_labels,
    tune_threshold,
    window_aggregates,
    _predict_labels_probs,
)


class MetaLabelingModel:
    """Двухуровневая Meta-Labeling модель (López de Prado).

    Уровень 1 (primary): SimpleLSTM обучается предсказывать сторону сделки
    (long / flat) по оконным признакам. Используется early stopping по val
    net-Sharpe (исправление причины #2 из CP6).

    Уровень 2 (secondary): GradientBoostingClassifier обучается предсказывать
    успешность сделки primary-модели. Признаки вторичной модели —
    агрегаты окна (mean, std, last_step) + вероятность primary.

    Итоговая позиция: входим в long ТОЛЬКО если primary сигнализирует AND
    вторичная модель уверена (meta_prob >= meta_threshold).
    Размер позиции = meta_prob (position sizing).

    Параметры:
        hidden_size  -- размер скрытого состояния LSTM.
        num_layers   -- число слоёв LSTM.
        dropout      -- dropout в LSTM и перед FC.
        lr           -- learning rate Adam.
        batch_size   -- размер батча.
        epochs       -- максимальное число эпох.
        patience     -- patience для early stopping.
        window       -- длина окна (баров).
        horizon      -- горизонт forward-return (баров).
        min_holding  -- минимальное удержание позиции (баров).
        cost_bps     -- транзакционные издержки (базисные пункты).
        gbdt_params  -- гиперпараметры GradientBoostingClassifier.
        device       -- torch.device (авто если None).
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 30,
        patience: int = 5,
        window: int = 60,
        horizon: int = HORIZON,
        min_holding: int = MIN_HOLDING,
        cost_bps: float = 7.0,
        gbdt_params: Optional[dict] = None,
        device=None,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.window = window
        self.horizon = horizon
        self.min_holding = min_holding
        self.cost_per_turn = cost_bps / 1e4
        self.periods_per_year = (60.0 * 24.0 * 365.0) / horizon

        # Гиперпараметры GBDT
        if gbdt_params is None:
            gbdt_params = {
                "n_estimators": 150,
                "max_depth": 3,
                "learning_rate": 0.05,
                "random_state": SEED,
            }
        self.gbdt_params = gbdt_params

        # Устройство для PyTorch
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # Компоненты модели (инициализируются в fit)
        self.primary: Optional[SimpleLSTM] = None
        self.secondary: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.n_features: Optional[int] = None
        self.meta_threshold: float = 0.5
        self.best_min_hold: int = min_holding
        self.best_val_sharpe: float = -1e18

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
        fwd_va: pd.Series,
        idx_va_w,
        X_te: Optional[np.ndarray] = None,
        y_te: Optional[np.ndarray] = None,
        fwd_te: Optional[pd.Series] = None,
        idx_te_w=None,
        seed: int = SEED,
    ) -> dict:
        """Обучает primary SimpleLSTM и secondary GradientBoostingClassifier.

        Аргументы:
            X_tr     -- признаки train (N_tr, n_features), уже нормализованные.
            y_tr     -- метки train (N_tr,), int 0/1.
            X_va     -- признаки val.
            y_va     -- метки val.
            fwd_va   -- forward-return Series на val (для early stopping и подбора порога).
            idx_va_w -- индекс val-окон.
            X_te     -- признаки test (опционально, для итоговых метрик).
            y_te     -- метки test.
            fwd_te   -- forward-return Series на test.
            idx_te_w -- индекс test-окон.
            seed     -- seed для инициализации.

        Возвращает:
            dict метрик train/val/test.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.n_features = X_tr.shape[1]

        # Датасеты и загрузчики
        ds_tr = WindowDataset(X_tr, y_tr, self.window)
        ds_va = WindowDataset(X_va, y_va, self.window)

        # Веса классов (борьба с дисбалансом)
        ct = np.bincount(ds_tr.labels, minlength=2)
        cw = ct.sum() / (2.0 * np.maximum(ct, 1))
        w_tensor = torch.tensor(cw, dtype=torch.float32).to(self.device)

        ld_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)
        ld_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False)

        # --- Primary: SimpleLSTM ---
        self.primary = SimpleLSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=w_tensor)
        optimizer = optim.Adam(self.primary.parameters(), lr=self.lr)

        print("Обучение primary SimpleLSTM (early stopping по val net-Sharpe)...")
        self.best_val_sharpe = fit_classifier_sharpe(
            self.primary,
            ld_tr,
            ld_va,
            criterion,
            optimizer,
            fwd_val=fwd_va,
            idx_val_w=idx_va_w,
            epochs=self.epochs,
            patience=self.patience,
            tag="Primary",
            periods_per_year=self.periods_per_year,
            min_hold=self.min_holding,
            device=self.device,
        )

        # Предсказания primary на train/val
        ld_tr_eval = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=False)
        _, y_tr_p, p1_tr = _predict_labels_probs(self.primary, ld_tr_eval, self.device)
        _, y_va_p, p1_va = _predict_labels_probs(self.primary, ld_va, self.device)

        # --- Мета-метки ---
        acted_tr, meta_y_tr = meta_labels(p1_tr, y_tr_p)
        acted_va, meta_y_va = meta_labels(p1_va, y_va_p)

        # Признаки secondary: mean | std | last_step | primary_prob
        agg_tr = window_aggregates(ds_tr)
        agg_va = window_aggregates(ds_va)
        meta_X_tr = np.column_stack([agg_tr, p1_tr])
        meta_X_va = np.column_stack([agg_va, p1_va])

        # Обучаем secondary только на примерах, где primary решил действовать
        mask_tr = acted_tr == 1
        print(f"Обучение secondary GBDT на {mask_tr.sum()} примерах (acted==1)...")
        self.secondary = GradientBoostingClassifier(**self.gbdt_params)
        if mask_tr.sum() >= 10:
            self.secondary.fit(meta_X_tr[mask_tr], meta_y_tr[mask_tr])
        else:
            # Слишком мало примеров — обучаем на всей выборке
            print("Предупреждение: мало примеров с acted==1, обучаем на полной train.")
            self.secondary.fit(meta_X_tr, meta_y_tr)

        # Подбор порога мета-вероятности по val net-Sharpe
        meta_prob_va = self.secondary.predict_proba(meta_X_va)[:, 1]
        combined_va = np.where(p1_va >= 0.5, meta_prob_va, 0.0)
        bp = tune_threshold(
            combined_va,
            fwd_va,
            idx_va_w,
            periods_per_year=self.periods_per_year,
            thr_grid=np.linspace(0.30, 0.70, 41),
            hold_options=[1, self.min_holding],
            mode="simple",
        )
        self.meta_threshold = bp["threshold"]
        self.best_min_hold = bp["min_hold"]
        print(
            f"Подобранный meta_threshold={self.meta_threshold:.3f}, "
            f"min_hold={self.best_min_hold}"
        )

        # Собираем метрики
        metrics: dict = {}

        # -- train --
        try:
            metrics["train_roc_auc"] = float(roc_auc_score(y_tr_p, p1_tr))
        except ValueError:
            metrics["train_roc_auc"] = float("nan")
        metrics["train_accuracy"] = float(
            accuracy_score(y_tr_p, (p1_tr >= 0.5).astype(int))
        )
        in_pos_tr = p1_tr >= 0.5
        if in_pos_tr.sum() > 0:
            metrics["train_hit_rate"] = float(
                (y_tr_p[in_pos_tr] == 1).mean()
            )
        else:
            metrics["train_hit_rate"] = float("nan")

        # -- val --
        try:
            metrics["val_roc_auc"] = float(roc_auc_score(y_va_p, p1_va))
        except ValueError:
            metrics["val_roc_auc"] = float("nan")
        metrics["val_accuracy"] = float(
            accuracy_score(y_va_p, (p1_va >= 0.5).astype(int))
        )
        metrics["val_net_sharpe"] = float(self.best_val_sharpe)

        # -- test (опционально) --
        if X_te is not None and y_te is not None and fwd_te is not None:
            ds_te = WindowDataset(X_te, y_te, self.window)
            ld_te = DataLoader(ds_te, batch_size=self.batch_size, shuffle=False)
            pred_te, y_te_p, p1_te = _predict_labels_probs(
                self.primary, ld_te, self.device
            )
            agg_te = window_aggregates(ds_te)
            meta_X_te = np.column_stack([agg_te, p1_te])
            meta_prob_te = self.secondary.predict_proba(meta_X_te)[:, 1]
            size_te = np.where(p1_te >= 0.5, meta_prob_te, 0.0)
            size_te = np.where(meta_prob_te >= self.meta_threshold, size_te, 0.0)
            idx_te_ = idx_te_w if idx_te_w is not None else np.arange(len(size_te))
            raw_pos = pd.Series(size_te, index=idx_te_[: len(size_te)])
            pos_te = apply_min_holding(raw_pos, self.best_min_hold)
            te_eval = evaluate_on_forward(
                pos_te, fwd_te, name="MetaLabeling-test",
                prob=p1_te, true_label=y_te_p,
            )
            metrics["test_net_sharpe"] = float(te_eval.get("Sharpe net", float("nan")))
            metrics["test_pnl"] = float(te_eval.get("PnL net (sum logret)", float("nan")))
            try:
                metrics["test_roc_auc"] = float(roc_auc_score(y_te_p, p1_te))
            except ValueError:
                metrics["test_roc_auc"] = float("nan")
            metrics["test_accuracy"] = float(
                accuracy_score(y_te_p, (p1_te >= 0.5).astype(int))
            )
            in_pos_te = p1_te >= 0.5
            if in_pos_te.sum() > 0:
                metrics["test_hit_rate"] = float(
                    (y_te_p[in_pos_te] == 1).mean()
                )
            else:
                metrics["test_hit_rate"] = float("nan")

        return metrics

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    def predict_positions(
        self,
        X: np.ndarray,
        apply_holding: bool = True,
    ) -> np.ndarray:
        """Предсказывает позиции для массива признаков X.

        Шаги:
        1. Строит WindowDataset (window баров).
        2. Primary предсказывает вероятность long.
        3. Secondary фильтрует ложные сигналы.
        4. Применяет min_holding.

        Аргументы:
            X              -- массив признаков (N, n_features).
            apply_holding  -- применять ли min_holding.

        Возвращает:
            np.ndarray позиций (N - window,) в диапазоне [0, 1].
        """
        if self.primary is None or self.secondary is None:
            raise RuntimeError("Сначала вызовите fit() или load().")

        # Синтетические метки (не нужны для предсказания)
        dummy_y = np.zeros(len(X), dtype=np.int64)
        ds = WindowDataset(X, dummy_y, self.window)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        _, _, p1 = _predict_labels_probs(self.primary, loader, self.device)
        agg = window_aggregates(ds)
        meta_X = np.column_stack([agg, p1])
        meta_prob = self.secondary.predict_proba(meta_X)[:, 1]
        size = np.where(p1 >= 0.5, meta_prob, 0.0)
        size = np.where(meta_prob >= self.meta_threshold, size, 0.0)
        if apply_holding:
            pos_series = pd.Series(size)
            pos_series = apply_min_holding(pos_series, self.best_min_hold)
            return pos_series.values
        return size

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Сохраняет оба компонента модели и метаданные в директорию.

        Файлы:
            primary.pt   -- state_dict primary SimpleLSTM;
            secondary.pkl -- pickle secondary GradientBoostingClassifier;
            meta.pkl      -- метаданные (пороги, гиперпараметры).

        Аргументы:
            directory -- директория для сохранения.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Primary
        save_torch_model(
            self.primary,
            directory / "primary.pt",
            extra={
                "n_features": self.n_features,
                "window": self.window,
                "horizon": self.horizon,
            },
        )

        # Secondary
        save_sklearn_model(
            self.secondary,
            directory / "secondary.pkl",
            extra={
                "feature_layout": "window_mean | window_std | last_step | primary_prob",
                "meta_threshold": self.meta_threshold,
                "best_min_hold": self.best_min_hold,
            },
        )

        # Метаданные
        meta = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "window": self.window,
            "horizon": self.horizon,
            "min_holding": self.min_holding,
            "meta_threshold": self.meta_threshold,
            "best_min_hold": self.best_min_hold,
            "best_val_sharpe": self.best_val_sharpe,
            "n_features": self.n_features,
            "gbdt_params": self.gbdt_params,
        }
        with open(directory / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        print(f"[MetaLabelingModel] Сохранено в {directory}")

    @classmethod
    def load(cls, directory: Path, device=None) -> "MetaLabelingModel":
        """Загружает модель из директории (созданной методом save).

        Аргументы:
            directory -- директория с файлами primary.pt / secondary.pkl / meta.pkl.
            device    -- torch.device (авто если None).

        Возвращает:
            Экземпляр MetaLabelingModel с восстановленными компонентами.
        """
        directory = Path(directory)

        # Метаданные
        with open(directory / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
            lr=meta["lr"],
            batch_size=meta["batch_size"],
            epochs=meta["epochs"],
            patience=meta["patience"],
            window=meta["window"],
            horizon=meta["horizon"],
            min_holding=meta["min_holding"],
            gbdt_params=meta.get("gbdt_params"),
            device=device,
        )
        instance.meta_threshold = meta["meta_threshold"]
        instance.best_min_hold = meta["best_min_hold"]
        instance.best_val_sharpe = meta.get("best_val_sharpe", -1e18)
        instance.n_features = meta["n_features"]

        # Primary
        payload = torch.load(directory / "primary.pt", map_location=instance.device)
        instance.primary = SimpleLSTM(
            input_size=meta["n_features"],
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
        ).to(instance.device)
        instance.primary.load_state_dict(payload["state_dict"])
        instance.primary.eval()

        # Secondary
        with open(directory / "secondary.pkl", "rb") as f:
            sk_payload = pickle.load(f)
        instance.secondary = sk_payload["model"]

        print(f"[MetaLabelingModel] Загружена из {directory}")
        return instance


# =============================================================================
# pyfunc-обёртка для MLflow Model Registry
# =============================================================================


class MetaLabelingPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc-обёртка для Meta-Labeling модели.

    Позволяет логировать всю связку (primary LSTM + secondary GBDT)
    как единую mlflow.pyfunc-модель. После логирования модель можно
    загружать через mlflow.pyfunc.load_model() или по alias из Registry.

    Интерфейс predict():
    - Вход: pd.DataFrame с числовыми столбцами признаков (N, n_features).
    - Выход: pd.DataFrame с колонками 'position' и 'primary_prob'.

    Загрузка артефактов выполняется в load_context() из поддиректорий
    artifacts/primary.pt и artifacts/secondary.pkl.
    """

    def load_context(self, context) -> None:
        """Загружает компоненты из MLflow artifacts при инициализации.

        Аргументы:
            context -- mlflow.pyfunc.PythonModelContext с путями артефактов.
        """
        # Метаданные
        meta_path = context.artifacts["meta"]
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self._window = meta["window"]
        self._horizon = meta["horizon"]
        self._meta_threshold = meta["meta_threshold"]
        self._best_min_hold = meta["best_min_hold"]
        self._n_features = meta["n_features"]
        self._hidden_size = meta["hidden_size"]
        self._num_layers = meta["num_layers"]
        self._dropout = meta["dropout"]
        self._periods_per_year = (60.0 * 24.0 * 365.0) / self._horizon
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Primary LSTM
        primary_path = context.artifacts["primary"]
        payload = torch.load(primary_path, map_location=self._device)
        self._primary = SimpleLSTM(
            input_size=self._n_features,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        ).to(self._device)
        self._primary.load_state_dict(payload["state_dict"])
        self._primary.eval()

        # Secondary GBDT
        secondary_path = context.artifacts["secondary"]
        with open(secondary_path, "rb") as f:
            sk_payload = pickle.load(f)
        self._secondary = sk_payload["model"]

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Предсказывает позиции для входного DataFrame.

        Аргументы:
            context     -- mlflow.pyfunc.PythonModelContext.
            model_input -- pd.DataFrame (N, n_features), числовые признаки.

        Возвращает:
            pd.DataFrame с колонками:
                'position'     -- итоговый размер позиции [0, 1];
                'primary_prob' -- вероятность long от primary-модели.
        """
        X = model_input.values.astype(np.float32)
        dummy_y = np.zeros(len(X), dtype=np.int64)
        ds = WindowDataset(X, dummy_y, self._window)
        if len(ds) == 0:
            return pd.DataFrame({"position": [], "primary_prob": []})
        from torch.utils.data import DataLoader as _DataLoader
        loader = _DataLoader(ds, batch_size=256, shuffle=False)
        _, _, p1 = _predict_labels_probs(self._primary, loader, self._device)
        agg = window_aggregates(ds)
        meta_X = np.column_stack([agg, p1])
        meta_prob = self._secondary.predict_proba(meta_X)[:, 1]
        size = np.where(p1 >= 0.5, meta_prob, 0.0)
        size = np.where(meta_prob >= self._meta_threshold, size, 0.0)
        pos_series = apply_min_holding(pd.Series(size), self._best_min_hold)
        return pd.DataFrame(
            {"position": pos_series.values, "primary_prob": p1}
        )
