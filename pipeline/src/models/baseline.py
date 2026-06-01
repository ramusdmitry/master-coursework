"""Наивный baseline для прогноза временного ряда.

NaiveLastValueModel — повторяет последнее наблюдённое значение target на весь
горизонт. Это честный baseline для финансовых рядов (random walk): для цены
закрытия прогноз «завтра = сегодня» крайне трудно побить по RMSE.

Интерфейс (как в `runner.py`):
    model.fit(train_df, target_col=...)
    model.predict(horizon=...)  -> np.ndarray (horizon,)
    model.metadata()            -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveLastValueModel:
    """Прогноз = последнее значение train, повторённое на горизонт."""

    name = "naive_last_value"

    def __init__(self) -> None:
        self._last_value: float | None = None
        self._target_col: str | None = None
        self._n_train: int = 0

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "NaiveLastValueModel":
        """Запоминает последнее значение целевой колонки на train."""
        if target_col not in train_df.columns:
            raise KeyError(f"Нет колонки '{target_col}' в train_df.")
        self._target_col = target_col
        self._n_train = len(train_df)
        self._last_value = float(train_df[target_col].iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Возвращает массив из horizon одинаковых значений (last value)."""
        if self._last_value is None:
            raise RuntimeError("Сначала вызовите fit().")
        return np.full(int(horizon), self._last_value, dtype=float)

    def metadata(self) -> dict:
        """Метаданные модели для отчёта/трекинга."""
        return {
            "model": self.name,
            "target_col": self._target_col,
            "last_value": self._last_value,
            "n_train": self._n_train,
        }
