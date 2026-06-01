"""Адаптер AutoTS для пайплайна (пункт 1 плана: «посмотреть, что даёт auto-ts»).

Обёртка приводит AutoTS к единому интерфейсу пайплайна (fit/predict/metadata).

Важно: `import autots` выполняется ЛЕНИВО внутри fit(), чтобы модуль
импортировался даже без установленного AutoTS (тяжёлая зависимость). Если
AutoTS недоступен или обучение упало — модель деградирует до наивного
прогноза (последнее значение) и помечает это флагом used_fallback=True,
чтобы пайплайн оставался запускаемым end-to-end.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class AutoTSModel:
    """Адаптер AutoTS под интерфейс пайплайна.

    Параметры (из configs/base.yaml -> models.autots):
        forecast_length -- горизонт прогноза, заданный конфигом.
        frequency       -- частота ('infer' или, например, 'H').
        ensemble        -- стратегия ансамблирования AutoTS.
        model_list      -- набор моделей ('superfast', 'fast', ...).
        max_generations -- число генераций генетического поиска.
        num_validations -- число валидационных срезов.
        timestamp_col   -- имя колонки времени во входном DataFrame.
    """

    name = "autots"

    def __init__(
        self,
        forecast_length: int,
        frequency: str = "infer",
        ensemble: str = "simple",
        model_list: str = "superfast",
        max_generations: int = 3,
        num_validations: int = 2,
        timestamp_col: str = "timestamp",
    ) -> None:
        self.forecast_length = int(forecast_length)
        self.frequency = frequency
        self.ensemble = ensemble
        self.model_list = model_list
        self.max_generations = int(max_generations)
        self.num_validations = int(num_validations)
        self.timestamp_col = timestamp_col

        self._model = None
        self._target_col: str | None = None
        self._fallback_value: float | None = None
        self._used_fallback = False
        self._best_model_name: str | None = None

    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "AutoTSModel":
        """Обучает AutoTS на одной серии. При недоступности — fallback на naive."""
        if target_col not in train_df.columns:
            raise KeyError(f"Нет колонки '{target_col}' в train_df.")
        self._target_col = target_col
        self._fallback_value = float(train_df[target_col].iloc[-1])

        try:
            from autots import AutoTS
        except Exception as e:  # ImportError и пр.
            warnings.warn(
                f"AutoTS недоступен ({e}). Использую наивный fallback "
                "(последнее значение). Установите 'autots' для реального поиска."
            )
            self._used_fallback = True
            return self

        # Готовим длинный формат с датой и значением
        if self.timestamp_col in train_df.columns:
            ts = pd.to_datetime(train_df[self.timestamp_col])
        else:
            # Нет колонки времени — синтезируем равномерную сетку
            ts = pd.date_range("2000-01-01", periods=len(train_df), freq="H")
        long_df = pd.DataFrame(
            {
                "timestamp": pd.Series(ts).to_numpy(),
                "value": train_df[target_col].to_numpy(dtype=float),
                "series_id": target_col,
            }
        )

        try:
            model = AutoTS(
                forecast_length=self.forecast_length,
                frequency=self.frequency,
                ensemble=self.ensemble,
                model_list=self.model_list,
                max_generations=self.max_generations,
                num_validations=self.num_validations,
                no_negatives=False,
                verbose=0,
            )
            self._model = model.fit(
                long_df,
                date_col="timestamp",
                value_col="value",
                id_col="series_id",
            )
            try:
                self._best_model_name = str(self._model.best_model_name)
            except Exception:
                self._best_model_name = None
        except Exception as e:
            warnings.warn(
                f"Обучение AutoTS упало ({e}). Использую наивный fallback."
            )
            self._used_fallback = True
            self._model = None
        return self

    # ------------------------------------------------------------------

    def predict(self, horizon: int) -> np.ndarray:
        """Прогноз на horizon шагов. При fallback — повтор последнего значения."""
        horizon = int(horizon)
        if self._used_fallback or self._model is None:
            return np.full(horizon, self._fallback_value, dtype=float)

        try:
            prediction = self._model.predict(forecast_length=horizon)
            forecast = prediction.forecast
            vals = np.asarray(forecast.iloc[:, 0].to_numpy(), dtype=float)
        except Exception as e:
            warnings.warn(f"Предикт AutoTS упал ({e}). Fallback на last value.")
            return np.full(horizon, self._fallback_value, dtype=float)

        # Выравниваем длину под horizon
        if len(vals) >= horizon:
            return vals[:horizon]
        pad_val = vals[-1] if len(vals) else self._fallback_value
        pad = np.full(horizon - len(vals), pad_val, dtype=float)
        return np.concatenate([vals, pad])

    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        """Метаданные модели для отчёта/трекинга."""
        return {
            "model": self.name,
            "target_col": self._target_col,
            "forecast_length": self.forecast_length,
            "frequency": self.frequency,
            "ensemble": self.ensemble,
            "model_list": self.model_list,
            "max_generations": self.max_generations,
            "num_validations": self.num_validations,
            "best_model_name": self._best_model_name,
            "used_fallback": self._used_fallback,
        }
