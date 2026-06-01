"""Загрузка сырых CSV-серий для пайплайна.

Файлы формируются `binance_spot_downloader.py` и имеют колонки:
    timestamp, open, high, low, close, volume
где timestamp — ISO-строка (UTC). Имя файла вида `data/raw/BTCUSDT_1h.csv`,
символ серии извлекается из имени файла (stem).

Интерфейс ожидается `pipeline/src/train/runner.py::run_train`.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict

import pandas as pd


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Парсит колонку времени: ISO-строки или epoch (s/ms/us/ns).

    Если значения числовые — определяет единицу по медиане (как в downloader);
    иначе парсит как строку. Возвращает tz-naive datetime для совместимости
    с AutoTS (он не любит tz-aware индексы).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        med = float(numeric.dropna().median())
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        ts = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(series, utc=True, errors="coerce")
    # убираем tz-инфо: AutoTS работает с naive-датами
    return ts.dt.tz_localize(None)


def load_raw_series(
    input_glob: str,
    timestamp_col: str,
    target_col: str,
) -> Dict[str, pd.DataFrame]:
    """Загружает все CSV по маске и возвращает словарь {символ: DataFrame}.

    Каждый DataFrame:
    - содержит как минимум колонки `timestamp_col` и `target_col`;
    - отсортирован по времени по возрастанию;
    - без дублей по времени;
    - индекс сброшен (RangeIndex), время остаётся отдельной колонкой
      (важно: `build_basic_features` делает reset_index, поэтому время
      сохраняется именно как колонка).

    Аргументы:
        input_glob    -- glob-маска путей (например, 'data/raw/*_1h.csv').
        timestamp_col -- имя колонки времени.
        target_col    -- имя целевой колонки (например, 'close').

    Возвращает:
        dict {symbol: DataFrame}. Символ = имя файла без расширения.

    Исключения:
        FileNotFoundError -- если по маске не найдено ни одного файла.
        KeyError          -- если в файле нет нужных колонок.
    """
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(
            f"По маске '{input_glob}' не найдено CSV-файлов. "
            "Положите данные в data/raw/ (см. binance_spot_downloader.py)."
        )

    series_map: Dict[str, pd.DataFrame] = {}
    for p in paths:
        symbol = Path(p).stem
        df = pd.read_csv(p)
        missing = [c for c in (timestamp_col, target_col) if c not in df.columns]
        if missing:
            raise KeyError(
                f"В файле {p} нет колонок {missing}. Доступны: {list(df.columns)}"
            )
        df[timestamp_col] = _parse_timestamp(df[timestamp_col])
        df = (
            df.dropna(subset=[timestamp_col, target_col])
            .sort_values(timestamp_col)
            .drop_duplicates(subset=[timestamp_col], keep="first")
            .reset_index(drop=True)
        )
        series_map[symbol] = df

    return series_map
