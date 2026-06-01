"""Хронологический train/test-сплит без утечки будущего.

Интерфейс ожидается `pipeline/src/train/runner.py`:
    split = train_test_split_ts(feat_df, test_size)
    split.train, split.test  # DataFrame'ы
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    """Результат хронологического сплита: train и test (в порядке времени)."""

    train: pd.DataFrame
    test: pd.DataFrame


def train_test_split_ts(df: pd.DataFrame, test_size: float) -> Split:
    """Делит DataFrame по времени: первые (1-test_size) строк → train, хвост → test.

    Данные считаются уже отсортированными по времени (см. load_raw_series).
    Никакого перемешивания — строго хронологический сплит, чтобы test лежал
    в будущем относительно train (без утечки).

    Аргументы:
        df        -- упорядоченный по времени DataFrame признаков.
        test_size -- доля хвоста под тест в (0, 1).

    Возвращает:
        Split(train, test). В каждом — не менее одной строки.

    Исключения:
        ValueError -- если df слишком мал или test_size вне (0, 1).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size должен быть в (0, 1), получено {test_size}")
    n = len(df)
    if n < 2:
        raise ValueError(f"Слишком мало строк для сплита: {n}")

    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 1)  # хотя бы 1 строка в train
    n_train = n - n_test

    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)
    return Split(train=train, test=test)
