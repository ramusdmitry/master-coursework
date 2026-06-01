"""
Общие утилиты проекта CP7 (предсказание цены крипты).

Содержит:
- константы проекта (SEED, ACTIVE_ASSET, HORIZON, MIN_HOLDING, COST_BPS и др.);
- set_seed — инициализация генераторов случайных чисел;
- load_splits — загрузка parquet-сплитов из директории;
- resolve_y_column — поиск нужной колонки таргета в DataFrame с MultiIndex;
- WindowDataset — Dataset для окон временного ряда;
- SimpleLSTM — базовый LSTM-классификатор (primary-модель);
- sharpe_ratio, forward_return, net_returns — финансовые метрики;
- triple_barrier_labels — разметка тройным барьером;
- apply_min_holding, apply_hysteresis — фильтрация позиций;
- tune_threshold — подбор порога по val net-Sharpe;
- evaluate_strategy, evaluate_on_forward — торговая оценка стратегии;
- fit_classifier_sharpe — обучение с early stopping по net-Sharpe;
- save_torch_model, save_sklearn_model — сохранение моделей;
- make_synthetic_splits — генератор синтетических данных (fallback).

Все функции и классы извлечены из build_cp6.py без изменений логики.
"""

from __future__ import annotations

import ast
import math
import os
import pickle
import random
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Константы проекта
# =============================================================================

SEED: int = 42
"""Глобальный seed для воспроизводимости."""

ACTIVE_ASSET: str = "BTC"
"""Активный торгуемый актив (используется как префикс признаков BTC__)."""

HORIZON: int = 15
"""Горизонт forward-return в барах; совпадает с VERTICAL_TB."""

MIN_HOLDING: int = 15
"""Минимальное удержание позиции в барах после входа."""

COST_BPS: float = 7.0
"""Суммарные транзакционные издержки (fee + slippage) в базисных пунктах."""

COST_PER_TURN: float = COST_BPS / 1e4
"""Доля от позиции, списываемая при каждой смене позиции."""

PER_YEAR_1M: float = 60.0 * 24.0 * 365.0
"""Число 1-минутных баров в году."""

PER_YEAR_FWD: float = PER_YEAR_1M / HORIZON
"""Число forward-return периодов в году (для Sharpe на горизонте HORIZON)."""

VOL_WINDOW_TB: int = 60
"""Окно волатильности для разметки тройным барьером."""

PT_MULT: float = 1.0
"""Множитель take-profit барьера (в единицах волатильности)."""

SL_MULT: float = 1.0
"""Множитель stop-loss барьера (в единицах волатильности)."""

VERTICAL_TB: int = HORIZON
"""Вертикальный барьер тройной разметки (баров); равен HORIZON."""

# =============================================================================
# Инициализация seed
# =============================================================================


def set_seed(seed: int = SEED) -> None:
    """Инициализирует все генераторы случайных чисел для воспроизводимости.

    Устанавливает seed для: Python random, NumPy, PyTorch CPU и CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_tracking_uri(
    tracking_uri: str,
    fallback_db: str = "mlflow_local.db",
    timeout: float = 2.0,
) -> str:
    """Возвращает рабочий MLflow tracking URI с graceful-fallback.

    Если tracking_uri указывает на http(s)-сервер и он недоступен (Docker не
    поднят), переключается на локальный sqlite-бэкенд. Sqlite, в отличие от
    file-store (./mlruns), поддерживает Model Registry и алиасы — поэтому
    регистрация модели и тег/алиас PRD продолжают работать без сервера.

    Так CLI-скрипты (src/train.py, src/predict_prd.py) перестают падать с
    ConnectionError, если MLflow-сервер не запущен (поведение как в ноутбуках,
    у которых был fallback на ./mlruns).

    Аргументы:
        tracking_uri -- сконфигурированный URI (например, http://localhost:5000).
        fallback_db  -- путь к sqlite-файлу для локального fallback.
        timeout      -- таймаут проверки доступности сервера (сек).

    Возвращает:
        Рабочий URI: исходный http(s), если сервер отвечает; иначе
        'sqlite:///<fallback_db>' (абсолютный путь).
    """
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in ("http", "https"):
        # sqlite/file/прочее — отдаём как есть
        return tracking_uri

    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return tracking_uri
    except OSError:
        local = (Path.cwd() / fallback_db).resolve()
        fallback_uri = f"sqlite:///{local}"
        print(
            f"[MLflow] Сервер {tracking_uri} недоступен — переключаюсь на "
            f"локальный бэкенд {fallback_uri} (Registry+PRD поддерживаются). "
            "Подними Docker (docker compose up -d) для централизованного трекинга."
        )
        return fallback_uri


# =============================================================================
# Загрузка данных
# =============================================================================


def load_splits(data_dir: Path):
    """Загружает parquet-сплиты из директории data_dir.

    Ожидаемые файлы: X_train.parquet, X_val.parquet, X_test.parquet,
    y_train.parquet, y_val.parquet, y_test.parquet.

    Возвращает:
        (X_train, y_train, X_val, y_val, X_test, y_test) — DataFrame'ы.
    """
    data_dir = Path(data_dir)
    return (
        pd.read_parquet(data_dir / "X_train.parquet"),
        pd.read_parquet(data_dir / "y_train.parquet"),
        pd.read_parquet(data_dir / "X_val.parquet"),
        pd.read_parquet(data_dir / "y_val.parquet"),
        pd.read_parquet(data_dir / "X_test.parquet"),
        pd.read_parquet(data_dir / "y_test.parquet"),
    )


def resolve_y_column(df: pd.DataFrame, asset: str, name: str = "y_bin"):
    """Находит колонку таргета в DataFrame с MultiIndex-колонками.

    y-таблицы имеют колонки вида (asset, name) — иногда как кортеж Python,
    иногда как строка-представление кортежа. Функция обрабатывает оба случая.

    Аргументы:
        df   -- DataFrame с колонками y_train / y_val / y_test.
        asset -- строка актива, например 'BTC'.
        name  -- имя таргета, например 'y_bin' или 'y_reg'.

    Возвращает:
        Ключ колонки (для df[key]).

    Исключения:
        KeyError — если колонка не найдена.
    """
    key = (asset, name)
    if key in df.columns:
        return key
    for c in df.columns:
        if c == key:
            return c
        if isinstance(c, str) and c.startswith("("):
            try:
                if ast.literal_eval(c) == key:
                    return c
            except (ValueError, SyntaxError):
                pass
    raise KeyError(
        f"Нет таргета {key}. Примеры колонок: {list(df.columns)[:10]}"
    )


# =============================================================================
# Dataset для временных рядов
# =============================================================================


class WindowDataset(Dataset):
    """PyTorch Dataset для скользящих окон временного ряда (ленивый).

    Окно нарезается в __getitem__, а не материализуется заранее — это снимает
    OOM на полных данных (раньше np.stack всех окон давал ~6-7 ГБ только на
    train/test). Математически идентично прежней версии: окно [i-window:i] с
    меткой y[i] для i >= window.

    Аргументы:
        X      -- массив признаков (N, n_features).
        y      -- массив меток (N,).
        window -- длина окна (число баров на вход модели).
    """

    def __init__(self, X, y, window: int):
        self.window = window
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.asarray(y).astype(np.int64)
        self.n = max(0, len(self.X) - window)
        # метки окон: y[window], y[window+1], ... (совместимо со старым .labels)
        self.labels = self.y[window: window + self.n]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.window]          # окно [idx, idx+window)
        return (
            torch.from_numpy(x),
            torch.tensor(self.y[idx + self.window], dtype=torch.long),
        )

    def __getitems__(self, indices):
        """Батчевая выборка: режет окна целого батча одной векторной операцией.
        DataLoader использует этот метод, если он есть — на порядок меньше Python-
        вызовов, чем 512 отдельных __getitem__. Семантически идентично __getitem__.
        """
        idx = np.asarray(indices, dtype=np.int64)
        rows = idx[:, None] + np.arange(self.window, dtype=np.int64)[None, :]  # (B, window)
        xb = self.X[rows]                                  # (B, window, n_features)
        yb = self.y[idx + self.window].astype(np.int64)    # (B,)
        xt = torch.from_numpy(xb)
        yt = torch.from_numpy(yb)
        return [(xt[i], yt[i]) for i in range(len(idx))]


# =============================================================================
# Архитектура: SimpleLSTM (primary-модель Meta-Labeling)
# =============================================================================


class SimpleLSTM(nn.Module):
    """LSTM-классификатор с двумя классами (long / flat).

    Принимает тензор (B, T, F), возвращает логиты (B, 2).
    Используется как primary-модель в Meta-Labeling.

    Аргументы:
        input_size  -- число входных признаков F.
        hidden_size -- размер скрытого состояния LSTM.
        num_layers  -- число слоёв LSTM.
        dropout     -- вероятность dropout (применяется между слоями LSTM
                       и перед FC-головой).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)
        # Сохраняем гиперпараметры для сериализации
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход: (B, T, F) -> логиты (B, 2)."""
        out, _ = self.lstm(x)          # (B, T, H)
        h = out[:, -1, :]              # берём последний шаг
        return self.fc(self.drop(h))   # (B, 2)


# =============================================================================
# Финансовые метрики и утилиты
# =============================================================================


def sharpe_ratio(log_returns, periods_per_year: float) -> float:
    """Вычисляет аннуализированный коэффициент Шарпа.

    Аргументы:
        log_returns      -- Series или array логарифмических доходностей.
        periods_per_year -- число периодов в году (60*24*365 для 1m баров).

    Возвращает:
        Sharpe ratio (float). Если std == 0 — возвращает 0.0.
    """
    log_returns = pd.Series(log_returns)
    if len(log_returns) == 0 or log_returns.std() == 0:
        return 0.0
    return float(log_returns.mean() / log_returns.std() * np.sqrt(periods_per_year))


def forward_return(y_reg_series: pd.Series, horizon: int) -> pd.Series:
    """Суммарная log-доходность за следующие `horizon` баров (без утечки).

    r_fwd[t] = r[t+1] + ... + r[t+horizon].
    Последние `horizon` значений будут NaN (нет достаточно будущих баров).

    Реализация: rolling(horizon).sum() считает скользящую сумму за horizon
    баров назад, shift(-horizon) сдвигает назад -> сумма ВПЕРЁД.

    Используется и для создания меток, и для оценки стратегии (горизонт
    таргета и горизонт оценки совпадают — исправление ошибки несоответствия).
    """
    r = y_reg_series.astype(float)
    fwd = r.rolling(window=horizon, min_periods=horizon).sum().shift(-horizon)
    return fwd


def net_returns(
    positions: pd.Series,
    asset_returns: pd.Series,
    cost_per_turn: float = COST_PER_TURN,
) -> pd.Series:
    """Чистая доходность стратегии long/flat с учётом издержек на смену позиции.

    Аргументы:
        positions     -- 0/1 (или доля капитала) на каждый шаг.
        asset_returns -- log-returns актива.
        cost_per_turn -- доля от позиции, списываемая при смене (= COST_BPS/1e4).

    Возвращает:
        pd.Series чистых доходностей.
    """
    positions = positions.reindex(asset_returns.index).fillna(0.0)
    gross = positions * asset_returns
    turn = positions.diff().abs().fillna(positions.abs())
    cost = turn * cost_per_turn
    return gross - cost


# =============================================================================
# Разметка тройным барьером
# =============================================================================


def triple_barrier_labels(
    r,
    vol_window: int,
    pt_mult: float,
    sl_mult: float,
    vertical: int,
    eps: float = 1e-8,
) -> pd.Series:
    """Создаёт бинарные метки методом тройного барьера (López de Prado).

    Для каждого бара pos >= vol_window вычисляет:
    - верхний барьер: pt_mult * vol (take-profit);
    - нижний барьер: -sl_mult * vol (stop-loss);
    - вертикальный барьер: vertical баров.

    Метка = 1, если цена достигла верхнего барьера до нижнего или вертикального;
    метка = 0 иначе.

    Аргументы:
        r          -- Series 1-минутных log-returns.
        vol_window -- длина окна для расчёта волатильности.
        pt_mult    -- множитель take-profit.
        sl_mult    -- множитель stop-loss.
        vertical   -- длина вертикального барьера (баров).
        eps        -- минимальная волатильность (защита от деления на ноль).

    Возвращает:
        pd.Series с метками 0/1 (NaN для первых vol_window и последних vertical-1 баров).
    """
    r = r.astype(float).sort_index()
    vals = r.values
    n = len(vals)
    out = np.full(n, np.nan)
    for pos in range(vol_window, n - vertical + 1):
        sig = float(vals[pos - vol_window:pos].std()) or eps
        if sig < eps:
            sig = eps
        upper, lower = pt_mult * sig, -sl_mult * sig
        cum, lab = 0.0, 0
        for k in range(vertical):
            cum += vals[pos + k]
            if cum >= upper:
                lab = 1
                break
            if cum <= lower:
                lab = 0
                break
        else:
            lab = 0
        out[pos] = lab
    return pd.Series(out, index=r.index)


# =============================================================================
# Фильтрация позиций
# =============================================================================


def apply_min_holding(positions: pd.Series, min_hold: int) -> pd.Series:
    """Постобработка позиций: удерживает позицию минимум min_hold баров.

    Алгоритм: сканируем последовательно; при изменении позиции фиксируем
    новое значение и не меняем следующие min_hold-1 баров.

    Снижает число сделок примерно в min_hold раз -> издержки падают
    пропорционально (исправление причины #3 из CP6).

    Аргументы:
        positions -- pd.Series позиций 0/1 (или дробных).
        min_hold  -- минимальное число баров удержания.

    Возвращает:
        pd.Series отфильтрованных позиций.
    """
    vals = positions.values.copy()
    n = len(vals)
    hold_counter = 0
    current_pos = vals[0] if n > 0 else 0
    out = np.empty(n, dtype=float)
    for i in range(n):
        if hold_counter > 0:
            out[i] = current_pos
            hold_counter -= 1
        else:
            if vals[i] != current_pos:
                current_pos = vals[i]
                hold_counter = min_hold - 1
            out[i] = current_pos
    return pd.Series(out, index=positions.index)


def apply_hysteresis(
    prob: np.ndarray,
    enter_thr: float,
    exit_thr: float,
) -> np.ndarray:
    """Двухпороговый сигнал: входим при prob>=enter_thr, выходим при prob<exit_thr.

    При prob в [exit_thr, enter_thr) — держим прежнюю позицию (гистерезис).
    Устраняет «дёргание» у одного порога, снижает число сделок.

    Аргументы:
        prob       -- массив вероятностей (float).
        enter_thr  -- порог входа.
        exit_thr   -- порог выхода (exit_thr < enter_thr).

    Возвращает:
        np.ndarray позиций 0.0/1.0.
    """
    n = len(prob)
    out = np.zeros(n, dtype=float)
    pos = 0.0
    for i in range(n):
        p = float(prob[i])
        if p >= enter_thr:
            pos = 1.0
        elif p < exit_thr:
            pos = 0.0
        # иначе: держим pos (гистерезис)
        out[i] = pos
    return out


# =============================================================================
# Подбор порога
# =============================================================================


def tune_threshold(
    prob_val: np.ndarray,
    fwd_val: pd.Series,
    idx_val,
    periods_per_year: float = PER_YEAR_FWD,
    thr_grid=None,
    hold_options=None,
    mode: str = "simple",
) -> dict:
    """Перебирает пороги (и опционально min_hold / гистерезис) по val net-Sharpe.

    Режимы:
        'simple'     -- один порог, один min_hold;
        'hysteresis' -- enter_thr / exit_thr (exit_thr = enter_thr - gap).

    Всё подбирается ТОЛЬКО на val; финальная оценка должна выполняться
    на test снаружи (строгое разделение).

    Аргументы:
        prob_val         -- вероятности на val-сете.
        fwd_val          -- forward-return Series на val.
        idx_val          -- индекс val-окон (DatetimeIndex или RangeIndex).
        periods_per_year -- число периодов в году для Sharpe.
        thr_grid         -- grid порогов; по умолчанию linspace(0.30, 0.72, 43).
        hold_options     -- список вариантов min_hold; по умолчанию [1, HORIZON].
        mode             -- 'simple' или 'hysteresis'.

    Возвращает:
        dict с ключами threshold, min_hold, val_net_sharpe (и exit_thr в режиме hysteresis).
    """
    if thr_grid is None:
        thr_grid = np.linspace(0.30, 0.72, 43)
    if hold_options is None:
        hold_options = [1, HORIZON]

    best_sh = -1e18
    best_params: dict = {"threshold": 0.5, "min_hold": 1, "val_net_sharpe": -1e18}
    fwd_series = pd.Series(fwd_val.values, index=idx_val[: len(fwd_val)])

    if mode == "simple":
        for t in thr_grid:
            raw_pos = pd.Series(
                (prob_val >= t).astype(float), index=idx_val[: len(prob_val)]
            )
            for mh in hold_options:
                pos = apply_min_holding(raw_pos, mh)
                pos_aligned = pos.reindex(fwd_series.index).fillna(0.0)
                sh = sharpe_ratio(
                    net_returns(pos_aligned, fwd_series), periods_per_year
                )
                if sh > best_sh:
                    best_sh = sh
                    best_params = {
                        "threshold": float(t),
                        "min_hold": int(mh),
                        "val_net_sharpe": float(sh),
                    }
    elif mode == "hysteresis":
        gaps = [0.05, 0.10, 0.15]
        for t in thr_grid:
            for gap in gaps:
                exit_t = max(0.0, t - gap)
                raw = apply_hysteresis(prob_val, enter_thr=t, exit_thr=exit_t)
                raw_pos = pd.Series(raw, index=idx_val[: len(prob_val)])
                for mh in hold_options:
                    pos = apply_min_holding(raw_pos, mh)
                    pos_aligned = pos.reindex(fwd_series.index).fillna(0.0)
                    sh = sharpe_ratio(
                        net_returns(pos_aligned, fwd_series), periods_per_year
                    )
                    if sh > best_sh:
                        best_sh = sh
                        best_params = {
                            "threshold": float(t),
                            "exit_thr": float(exit_t),
                            "min_hold": int(mh),
                            "val_net_sharpe": float(sh),
                        }
    return best_params


# =============================================================================
# Оценка стратегии
# =============================================================================


def evaluate_strategy(
    positions,
    asset_returns,
    periods_per_year: float,
    name: str = "strategy",
    prob=None,
    true_label=None,
) -> dict:
    """Информативная оценка торговой стратегии с жёстким выравниванием индексов.

    Исправление: при рассинхроне длин positions и asset_returns reindex
    тихо заполнял NaN нулями -> метрики «схлопывались» в 0.
    Теперь работаем только по общему индексу и диагностируем вырожденные позиции.

    Аргументы:
        positions        -- Series позиций 0/1 (или дробных).
        asset_returns    -- Series log-returns актива.
        periods_per_year -- число периодов в году для Sharpe.
        name             -- название стратегии (для отчёта).
        prob             -- вероятности (для ROC-AUC; опционально).
        true_label       -- истинные метки (для ROC-AUC; опционально).

    Возвращает:
        dict с метриками стратегии.
    """
    positions = pd.Series(positions).astype(float)
    asset_returns = pd.Series(asset_returns).astype(float)
    common = positions.index.intersection(asset_returns.index)
    if len(common) == 0:
        # индексы не совпадают — выравниваем позиционно
        m = min(len(positions), len(asset_returns))
        positions = pd.Series(
            positions.values[:m], index=asset_returns.index[:m]
        )
        common = positions.index
    pos = positions.reindex(common).fillna(0.0)
    ret = asset_returns.reindex(common).fillna(0.0)

    gross = pos * ret
    turn = pos.diff().abs().fillna(pos.abs())
    net = gross - turn * COST_PER_TURN
    n_trades = int(turn.sum())

    frac_long = float((pos > 0).mean())
    n_unique_pos = int(pd.unique(np.round(pos.values, 4)).size)
    degenerate = n_unique_pos <= 1  # позиция не меняется

    res = {
        "Стратегия": name,
        "Sharpe gross": round(sharpe_ratio(gross, periods_per_year), 4),
        "Sharpe net": round(sharpe_ratio(net, periods_per_year), 4),
        "PnL net (sum logret)": round(float(net.sum()), 5),
        "% времени в long": round(100 * frac_long, 1),
        "Сделок (смен позиции)": n_trades,
        "Вырождена?": "ДА" if degenerate else "нет",
    }
    if prob is not None and true_label is not None:
        prob = np.asarray(prob)
        true_label = np.asarray(true_label)
        try:
            res["ROC-AUC"] = round(float(roc_auc_score(true_label, prob)), 4)
        except ValueError:
            res["ROC-AUC"] = float("nan")
        in_pos = pos.values > 0
        if in_pos.sum() > 0:
            res["Hit-rate входов"] = round(
                100 * float((ret.values[in_pos] > 0).mean()), 1
            )
        else:
            res["Hit-rate входов"] = float("nan")
    return res


def evaluate_on_forward(
    positions: pd.Series,
    fwd_series: pd.Series,
    name: str = "strategy",
    prob=None,
    true_label=None,
) -> dict:
    """Оценка стратегии на forward-return горизонта HORIZON.

    Обёртка над evaluate_strategy; periods_per_year = PER_YEAR_FWD.
    Позиции и fwd_series должны быть выровнены по индексу.
    """
    return evaluate_strategy(
        positions,
        fwd_series,
        PER_YEAR_FWD,
        name=name,
        prob=prob,
        true_label=true_label,
    )


# =============================================================================
# Вспомогательные функции обучения (используются в MetaLabelingModel)
# =============================================================================


def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    """Одна эпоха обучения: прогоняет батчи, считает loss, обновляет веса.

    Возвращает средний train loss за эпоху.
    """
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        tot += loss.item() * xb.size(0)
        n += xb.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def _predict_labels_probs(model, loader, device):
    """Предсказывает метки и вероятности для всего DataLoader'а.

    Возвращает:
        (preds, ys, probs) -- np.ndarray.
    """
    model.eval()
    preds, ys, probs = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        preds.append(torch.argmax(logits, 1).cpu().numpy())
        ys.append(yb.numpy())
    if not preds:
        return np.array([]), np.array([]), np.array([])
    return (
        np.concatenate(preds),
        np.concatenate(ys),
        np.concatenate(probs),
    )


def fit_classifier_sharpe(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    fwd_val: pd.Series,
    idx_val_w,
    epochs: int = 30,
    patience: int = 5,
    tag: str = "",
    periods_per_year: float = PER_YEAR_FWD,
    min_hold: int = 1,
    device=None,
    history_out: Optional[list] = None,
) -> float:
    """Обучение с early stopping по val net-Sharpe на forward-return.

    Вместо ROC-AUC (бесполезен при AUC~0.51-0.55) отбираем эпоху по
    торговой метрике: val net-Sharpe на forward-return горизонта HORIZON.

    Аргументы:
        model            -- nn.Module (SimpleLSTM или другой классификатор).
        train_loader     -- DataLoader для обучающей выборки.
        val_loader       -- DataLoader для валидационной выборки.
        criterion        -- функция потерь (CrossEntropyLoss).
        optimizer        -- оптимизатор (Adam).
        fwd_val          -- pd.Series forward-return на val.
        idx_val_w        -- индекс val-окон (выровнен с предсказаниями модели).
        epochs           -- максимальное число эпох.
        patience         -- число эпох без улучшения до остановки.
        tag              -- метка для логов.
        periods_per_year -- число периодов в году для Sharpe.
        min_hold         -- минимальное удержание при оценке.
        device           -- torch.device; если None — авто-определение.
        history_out      -- если передан list, в него по эпохам пишутся словари
                            {epoch, train_loss, val_net_sharpe} (для learning curve
                            в MLflow). Не ломает существующих вызовов.

    Возвращает:
        best_val_sharpe (float) -- лучший val net-Sharpe.
    """
    if device is None:
        device = next(model.parameters()).device
    best_sh, best_state, stale = -1e18, None, 0
    fwd_s = pd.Series(fwd_val.values[: len(idx_val_w)], index=idx_val_w)
    for ep in range(1, epochs + 1):
        tl = _train_epoch(model, train_loader, criterion, optimizer, device)
        _, _, p1v = _predict_labels_probs(model, val_loader, device)
        n_p = min(len(p1v), len(idx_val_w))
        raw_pos = pd.Series(
            (p1v[:n_p] >= 0.5).astype(float), index=idx_val_w[:n_p]
        )
        pos_mh = apply_min_holding(raw_pos, min_hold) if min_hold > 1 else raw_pos
        pos_aligned = pos_mh.reindex(fwd_s.index).fillna(0.0)
        val_sh = sharpe_ratio(
            net_returns(pos_aligned, fwd_s), periods_per_year
        )
        if history_out is not None:
            history_out.append(
                {"epoch": ep, "train_loss": float(tl), "val_net_sharpe": float(val_sh)}
            )
        print(
            f"[{tag}] {ep:02d}/{epochs} loss={tl:.4f} val_netSharpe={val_sh:.3f}"
        )
        if val_sh > best_sh:
            best_sh = val_sh
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[{tag}] early stopping (val net-Sharpe не растёт)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_sh


# =============================================================================
# Сохранение моделей
# =============================================================================


def save_torch_model(model, path, extra: Optional[dict] = None) -> Path:
    """Сохраняет state_dict torch-модели + метаданные.

    Аргументы:
        model -- nn.Module.
        path  -- путь к файлу (Path или str).
        extra -- дополнительные метаданные (dict).

    Возвращает:
        Path сохранённого файла.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "state_dict": model.state_dict(),
        "class_name": type(model).__name__,
    }
    # Извлекаем гиперпараметры из атрибутов модели (если есть)
    for attr in ("_input_size", "_hidden_size", "_num_layers", "_dropout"):
        if hasattr(model, attr):
            payload[attr.lstrip("_")] = getattr(model, attr)
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[сохранено] {path}")
    return path


def save_sklearn_model(model, path, extra: Optional[dict] = None) -> Path:
    """Сохраняет sklearn-модель (+ метаданные) в pickle-файл.

    Аргументы:
        model -- sklearn-оценщик.
        path  -- путь к файлу (Path или str).
        extra -- дополнительные метаданные (dict).

    Возвращает:
        Path сохранённого файла.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "meta": extra or {}}, f)
    print(f"[сохранено] {path}")
    return path


# =============================================================================
# Мета-метки
# =============================================================================


def meta_labels(primary_prob: np.ndarray, true_label: np.ndarray, thr: float = 0.5):
    """Вычисляет мета-метки для вторичного классификатора.

    Мета-метка = 1, если primary сказал 'long' (prob >= thr) И сделка
    реально оказалась успешной (label == 1).

    Аргументы:
        primary_prob -- вероятности primary-модели.
        true_label   -- истинные метки (0/1).
        thr          -- порог входа primary.

    Возвращает:
        (acted, success) -- np.ndarray: acted = где primary решил войти;
                           success = где вход оказался успешным.
    """
    acted = (primary_prob >= thr).astype(int)
    success = ((acted == 1) & (np.asarray(true_label) == 1)).astype(int)
    return acted, success


def window_aggregates(ds: WindowDataset, chunk: int = 50000) -> np.ndarray:
    """Агрегаты признаков окна: mean | std | last_step.

    Используется для формирования признаков вторичного классификатора.
    Форма выхода: (N, 3 * n_features). Считается через sliding_window_view
    чанками (без материализации всех окон) — идентично прежней реализации
    seq.mean(1)|seq.std(1)|seq[:,-1,:].
    """
    from numpy.lib.stride_tricks import sliding_window_view

    X, w, n = ds.X, ds.window, ds.n
    F = X.shape[1]
    if n <= 0:
        return np.empty((0, 3 * F), dtype=np.float32)
    sw = sliding_window_view(X, w, axis=0)  # (len-w+1, F, w) — view, без копии
    out = np.empty((n, 3 * F), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = sw[s:e]                      # (e-s, F, w)
        out[s:e, :F] = block.mean(axis=2)
        out[s:e, F:2 * F] = block.std(axis=2)
        out[s:e, 2 * F:] = block[:, :, -1]
    return out


# =============================================================================
# Синтетический генератор данных (fallback при отсутствии реальных parquet)
# =============================================================================


def make_synthetic_splits(
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 1000,
    n_features: int = 20,
    asset: str = ACTIVE_ASSET,
    seed: int = SEED,
) -> tuple:
    """Генерирует синтетические сплиты для тестирования пайплайна.

    Создаёт X_train/val/test (DataFrame с колонками asset__feature_i)
    и y_train/val/test (DataFrame с колонками (asset, 'y_bin') и (asset, 'y_reg')).

    Используется как fallback, когда реальные parquet-файлы недоступны.
    Обеспечивает полный запуск src/train.py без реальных данных.

    Аргументы:
        n_train    -- число строк в train.
        n_val      -- число строк в val.
        n_test     -- число строк в test.
        n_features -- число признаков.
        asset      -- имя актива (префикс колонок).
        seed       -- seed для NumPy.

    Возвращает:
        (X_train, y_train, X_val, y_val, X_test, y_test) -- DataFrame'ы.
    """
    rng = np.random.default_rng(seed)
    feature_cols = [f"{asset}__{i:02d}" for i in range(n_features)]

    def _make_X(n: int) -> pd.DataFrame:
        data = rng.standard_normal((n, n_features)).astype(np.float32)
        return pd.DataFrame(data, columns=feature_cols)

    def _make_y(n: int) -> pd.DataFrame:
        # Синтетическая бинарная метка: немного выше 50% для классовой связи
        y_reg = rng.standard_normal(n).astype(np.float32) * 0.001
        y_bin = (y_reg > 0).astype(np.int64)
        cols = pd.MultiIndex.from_tuples(
            [(asset, "y_bin"), (asset, "y_reg")], names=["asset", "name"]
        )
        return pd.DataFrame(
            np.column_stack([y_bin, y_reg]),
            columns=cols,
        )

    X_train = _make_X(n_train)
    X_val = _make_X(n_val)
    X_test = _make_X(n_test)
    y_train = _make_y(n_train)
    y_val = _make_y(n_val)
    y_test = _make_y(n_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
