"""
features_polars.py — порт build_features (fallback-ветка, pandas_ta недоступен) на polars.

Точное соответствие build_processed_data.build_features для ОДНОГО актива:
OHLCV + ret_1, ret_abs, hl_spread, oc_spread, vol_roll, ema_12, ema_26, rsi_14, vol_ma_24.
RSI — простое rolling(14).mean() (fallback), EMA — ewm(span, adjust=False), без atr_14.

polars считает это лениво/во float64 (сверка с pandas) — пик RAM ~feature-фрейм, а не
×3 копии pandas. Это снимает OOM на длинном 1s. Сверка корректности: verify_features.py.
"""
from __future__ import annotations

import polars as pl

# порядок фич ровно как в pandas build_features (fallback-ветка)
FEATURE_COLS = ["open", "high", "low", "close", "volume", "ret_1", "ret_abs",
                "hl_spread", "oc_spread", "vol_roll", "ema_12", "ema_26",
                "rsi_14", "vol_ma_24"]

_AGG = [pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
        pl.col("close").last(), pl.col("volume").sum()]


def resample_lazy(path: str, freq: str) -> pl.LazyFrame:
    """polars scan_csv + group_by_dynamic -> ленивый OHLCV (timestamp отсортирован)."""
    lf = pl.scan_csv(path).with_columns(
        pl.col("timestamp").str.to_datetime(format="%Y-%m-%d %H:%M:%S%z", time_unit="us").set_sorted())
    return (lf.group_by_dynamic("timestamp", every=freq, closed="left").agg(_AGG)
            .drop_nulls(subset=["close"]))


def add_features(df: pl.DataFrame, bd: int) -> pl.DataFrame:
    """Добавляет фичи к ресемпленному OHLCV (polars DataFrame). float64, как pandas."""
    c = pl.col
    # стадия 1: ret_1 (нужен для ret_abs, vol_roll)
    df = df.with_columns((c("close").log() - c("close").shift(1).log()).alias("ret_1"))
    delta = c("close").diff()
    df = df.with_columns([
        c("ret_1").abs().alias("ret_abs"),
        ((c("high") - c("low")) / c("close")).alias("hl_spread"),
        ((c("open") - c("close")) / c("close")).alias("oc_spread"),
        (c("ret_1").rolling_std(bd) * (bd ** 0.5)).alias("vol_roll"),
        c("close").ewm_mean(span=12, adjust=False).alias("ema_12"),
        c("close").ewm_mean(span=26, adjust=False).alias("ema_26"),
        c("volume").rolling_mean(24).alias("vol_ma_24"),
        # up/down для RSI (np.where(delta>0,delta,0) -> null delta даёт 0.0)
        pl.when(delta > 0).then(delta).otherwise(0.0).rolling_mean(14).alias("_ru"),
        pl.when(delta < 0).then(-delta).otherwise(0.0).rolling_mean(14).alias("_rd"),
    ])
    df = df.with_columns(
        (100.0 - 100.0 / (1.0 + c("_ru") / (c("_rd") + 1e-12))).alias("rsi_14")
    ).drop(["_ru", "_rd"])
    return df


def compute(path: str, freq: str, bd: int) -> pl.DataFrame:
    """Полный путь: path 1s -> ресемпл -> фичи -> y_reg, дроп null. Возвращает polars DF."""
    rs = resample_lazy(path, freq).collect(engine="streaming")
    df = add_features(rs, bd)
    # y_reg = log(close.shift(-1)/close) (как make_targets, horizon=1)
    df = df.with_columns(
        (pl.col("close").shift(-1).log() - pl.col("close").log()).alias("y_reg"))
    df = df.drop_nulls(subset=FEATURE_COLS + ["y_reg"])
    # фичи -> float32 (модель всё равно float32): вдвое меньше RAM на длинном 1s.
    # y_reg оставляем float64 (точность доходностей для Sharpe).
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])
