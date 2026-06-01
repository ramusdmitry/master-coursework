#!/usr/bin/env python3
"""
build_processed_data.py — восстановление processed-паркетов из сырых CSV.

Точно воспроизводит препроцессинг из checkpoint-2-eda.ipynb (без EDA-графиков):
  1. load_all_assets   — чтение data/raw/{ASSET}_1m.csv, ресемпл на сетку интервала;
  2. all_df_interp     — интерполяция пропусков (time, limit=2) + ffill объёма;
  3. build_features    — признаки {ASSET}__* (ret_1, rsi_14, ema, atr_14, ...);
  4. make_targets      — y_reg (log-доходность за HORIZON) и y_bin (направление);
  5. time_split        — хронологический сплит по датам SPLIT (без утечки);
  6. to_parquet        — сохранение X_/y_ train/val/test в data/processed/.

Индикаторы (rsi/ema/atr) считаются через pandas_ta — те же формулы (Wilder RMA),
что в оригинале (0.3.14b0); используется доступная сейчас 0.4.x (формулы идентичны).

Запуск из корня репозитория (там, где лежит data/raw):
    python build_processed_data.py
    python build_processed_data.py --raw-dir data/raw --out-dir data/processed
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import pandas_ta as ta
except Exception:  # pragma: no cover
    ta = None

# =============================================================================
# Конфигурация (как в checkpoint-2-eda.ipynb)
# =============================================================================

ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP"]
INTV = "1m"
HORIZON = 1

_DAY_STEPS = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
DAY_N = _DAY_STEPS.get(INTV, 24)

SPLIT = {
    "train_end": "2024-08-31",
    "val_end": "2024-12-31",
    "test_end": "2025-09-30",
}

CSV_PATTERN = "{asset}_{intv}.csv"

_FREQ_MAP = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


# =============================================================================
# 1. Загрузка и выравнивание на сетку интервала
# =============================================================================


def _read_single_path(path: Path) -> pd.DataFrame:
    """Читает CSV/parquet OHLCV, приводит к UTC-индексу и числовым типам."""
    if path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    lower = [x.lower() for x in df.columns]
    missing = [c for c in required if c not in lower]
    if missing:
        raise ValueError(f"В файле {path} нет колонок: {missing}")

# Order-flow поля (если есть в CSV) — давление агрессивных покупателей/продавцов.
OF_COLS = ["quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]


def _read_single_path(path: Path) -> pd.DataFrame:
    df_cols_in = None
    if path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    lower = [x.lower() for x in df.columns]
    missing = [c for c in required if c not in lower]
    if missing:
        raise ValueError(f"В файле {path} нет колонок: {missing}")

    df = df.rename(columns={k: k.lower().strip() for k in df.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    keep = ["open", "high", "low", "close", "volume"]
    for c in OF_COLS:
        if c in df.columns:
            keep.append(c)
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[keep]


def align_to_interval(df: pd.DataFrame, intv: str) -> pd.DataFrame:
    """Ресемпл на регулярную сетку интервала с OHLCV(+order-flow)-агрегацией."""
    freq = _FREQ_MAP.get(intv, intv)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in OF_COLS:
        if c in df.columns:
            agg[c] = "sum"  # объёмные поля суммируются на баре интервала
    return df.resample(freq).agg(agg)


def load_all_assets(assets, raw_dir: Path) -> dict:
    """Загружает все активы в dict {asset: DataFrame} с выравниванием."""
    frames = {}
    for a in assets:
        csv_path = raw_dir / CSV_PATTERN.format(asset=a, intv=INTV)
        if not csv_path.exists():
            print(f"[WARN] {a}: не найден {csv_path}")
            continue
        df = align_to_interval(_read_single_path(csv_path), INTV)
        frames[a] = df
        print(f"Loaded {a}: {len(df):,} rows [{df.index.min()} .. {df.index.max()}]")
    return frames


def build_all_df(frames: dict) -> pd.DataFrame:
    """Собирает MultiIndex-таблицу (asset, ohlcv) из словаря активов."""
    all_df = None
    for a, df in frames.items():
        df = df.copy()
        df.columns = pd.MultiIndex.from_product([[a], df.columns])
        all_df = df if all_df is None else all_df.join(df, how="outer")
    if all_df is None:
        raise RuntimeError("Не удалось загрузить ни один актив.")
    all_df = all_df[~all_df.index.duplicated(keep="first")].sort_index()
    all_df = all_df.tz_convert("UTC") if all_df.index.tz is not None else all_df.tz_localize("UTC")
    return all_df


def interpolate(all_df: pd.DataFrame) -> pd.DataFrame:
    """Интерполяция цен (time, limit=2) и ffill объёма (limit=1) — как в EDA."""
    interp = all_df.copy()
    for a in interp.columns.levels[0]:
        for c in ["open", "high", "low", "close"]:
            interp[(a, c)] = interp[(a, c)].interpolate(method="time", limit=2)
        for c in ["volume"] + OF_COLS:
            if (a, c) in interp.columns:
                interp[(a, c)] = interp[(a, c)].ffill(limit=1)
    return interp


# =============================================================================
# 2. Признаки (точная копия build_features из EDA)
# =============================================================================


def build_features(df: pd.DataFrame, order_flow: bool = False) -> pd.DataFrame:
    """Признаки по каждому активу с префиксом {asset}__ + кросс-активные.

    order_flow=False — точная копия EDA (15 фич/актив, для воспроизводимости).
    order_flow=True  — добавляет order-flow фичи из полей Binance kline:
        taker_buy_ratio  — доля агрессивных покупок (taker_buy_base/volume);
        ofi              — order-flow imbalance = 2*ratio-1 in [-1,1];
        avg_trade_size   — средний размер сделки (volume/trades);
        trades_log       — log1p числа сделок (интенсивность).
    """
    out = pd.DataFrame(index=df.index)
    for a in df.columns.levels[0]:
        sub = df[a]
        # База — строго OHLCV (сырые order-flow поля в фичи напрямую не идут)
        o = sub[["open", "high", "low", "close", "volume"]].copy()

        o["ret_1"] = np.log(o["close"] / o["close"].shift(1))
        o["ret_abs"] = o["ret_1"].abs()
        o["hl_spread"] = (o["high"] - o["low"]) / o["close"]
        o["oc_spread"] = (o["open"] - o["close"]) / o["close"]
        o["vol_roll"] = o["ret_1"].rolling(DAY_N).std() * np.sqrt(DAY_N)

        if ta is not None:
            o["rsi_14"] = ta.rsi(o["close"], length=14)
            o["ema_12"] = ta.ema(o["close"], length=12)
            o["ema_26"] = ta.ema(o["close"], length=26)
            o["atr_14"] = ta.atr(o["high"], o["low"], o["close"], length=14)
        else:
            o["ema_12"] = o["close"].ewm(span=12, adjust=False).mean()
            o["ema_26"] = o["close"].ewm(span=26, adjust=False).mean()
            delta = o["close"].diff()
            up = np.where(delta > 0, delta, 0.0)
            down = np.where(delta < 0, -delta, 0.0)
            roll_up = pd.Series(up, index=o.index).rolling(14).mean()
            roll_down = pd.Series(down, index=o.index).rolling(14).mean()
            rs = roll_up / (roll_down + 1e-12)
            o["rsi_14"] = 100 - (100 / (1 + rs))

        o["vol_ma_24"] = o["volume"].rolling(24).mean()

        # --- Order-flow фичи (пункт 3 плана: давление покупателей/продавцов) ---
        if order_flow and "taker_buy_base" in sub.columns:
            vol = sub["volume"].replace(0, np.nan)
            o["taker_buy_ratio"] = (sub["taker_buy_base"] / vol).clip(0, 1)
            o["ofi"] = 2.0 * o["taker_buy_ratio"] - 1.0
            o["avg_trade_size"] = sub["volume"] / sub["trades"].replace(0, np.nan)
            o["trades_log"] = np.log1p(sub["trades"].clip(lower=0))
            # rolling order-flow: устойчивое давление, моментум, дивергенция OFI↔цена
            o["ofi_ma_5"] = o["ofi"].rolling(5).mean()
            o["ofi_ma_15"] = o["ofi"].rolling(15).mean()
            o["ofi_ma_60"] = o["ofi"].rolling(60).mean()
            o["ofi_mom"] = o["ofi_ma_5"] - o["ofi_ma_60"]      # моментум давления
            o["taker_ratio_std_15"] = o["taker_buy_ratio"].rolling(15).std()
            signed = 2.0 * sub["taker_buy_base"] - sub["volume"]   # signed volume (CVD-шаг)
            o["signed_vol_ma_15"] = (signed.rolling(15).mean()
                                     / (sub["volume"].rolling(15).mean() + 1e-9))
            o["ofi_x_ret"] = o["ofi"] * o["ret_1"]              # дивергенция давление/движение

        o.columns = [f"{a}__{c}" for c in o.columns]
        out = out.join(o, how="outer")

    if {"BTC__close", "ETH__close"}.issubset(set(out.columns)):
        out["spread_btc_eth"] = np.log(out["BTC__close"]) - np.log(out["ETH__close"])
        out["corr_btc_eth_24"] = out["BTC__ret_1"].rolling(24).corr(out["ETH__ret_1"])

    return out


# =============================================================================
# 3. Таргеты (точная копия make_targets из EDA)
# =============================================================================


def make_targets(df_close: pd.Series, horizon: int = 1):
    """y_reg = log(close[t+h]/close[t]); y_bin = 1 если y_reg > 0."""
    y_reg = np.log(df_close.shift(-horizon) / df_close)
    y_bin = (y_reg > 0).astype(int)
    return y_reg, y_bin


def merge_onchain(features: pd.DataFrame, onchain_path: str) -> pd.DataFrame:
    """Добавляет on-chain фичи (BTC__oc_*) на минутный грид без look-ahead.

    On-chain метрики дневные. Чтобы не заглядывать в будущее, значение дня D
    становится известно только на D+1 (shift(1)), затем forward-fill на минуты.
    Фичи: дневной pct_change и 30-дневный z-score уровня (стационаризация).
    """
    oc = pd.read_csv(onchain_path)
    oc["date"] = pd.to_datetime(oc["date"], utc=True)
    oc = oc.set_index("date").sort_index()

    feat_oc = pd.DataFrame(index=oc.index)
    for c in oc.columns:
        feat_oc[f"BTC__oc_{c}_chg"] = oc[c].pct_change()
        roll = oc[c].rolling(30, min_periods=10)
        feat_oc[f"BTC__oc_{c}_z"] = (oc[c] - roll.mean()) / (roll.std() + 1e-9)

    feat_oc = feat_oc.shift(1)  # значение дня D доступно только на D+1 (без утечки)
    feat_oc_min = feat_oc.reindex(features.index, method="ffill")
    return features.join(feat_oc_min)


def merge_futures(features: pd.DataFrame, fut_dir: str) -> pd.DataFrame:
    """Добавляет futures-фичи позиционирования (BTC__fut_*) без look-ahead.

    metrics (5-мин): open interest, top-trader/account long-short ratio, taker l/s vol.
    funding (8ч): ставка финансирования. Оба сдвигаются на 1 шаг (значение доступно
    только после закрытия окна) и forward-fill на 1m-грид.
    """
    fd = Path(fut_dir)
    m = pd.read_csv(fd / "btc_futures_metrics.csv")
    m["t"] = pd.to_datetime(m["create_time"], utc=True, errors="coerce")
    m = m.dropna(subset=["t"]).set_index("t").sort_index()
    m = m[~m.index.duplicated(keep="first")]
    fm = pd.DataFrame(index=m.index)
    oi = pd.to_numeric(m["sum_open_interest"], errors="coerce")
    fm["BTC__fut_oi_chg"] = oi.pct_change()
    roll = oi.rolling(288, min_periods=50)               # сутки 5-мин баров
    fm["BTC__fut_oi_z"] = (oi - roll.mean()) / (roll.std() + 1e-9)
    fm["BTC__fut_toptrader_ls"] = pd.to_numeric(m["sum_toptrader_long_short_ratio"], errors="coerce")
    fm["BTC__fut_account_ls"] = pd.to_numeric(m["count_long_short_ratio"], errors="coerce")
    fm["BTC__fut_taker_ls"] = pd.to_numeric(m["sum_taker_long_short_vol_ratio"], errors="coerce")
    fm = fm.shift(1)
    fm_min = fm.reindex(features.index, method="ffill")

    f = pd.read_csv(fd / "btc_funding.csv")
    f["t"] = pd.to_datetime(f["calc_time"], unit="ms", utc=True, errors="coerce")  # epoch ms
    f = f.dropna(subset=["t"]).set_index("t").sort_index()
    f = f[~f.index.duplicated(keep="first")]
    fr = pd.to_numeric(f["last_funding_rate"], errors="coerce")
    fu = pd.DataFrame(index=f.index)
    fu["BTC__fut_funding"] = fr
    rollf = fr.rolling(21, min_periods=5)                 # ~неделя (3/день × 7)
    fu["BTC__fut_funding_z"] = (fr - rollf.mean()) / (rollf.std() + 1e-9)
    fu = fu.shift(1)
    fu_min = fu.reindex(features.index, method="ffill")

    out = features.join(fm_min).join(fu_min)
    # inf (например pct_change при OI=0) -> nan, чтобы финальный dropna их убрал
    fut_cols = list(fm.columns) + list(fu.columns)
    out[fut_cols] = out[fut_cols].replace([np.inf, -np.inf], np.nan)
    return out


def build_targets(all_df_interp: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Собирает MultiIndex-таблицу таргетов (asset, y_reg/y_bin)."""
    targets = {}
    for a in all_df_interp.columns.levels[0]:
        y_reg, y_bin = make_targets(all_df_interp[a]["close"], horizon)
        targets[(a, "y_reg")] = y_reg
        targets[(a, "y_bin")] = y_bin
    return pd.DataFrame(targets)


# =============================================================================
# 4. Хронологический сплит (точная копия time_split из EDA)
# =============================================================================


def time_split(df: pd.DataFrame, split_cfg: dict):
    """Делит по датам: train ≤ train_end < val ≤ val_end < test ≤ test_end."""
    idx = df.index
    tr_end = pd.to_datetime(split_cfg["train_end"], utc=True)
    va_end = pd.to_datetime(split_cfg["val_end"], utc=True)
    te_end = pd.to_datetime(split_cfg["test_end"], utc=True)
    i_tr = idx <= tr_end
    i_va = (idx > tr_end) & (idx <= va_end)
    i_te = (idx > va_end) & (idx <= te_end)
    return df.loc[i_tr], df.loc[i_va], df.loc[i_te]


# =============================================================================
# main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка processed-паркетов из data/raw")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--order-flow", action="store_true",
                        help="добавить order-flow фичи (taker_buy_ratio, ofi, ...)")
    parser.add_argument("--onchain", default=None,
                        help="путь к on-chain CSV (btc_onchain_daily.csv) для добавления фич")
    parser.add_argument("--futures", default=None,
                        help="путь к каталогу data/futures (metrics+funding) для добавления фич")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"pandas_ta: {'есть ('+ta.version+')' if ta is not None else 'НЕТ (fallback без atr_14)'}")
    print("=== 1. Загрузка активов ===")
    frames = load_all_assets(ASSETS, raw_dir)
    all_df = build_all_df(frames)
    print("all_df shape:", all_df.shape)

    print("=== 2. Интерполяция ===")
    all_df_interp = interpolate(all_df)

    print("=== 3. Признаки ===")
    print(f"order-flow фичи: {'ВКЛ' if args.order_flow else 'выкл (baseline)'}")
    features = build_features(all_df_interp, order_flow=args.order_flow)
    if args.onchain:
        features = merge_onchain(features, args.onchain)
        print(f"on-chain фичи добавлены из {args.onchain}")
    if args.futures:
        features = merge_futures(features, args.futures)
        print(f"futures фичи добавлены из {args.futures}")
    print("features shape:", features.shape)

    print("=== 4. Таргеты (HORIZON=%d) ===" % HORIZON)
    targets = build_targets(all_df_interp, HORIZON)
    print("targets shape:", targets.shape)

    print("=== 5. Объединение + dropna + сплит ===")
    X = features.copy()
    y = targets.copy()
    dataset = pd.concat([X, y], axis=1)
    na_before = int(dataset.isna().sum().sum())
    print(f"Пропусков до dropna: {na_before:,}")
    dataset = dataset.dropna()
    print(f"После dropna: {len(dataset):,} строк")

    X_train, X_val, X_test = time_split(dataset[X.columns], SPLIT)
    y_train, y_val, y_test = time_split(dataset[y.columns], SPLIT)
    print(f"Train: X{X_train.shape} y{y_train.shape}")
    print(f"Val:   X{X_val.shape} y{y_val.shape}")
    print(f"Test:  X{X_test.shape} y{y_test.shape}")

    print("=== 6. Сохранение в parquet ===")
    X_train.to_parquet(out_dir / "X_train.parquet")
    y_train.to_parquet(out_dir / "y_train.parquet")
    X_val.to_parquet(out_dir / "X_val.parquet")
    y_val.to_parquet(out_dir / "y_val.parquet")
    X_test.to_parquet(out_dir / "X_test.parquet")
    y_test.to_parquet(out_dir / "y_test.parquet")
    print("Сохранено в", out_dir.resolve())


if __name__ == "__main__":
    main()
