"""
demo/model_strategy.py — LSTM-стратегия «наша модель» для демо.

Загружает models/model.pth (+scaler.pkl), считает 15 фич из окна баров так же, как
при обучении (build_features с pandas_ta: Wilder rsi_14/atr_14, ema span, vol_roll
DAY_N), прогоняет SimpleLSTM, возвращает +1 (P(up)>=0.5) или −1. Модель обучена на
1m (horizon 1мин, val ROC 0.516 ≈ монетка) — демо честно покажет, что edge нет.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from demo.engine import Strategy

# порядок 15 фич ровно как build_processed_data.build_features (ветка с pandas_ta)
COLS = ["open", "high", "low", "close", "volume", "ret_1", "ret_abs", "hl_spread",
        "oc_spread", "vol_roll", "rsi_14", "ema_12", "ema_26", "atr_14", "vol_ma_24"]


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


def _rma(s, n):   # Wilder RMA = ewm(alpha=1/n, adjust=False), как в pandas_ta
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def build15(bars: list[dict], day_n: int) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    o, h, l, c, v = df["o"], df["h"], df["l"], df["c"], df["v"]
    f = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    ret1 = np.log(c / c.shift(1))
    f["ret_1"] = ret1
    f["ret_abs"] = ret1.abs()
    f["hl_spread"] = (h - l) / c
    f["oc_spread"] = (o - c) / c
    f["vol_roll"] = ret1.rolling(day_n, min_periods=60).std() * np.sqrt(day_n)
    delta = c.diff()
    rs = _rma(delta.clip(lower=0), 14) / _rma(-delta.clip(upper=0), 14)
    f["rsi_14"] = 100 - 100 / (1 + rs)
    f["ema_12"] = c.ewm(span=12, adjust=False).mean()
    f["ema_26"] = c.ewm(span=26, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = _rma(tr, 14)
    f["vol_ma_24"] = v.rolling(24).mean()
    return f


class LSTMStrategy(Strategy):
    name = "LSTM (наша)"

    def __init__(self, model_path="models/model.pth", scaler_path="models/scaler.pkl",
                 window=60, day_n=1440):
        ck = torch.load(model_path, map_location="cpu")
        self.m = SimpleLSTM(ck["input_size"], ck["hidden_size"], ck["num_layers"], ck["dropout"])
        self.m.load_state_dict(ck["model_state_dict"]); self.m.eval()
        with open(scaler_path, "rb") as fh:
            self.sc = pickle.load(fh)
        # имена фич, на которых обучался scaler (BTC__open…); порядок = COLS
        self.feat_names = list(getattr(self.sc, "feature_names_in_", COLS))
        self.window = window
        self.day_n = day_n
        self.maxlb = window + 80   # хватает для всех окон, кроме vol_roll(min_periods=60)

    def signal(self, bars: list[dict]) -> int:
        if len(bars) < self.window + 65:
            return 0   # прогрев
        bars = bars[-(self.day_n + self.window + 5):]   # ограничить окно расчёта
        f = build15(bars, self.day_n)[COLS].dropna()
        if len(f) < self.window:
            return 0
        # DataFrame с именами как у scaler -> валидация совпадения, без warning
        win = f.iloc[-self.window:].copy()
        win.columns = self.feat_names
        X = self.sc.transform(win)
        with torch.no_grad():
            p = torch.softmax(self.m(torch.FloatTensor(X).unsqueeze(0)), 1)[0, 1].item()
        return +1 if p >= 0.5 else -1
