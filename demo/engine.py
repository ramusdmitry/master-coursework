"""
demo/engine.py — движок бумажного сравнения стратегий (общий для консоли и веба).

НЕ торгует. На каждый закрытый бар: реализует доходность от ПРОШЛОГО сигнала каждой
стратегии, обновляет накопленный PnL%, hit% (доля угаданных направлений) и Sharpe,
затем считает новый сигнал на следующий бар. Сигнал: +1 long / −1 short / 0 flat.

Стратегии без модели (чистая цена); LSTM-стратегия добавляется отдельно (model.py).
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Optional


class Strategy:
    """База: получает историю баров (list of {o,h,l,c,v}) -> сигнал next-бар (+1/-1/0)."""
    name = "base"

    def signal(self, bars: list[dict]) -> int:
        raise NotImplementedError


def _closes(bars): return [b["c"] for b in bars]


class BuyHold(Strategy):
    name = "Buy & Hold"
    def signal(self, bars): return +1


class Momentum(Strategy):
    def __init__(self, n=12): self.n = n; self.name = f"Momentum({n})"
    def signal(self, bars):
        c = _closes(bars)
        if len(c) <= self.n: return +1
        return +1 if c[-1] > c[-1 - self.n] else -1


class MeanReversion(Strategy):
    def __init__(self, n=1): self.n = n; self.name = "Mean-Reversion"
    def signal(self, bars):
        c = _closes(bars)
        if len(c) <= self.n: return 0
        return -1 if c[-1] > c[-1 - self.n] else +1   # фейдим последнее движение


class MACrossover(Strategy):
    def __init__(self, fast=12, slow=48):
        self.f, self.s = fast, slow; self.name = f"MA-cross({fast}/{slow})"
    def signal(self, bars):
        c = _closes(bars)
        if len(c) < self.s: return +1
        return +1 if sum(c[-self.f:]) / self.f > sum(c[-self.s:]) / self.s else -1


class RandomStrat(Strategy):
    name = "Random (пол)"
    def __init__(self, seed=0): self.r = random.Random(seed)
    def signal(self, bars): return self.r.choice([+1, -1])


class PaperAccount:
    """Бумажный учёт одной стратегии: позиция, доходности, метрики."""
    def __init__(self, strat: Strategy, per_year: int = 24 * 365):
        self.strat = strat
        self.per_year = per_year
        self.pos = 0            # текущая позиция (сигнал, выставленный на этот бар)
        self.rets: list[float] = []   # реализованные по-баровые доходности стратегии
        self.hits = 0
        self.dir_count = 0      # сколько баров с ненулевой позицией (для hit%)
        self.next_signal = 0    # прогноз на следующий бар

    def on_bar(self, bar_ret: float):
        """Реализует PnL от позиции (=прошлый next_signal), считает hit."""
        r = self.pos * bar_ret
        self.rets.append(r)
        if self.pos != 0:
            self.dir_count += 1
            if (self.pos > 0) == (bar_ret > 0):
                self.hits += 1

    def set_next(self, sig: int):
        self.next_signal = sig
        self.pos = sig          # позиция на следующий бар = прогноз

    @property
    def pnl_pct(self) -> float:
        return 100.0 * (math.exp(sum(self.rets)) - 1.0) if self.rets else 0.0

    @property
    def hit_pct(self) -> Optional[float]:
        if isinstance(self.strat, BuyHold) or self.dir_count == 0:
            return None
        return 100.0 * self.hits / self.dir_count

    @property
    def sharpe(self) -> float:
        n = len(self.rets)
        if n < 2: return 0.0
        mean = sum(self.rets) / n
        var = sum((x - mean) ** 2 for x in self.rets) / (n - 1)
        sd = math.sqrt(var)
        return (mean / sd) * math.sqrt(self.per_year) if sd > 0 else 0.0


class Book:
    """Набор стратегий + единая лента баров. Возвращает снимок состояния на каждый бар."""
    def __init__(self, strategies: list[Strategy], per_year: int = 24 * 365):
        self.accts = [PaperAccount(s, per_year) for s in strategies]
        self.bars: list[dict] = []
        self.bar = 0

    def step(self, bar) -> dict:
        """Подать новый закрытый бар (dict {o,h,l,c,v} или число close)."""
        if not isinstance(bar, dict):
            bar = {"o": bar, "h": bar, "l": bar, "c": bar, "v": 0.0}
        close = bar["c"]
        if self.bars:
            bar_ret = math.log(close / self.bars[-1]["c"])
            for a in self.accts:
                a.on_bar(bar_ret)
        self.bars.append(bar)
        self.bar += 1
        for a in self.accts:
            a.set_next(a.strat.signal(self.bars))
        return self.snapshot(close)

    def snapshot(self, close: float) -> dict:
        rows = []
        for a in self.accts:
            rows.append({
                "name": a.strat.name,
                "signal": a.next_signal,          # +1/-1/0 прогноз
                "position": a.pos,
                "pnl_pct": round(a.pnl_pct, 2),
                "hit_pct": None if a.hit_pct is None else round(a.hit_pct, 1),
                "sharpe": round(a.sharpe, 2),
            })
        rows.sort(key=lambda x: x["pnl_pct"], reverse=True)   # лидер сверху
        return {"bar": self.bar, "price": close, "rows": rows}


def default_strategies() -> list[Strategy]:
    return [BuyHold(), Momentum(12), MeanReversion(), MACrossover(12, 48), RandomStrat(42)]
