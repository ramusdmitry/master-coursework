# Live Forecast Demo

Веб-демо сравнения стратегий прогнозирования (BTC). **НЕ торгует** — предсказывает
следующее движение и ведёт бумажный учёт: PnL%, hit%, Sharpe. Лидер сверху.

Стратегии: **Buy & Hold**, **LSTM (наша модель)**, Momentum, Mean-Reversion,
MA-crossover, Random (пол). Модель — `models/model.pth` (1m, val ROC 0.516 ≈ монетка).

## Запуск

Replay (история ускоренно, для показа без ожидания):
```bash
cd <worktree>
DEMO_MODE=replay DEMO_TF=1m DEMO_SINCE=2025-09-20 \
  ~/.envs/ds/bin/uvicorn demo.server:app --port 8000
```

Live (реальный рынок Binance, бар каждую минуту):
```bash
DEMO_MODE=live DEMO_TF=1m \
  ~/.envs/ds/bin/uvicorn demo.server:app --port 8000
```

Открыть http://localhost:8000

## Архитектура
- `engine.py` — движок бумажного учёта + стратегии (общий для веба и консоли).
- `model_strategy.py` — LSTM-стратегия (15 фич из OHLCV-окна, как при обучении).
- `feed.py` — источники баров: `replay_feed` (CSV) и `live_feed` (Binance WS, aiohttp).
- `server.py` — FastAPI + SSE; live-режим прогревает фичи через REST.
- `static/index.html` — таблица + график накопл. PnL (Chart.js, EventSource).

Бэкенд не зависит от `websockets` (сервер→браузер через SSE; Binance→сервер через aiohttp).
