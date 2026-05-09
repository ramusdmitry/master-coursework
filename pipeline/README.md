# Pipeline

Отдельный контур для экспериментов с моделями временных рядов.

---

## Передача коллеге: что уже есть и что дальше

### Сделано (текущий срез кода)

- **Каркас проекта:** `pipeline/configs/` (`base.yaml`, `hpo.yaml`), `pipeline/cli.py` с командами `prepare`, `train`, `evaluate`, `hpo`, `compare`, `export-backend`.
- **Конфигурация:** `pipeline/src/config.py` — загрузка YAML.
- **Фичи:** `pipeline/src/features/build.py` — базовые признаки (returns, скользящие средние).
- **Обучение и сравнение:** `pipeline/src/train/runner.py` — сценарий обучения, метрики, отчёты по прогонам, агрегация в `comparison.md` из JSON-отчётов `*_run_*.json`.
- **Метрики и отчёты:** `pipeline/src/evaluate/metrics.py`, `report.py`.
- **Трекинг:** `pipeline/src/tracking/wandb_tracker.py` — обёртка под Weights & Biases.
- **HPO:** `pipeline/src/hpo/optuna_search.py` — Optuna + связка с W&B (см. `configs/hpo.yaml`).
- **Совместимость с API:** `pipeline/src/compat/export_backend.py` — копирование `model.pth` / `scaler.pkl` в ожидаемый сервисом каталог `models/`.

### Критично: чего нет в дереве файлов, но ожидает код

`runner.py` импортирует модули, которых **пока нет** в репозитории (нужно реализовать или восстановить из ветки/архива):

- `pipeline/src/data/io.py` — загрузка сырых CSV по `data.input_glob` (функция `load_raw_series`).
- `pipeline/src/data/split.py` — временной сплит без утечки (`train_test_split_ts`).
- `pipeline/src/models/autots_model.py` — адаптер AutoTS (`fit` / `predict` / `metadata`).
- `pipeline/src/models/baseline.py` — наивный baseline, например `NaiveLastValueModel`.

Пока эти файлы не добавлены, команды `train` / `hpo` **не запустятся** (ошибка импорта).

### Что сделать дальше

1. Добавить каталоги `pipeline/src/data/` и `pipeline/src/models/` с перечисленными модулями; интерфейс уже задан вызовами в `runner.py` и `_build_model`.
2. Положить CSV в `data/raw/` по маске из `configs/base.yaml` (сейчас `data/raw/*_1h.csv`).
3. Выполнить `prepare` → `train` с `--model autots` → проверить `pipeline/reports/` (в т.ч. `autots_baseline.md` после прогона AutoTS).
4. Прогнать вторую модель (`naive_last_value`) и `compare` — должен появиться `comparison.md`.
5. Опционально: HPO через `hpo.yaml`, затем сверка прогонов в W&B.
6. По [плану](#оригинальный-план-кратко) ниже — при желании: перенос ноутбуков/`binance_spot_downloader.py` в `pipeline`, batch-матрица конфигов (сейчас `compare` ≈ пересборка отчёта из уже сохранённых JSON).

Полная версия плана в репозитории Cursor может лежать у автора по пути вроде `~/.cursor/plans/pipeline-autots-hpo_*.plan.md`; для коллеги достаточно разделов «оригинальный план» и «что сделать дальше» в этом файле.

---

## Оригинальный план (кратко)

**Цели:** независимый контур экспериментов; не трогать backend/API (`app/`, `alembic`); сначала этап **AutoTS First** (baseline, отчёт, потом остальное).

**Границы:** inference-контракт сервиса сохранять; экспериментальная логика — в `pipeline/`, параметры — через YAML.

**Этапы по смыслу:** (1) AutoTS baseline + W&B + отчёт; (2) конфигурируемый runner, при необходимости batch-сравнения; (3) Optuna HPO + ранжирование моделей в отчётах и W&B.

**Целевая структура (ориентир):** `configs/`, `src/data`, `src/features`, `src/models`, `src/train`, `src/evaluate`, `src/tracking`, `src/hpo`, `cli.py`, каталог отчётов/артефактов из конфига.

---

## Предварительные требования

- установить зависимости из `requirements.txt`;
- авторизоваться в W&B: `wandb login`;
- убедиться, что есть исходные CSV в `data/raw/*.csv`.

## Быстрый старт

```bash
python -m pipeline.cli prepare --config pipeline/configs/base.yaml
python -m pipeline.cli train --config pipeline/configs/base.yaml --model autots --run-name autots-baseline
python -m pipeline.cli train --config pipeline/configs/base.yaml --model naive_last_value --run-name naive-baseline
python -m pipeline.cli compare --config pipeline/configs/base.yaml
python -m pipeline.cli hpo --config pipeline/configs/hpo.yaml
```

## AutoTS First

1. Сначала запустить `autots-baseline`.
2. Проверить отчёт `pipeline/reports/autots_baseline.md`.
3. Только после этого запускать другие модели и HPO.

## Экспорт в текущий backend

Текущий сервис ожидает артефакты в `models/model.pth` и `models/scaler.pkl`.
Для совместимого экспорта:

```bash
python -m pipeline.cli export-backend \
  --source-model pipeline/artifacts/lstm/model.pth \
  --source-scaler pipeline/artifacts/lstm/scaler.pkl
```

Команда создаёт/обновляет:
- `models/model.pth`
- `models/scaler.pkl`
- `models/pipeline_export_manifest.json`
