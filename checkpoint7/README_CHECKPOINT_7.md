# Чекпоинт 7 — Наблюдаемость модели: MLflow + MinIO (S3)

Финальный чекпоинт годового проекта по краткосрочному предсказанию цены криптовалют.
На этом этапе модель и сервис делаются наблюдаемыми: параметры, метрики и артефакты
экспериментов фиксируются в **MLflow**, артефакты хранятся в **MinIO (S3)**, проводится
структурированный анализ ошибок, сравнение с baseline и проверка устойчивости.

Финальная (PRD) модель — **Meta-Labeling из чекпоинта 6**: первичная модель `SimpleLSTM`
задаёт направление сделки, вторичная `GradientBoostingClassifier` фильтрует ложные
срабатывания и управляет размером позиции (даёт лучший net-Sharpe в CP6).

---

## Структура проекта

```
checkpoint7/
├── README.md                       # этот файл
├── docker-compose.yml              # MLflow + MinIO + инициализация бакета
├── Dockerfile.mlflow               # образ MLflow с boto3/psycopg2
├── .env.example                    # шаблон переменных окружения
├── requirements.txt                # Python-зависимости
├── conf/                           # Hydra-конфиги (бонус: CLI + .yaml)
│   ├── config.yaml                 # корневой конфиг (defaults, seed)
│   ├── data/default.yaml           # путь к данным, актив, флаг синтетики
│   ├── model/meta_labeling.yaml    # гиперпараметры LSTM + GBDT
│   └── mlflow/local.yaml           # tracking_uri, эксперимент, имя модели, PRD-тег
├── src/
│   ├── common.py                   # утилиты, константы, модель SimpleLSTM, метрики
│   ├── model.py                    # MetaLabelingModel + pyfunc-обёртка
│   ├── train.py                    # обучение + MLflow + регистрация PRD (CLI/Hydra)
│   └── predict_prd.py              # загрузка PRD-модели + предикт (CLI/Hydra)
└── notebooks/
    ├── checkpoint-7-mlflow.ipynb       # research: обучение, логирование, анализ ошибок
    └── checkpoint-7-load-prd.ipynb     # чистый блокнот: загрузка PRD + тестовый предикт
```

---

## Предварительные требования

- Docker и Docker Compose
- Python 3.11 (для запуска скриптов/ноутбуков вне контейнера)
- Данные проекта в `data/processed/*.parquet` (`X_train/val/test`, `y_train/val/test`).
  Если данных нет — пайплайн автоматически использует синтетические данные
  (`make_synthetic_splits`), чтобы всё запускалось для демонстрации.

---

## Шаг 0. Установка зависимостей

```bash
cd checkpoint7
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Шаг 1. Поднять инфраструктуру (MLflow + MinIO) в Docker

1. Скопируйте шаблон переменных окружения и при необходимости поменяйте пароли:

   ```bash
   cp .env.example .env
   ```

2. Запустите контейнеры:

   ```bash
   docker compose up -d --build
   ```

   Поднимутся три сервиса:
   - **MinIO** — S3-хранилище: API на `http://localhost:9000`, веб-консоль на `http://localhost:9001`
     (логин/пароль из `.env`). Контейнер `minio-init` автоматически создаёт бакет `mlflow-artifacts`.
   - **MLflow** — сервер экспериментов на `http://localhost:5000`
     (backend store — SQLite, artifact store — `s3://mlflow-artifacts/` в MinIO).

3. Проверьте, что всё работает:
   - MLflow UI: открыть [http://localhost:5000](http://localhost:5000)
   - MinIO Console: открыть [http://localhost:9001](http://localhost:9001) → должен быть бакет `mlflow-artifacts`

> Для продакшн-варианта backend store SQLite можно заменить на Postgres
> (добавить сервис `postgres` и сменить `--backend-store-uri` в `docker-compose.yml`).

---

## Шаг 2. Настроить подключение ресёрч-блокнота к MLflow

Перед запуском кода вне контейнера экспортируйте переменные (или они подхватятся из `.env`):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
```

В ноутбуках эти значения уже задаются в начале. Если сервер не поднят, ноутбуки
автоматически переключаются на локальную папку `./mlruns` (чтобы их можно было
исполнить и без Docker для проверки логики).

---

## Шаг 3. Флоу версионирования эксперимента

Соответствие пунктам задания:

| Пункт задания | Где выполнено |
|---|---|
| 1. Выбор лучшей модели + обоснование, фиксация финальной версии, тег PRD | `notebooks/checkpoint-7-mlflow.ipynb`, раздел «Выбор лучшей модели»; тег PRD выставляется в `src/train.py` |
| 2. Переобучить выбранную модель с MLflow | `src/train.py` и раздел «Переобучение» в ноутбуке |
| 3. Логировать гиперпараметры и метрики (train/val/test) | `mlflow.log_params` / `mlflow.log_metrics` |
| 4. Сохранить артефакты в S3 (модель, графики, примеры) | confusion matrix, learning curve, sample_predictions.csv + pyfunc-модель → MinIO |
| 5. Воспроизводимость (seed, параметры, описание данных) | `set_seed`, логирование всех гиперов и описания данных |
| 6. Анализ ошибок (10–20 примеров) | раздел «Анализ ошибок» в ноутбуке |
| 7. Сравнение с baseline | раздел «Сравнение с baseline» (Buy & Hold + простое правило) |
| 8. Проверка устойчивости (robustness) | раздел «Проверка устойчивости» |
| 9. Чистый блокнот: загрузка PRD + предикт | `notebooks/checkpoint-7-load-prd.ipynb` |

### Вариант А. Через ресёрч-блокнот

Откройте и выполните `notebooks/checkpoint-7-mlflow.ipynb`. Он:
переобучает Meta-Labeling, логирует параметры/метрики/артефакты в MLflow+S3,
регистрирует модель в Model Registry и выставляет тег/alias **PRD**, затем проводит
анализ ошибок, сравнение с baseline и проверку устойчивости.

### Вариант Б (бонус). Через .py-скрипт и Hydra-конфиг из CLI

Запуск обучения управляется конфигом `conf/` (Hydra). Из корня `checkpoint7/`:

```bash
# базовый запуск
python -m src.train

# переопределение гиперпараметров прямо из CLI (возможности Hydra)
python -m src.train model.epochs=50 model.hidden_size=128 model.lr=5e-4 seed=123

# запуск на синтетических данных (без реальных parquet)
python -m src.train data.synthetic=true
```

Скрипт создаёт run в MLflow, логирует всё, регистрирует модель и помечает версию тегом PRD.

---

## Шаг 4. Загрузка PRD-модели и тестовый предикт (пункт 9)

### Вариант А. Чистый блокнот

Откройте и выполните `notebooks/checkpoint-7-load-prd.ipynb` — он загружает модель
по алиасу `models:/crypto_meta_labeling@PRD` из MLflow Registry и делает тестовый предикт.

### Вариант Б (бонус). Через .py-скрипт из CLI

```bash
python -m src.predict_prd
```

Скрипт находит версию модели с тегом/алиасом PRD, загружает её и печатает предсказания.

---

## Бонус: CLI + Hydra

Бонусная часть задания выполнена полностью:
- Запуск экспериментов — `.py`-скриптом (`src/train.py`) через CLI, гиперпараметры
  управляются Hydra-конфигами в `conf/*.yaml` и переопределяются из командной строки.
- Тестовый запуск PRD-модели — также `.py`-скриптом (`src/predict_prd.py`) через CLI.

Документация Hydra: [hydra.cc/docs/intro](https://hydra.cc/docs/intro/).

---

## Воспроизводимость

- Глобальный `seed=42` (Hydra `seed`, `set_seed`) фиксирует NumPy/PyTorch/random.
- Все гиперпараметры и описание данных логируются в MLflow как параметры run.
- Способ получения данных: хронологические сплиты `data/processed/*.parquet`
  (признаки с префиксом `BTC__`, целевые `y_bin` / `y_reg`); при отсутствии —
  воспроизводимая синтетика через `make_synthetic_splits(seed=42)`.

## Остановка

```bash
docker compose down            # остановить сервисы (данные сохранятся в томах)
docker compose down -v         # остановить и удалить тома (полная очистка)
```
