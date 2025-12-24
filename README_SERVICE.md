# ML Service для прогнозирования криптовалют

Полноценный ML-сервис на FastAPI для прогнозирования направления цены криптовалют на основе временных рядов.

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Создайте файл `.env` на основе `.env.example`:
```bash
cp .env.example .env
```

3. Выполните миграции базы данных:
```bash
alembic upgrade head
```

4. Создайте первого администратора:
```bash
python scripts/create_admin.py --username admin --password your_password
```

## Запуск сервиса

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Сервис будет доступен по адресу: http://localhost:8000

Документация API доступна по адресу: http://localhost:8000/docs

## Эндпоинты

### POST /forward
Предсказание на основе JSON данных (без изображений).

**Запрос:**
```json
{
  "data": [[...фичи...], [...фичи...], ...],  // минимум 60 временных точек
  "asset": "BTC"  // опционально
}
```

**Ответ:**
```json
{
  "prediction": 0,
  "probabilities": {
    "class_0": 0.52,
    "class_1": 0.48
  },
  "asset": "BTC"
}
```

### POST /forward (с изображением)
Предсказание на основе изображения (multipart/form-data).

**Запрос:**
- Content-Type: `multipart/form-data`
- Форма с полем `image` (файл изображения)
- Заголовки:
  - `X-Asset`: актив (опционально)
  - `X-Window-Size`: размер окна (опционально)

**Ответ:**
```json
{
  "image_base64": "...",
  "prediction": null,
  "metadata": {
    "width": 800,
    "height": 600,
    "format": "PNG",
    "size_bytes": 12345
  }
}
```

### GET /history
Получение истории всех запросов.

**Параметры запроса:**
- `skip`: количество записей для пропуска (по умолчанию 0)
- `limit`: максимальное количество записей (по умолчанию 100)

### DELETE /history
Удаление истории запросов (требует авторизации администратора).

**Заголовки:**
- `Authorization: Bearer <JWT_TOKEN>`
- `X-Delete-Token: DELETE_CONFIRM_TOKEN`

### GET /stats
Статистика запросов (PRO).

**Ответ:**
```json
{
  "total_requests": 100,
  "successful_requests": 95,
  "failed_requests": 5,
  "average_processing_time_ms": 12.5,
  "processing_time_quantiles": {
    "mean": 12.5,
    "50%": 10.0,
    "95%": 25.0,
    "99%": 50.0
  },
  "input_size_stats": {
    "mean": 900.0,
    "min": 60,
    "max": 1200,
    "median": 900.0
  }
}
```

### POST /token
Получение JWT токена для авторизации.

**Параметры:**
- `username`: имя пользователя
- `password`: пароль

### POST /users
Создание нового пользователя (требует авторизации администратора).

**Запрос:**
```json
{
  "username": "user1",
  "password": "password123",
  "is_admin": false
}
```

## Авторизация

Для использования защищенных эндпоинтов необходимо:

1. Получить JWT токен через `/token`
2. Передавать токен в заголовке `Authorization: Bearer <token>`

## Миграции базы данных

Создание новой миграции:
```bash
alembic revision --autogenerate -m "описание изменений"
```

Применение миграций:
```bash
alembic upgrade head
```

Откат миграции:
```bash
alembic downgrade -1
```

## Структура проекта

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # Основное приложение FastAPI
│   ├── database.py           # Настройка БД
│   ├── models.py             # Модели SQLAlchemy
│   ├── schemas.py            # Pydantic схемы
│   ├── ml_model.py           # ML модель и сервис
│   └── auth.py               # JWT авторизация
├── alembic/                 # Миграции Alembic
│   ├── versions/
│   └── env.py
├── alembic.ini              # Конфигурация Alembic
├── requirements.txt          # Зависимости
├── .env.example             # Пример переменных окружения
└── README_SERVICE.md        # Документация
```

## Примеры использования

### JSON запрос (curl)
```bash
curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[...60 временных точек по 15 фичей...]],
    "asset": "BTC"
  }'
```

### Запрос с изображением (curl)
```bash
curl -X POST "http://localhost:8000/forward" \
  -H "Content-Type: multipart/form-data" \
  -H "X-Asset: BTC" \
  -F "image=@path/to/image.png"
```

### Получение истории
```bash
curl "http://localhost:8000/history?skip=0&limit=10"
```

### Получение статистики
```bash
curl "http://localhost:8000/stats"
```

### Получение JWT токена
```bash
curl -X POST "http://localhost:8000/token?username=admin&password=your_password"
```

### Удаление истории (требует авторизации)
```bash
curl -X DELETE "http://localhost:8000/history" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "X-Delete-Token: DELETE_CONFIRM_TOKEN"
```

## Примечания

- Модель по умолчанию создается без обученных весов. Для продакшена необходимо загрузить обученную модель в `models/model.pth` и scaler в `models/scaler.pkl`.
- Для работы с изображениями требуется дополнительная реализация обработки через модель для изображений.
- В продакшене необходимо изменить `SECRET_KEY` и `DELETE_CONFIRM_TOKEN` в `.env`.
- Формат данных для JSON запроса: список из минимум 60 временных точек, каждая точка содержит 15 фичей (для BTC).

