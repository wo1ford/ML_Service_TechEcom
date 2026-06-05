# ML Prediction Service

Сервис прогнозирования дефицита курьеров на FastAPI. Взял и сделал простую модельку взяв идею с сореванования Kaggle [Courier Deficit Regression Challenge](https://www.kaggle.com/competitions/intern-regression-courier-deficit-challenge/).

## Запуск

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

http://localhost:8000 — интерфейс | http://localhost:8000/docs — Swagger  ( для полноты картины )

## Docker

```bash
python train_model.py
docker-compose up -d
```

## API

| Метод | Путь | Что делает |
|-------|------|------------|
| GET | `/api/v1/health` | Проверка, живой ли сервер |
| POST | `/api/v1/predict/single` | Одно предсказание |
| POST | `/api/v1/predict/batch` | Много предсказаний разом |
| POST | `/api/v1/predict/csv` | Загрузить CSV, получить CSV с ответом |

**Single:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"orders": 250, "couriers": 2}'
# → {"deficit": 1, "orders": 250, "couriers": 2}
```

**Batch:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"orders": 100, "couriers": 1}, {"orders": 300, "couriers": 2}]}'
```

**CSV:** на вход `orders,couriers`, на выходе `orders,couriers,deficit`.

## Модель

GradientBoostingRegressor из scikit-learn. Если модель не загружена — запасная: `max(0, ceil(orders / 100) - couriers)`.

## boundы)

**I/O-bound** — работа с диском/сетью. Процессор ждёт. В сервисе это чтение и запись CSV. Используется `aiofiles` — файлы читаются асинхронно, сервер не зависает.

**CPU-bound** — тяжёлые вычисления. Грузят процессор. В сервисе это предсказания модели для большого количества записей. Используется `ProcessPoolExecutor` — вычисления уходят в отдельные процессы и не мешают серверу отвечать на другие запросы.

До 10 записей — считается в основном потоке (быстрее). 10 и больше — через отдельные процессы (не блокирует)
