import csv
import time
import aiofiles
from io import StringIO
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.schemas import CourierInput, PredictionOutput, BatchInput, BatchOutput, HealthResponse
from app.ml_engine import ml_engine
from app.config import UPLOAD_DIR, BATCH_THRESHOLD, setup_logger


logger = setup_logger("router.predict")
router = APIRouter(tags=["predict"])
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=ml_engine.model_loaded,
        uptime_seconds=round(time.time() - START_TIME, 2)
    )


@router.post("/predict/single", response_model=PredictionOutput)
async def predict_single(data: CourierInput):
    deficit = ml_engine.predict_single(data.orders, data.couriers)
    return PredictionOutput(deficit=deficit, orders=data.orders, couriers=data.couriers)


@router.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(data: BatchInput):
    start = time.perf_counter()
    items = [item.to_tuple() for item in data.items]
    results = await ml_engine.predict_batch(items)
    elapsed = (time.perf_counter() - start) * 1000
    return BatchOutput(
        results=[PredictionOutput(deficit=d, orders=o, couriers=c) for (o, c), d in zip(items, results)],
        total=len(results),
        elapsed_ms=round(elapsed, 2)
    )


@router.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    start = time.perf_counter()
    
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Только CSV файлы")
    
    content = await file.read()
    if not content:
        raise HTTPException(400, "Пустой файл")
    
    for enc in ["utf-8-sig", "utf-8", "cp1251", "latin-1"]:
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise HTTPException(400, "Не удалось декодировать")
    
    reader = csv.DictReader(StringIO(text))
    rows = list(reader)
    
    if not rows:
        raise HTTPException(400, "CSV пуст")
    if "orders" not in (reader.fieldnames or []) or "couriers" not in (reader.fieldnames or []):
        raise HTTPException(400, "Нужны колонки orders и couriers")
    
    items, errors = [], []
    for i, row in enumerate(rows, start=2):
        try:
            o, c = int(str(row.get("orders", "")).strip()), int(str(row.get("couriers", "")).strip())
            if o < 0 or c < 0:
                errors.append({"row": i, "msg": "Отрицательные значения"})
                continue
            items.append((o, c))
        except (ValueError, TypeError):
            errors.append({"row": i, "msg": "Некорректные данные"})
    
    if not items:
        raise HTTPException(400, "Нет валидных строк")
    
    predictions = await ml_engine.predict_batch(items) if len(items) >= BATCH_THRESHOLD else [ml_engine.predict_single(o, c) for o, c in items]
    
    UPLOAD_DIR.mkdir(exist_ok=True)
    out_path = UPLOAD_DIR / f"result_{Path(file.filename).stem}.csv"
    
    async with aiofiles.open(out_path, "w", newline="", encoding="utf-8") as f:
        await f.write("orders,couriers,deficit\n")
        for (o, c), d in zip(items, predictions):
            await f.write(f"{o},{c},{d}\n")
    
    elapsed = time.perf_counter() - start
    logger.info("csv | rows=%d errors=%d | %.3fs", len(items), len(errors), elapsed)
    
    return FileResponse(str(out_path), filename=f"result_{Path(file.filename).name}", media_type="text/csv",
                        headers={"X-Processing-Time": f"{elapsed:.3f}", "X-Rows": str(len(items))})