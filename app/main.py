import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers.predict import router as predict_router
from app.ml_engine import ml_engine
from app.config import setup_logger


logger = setup_logger("main")

app = FastAPI(title="ML Prediction Service", version="1.0.0", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info("%s %s | %d | %.3fs", request.method, request.url.path, response.status_code, time.time() - start)
    return response


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("app/static/index.html")


app.include_router(predict_router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("ML Prediction Service v1.0.0")
    logger.info("Модель: %s", "загружена" if ml_engine.model_loaded else "fallback")
    logger.info("=" * 50)