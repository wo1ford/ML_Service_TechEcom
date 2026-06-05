import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

BASE_DIR: Path = Path(__file__).parent.parent
MODEL_PATH: Path = BASE_DIR / "model" / "model.pkl"
UPLOAD_DIR: Path = BASE_DIR / "uploads"
LOG_DIR: Path = BASE_DIR / "logs"

UPLOAD_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

FALLBACK_DIVISOR: int = 100
MAX_WORKERS: int = 4
BATCH_THRESHOLD: int = 10
HOST: str = "0.0.0.0"
PORT: int = 8000

# Логер
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger