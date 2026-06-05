import pickle
import numpy as np
import pandas as pd
import math
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

from app.config import MODEL_PATH, FALLBACK_DIVISOR, MAX_WORKERS, setup_logger


logger = setup_logger("ml_engine")
FEATURES_PATH = Path(__file__).parent.parent / "model" / "features.pkl"


class MLEngine:

    def __init__(self) -> None:
        self._model = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        if not MODEL_PATH.exists():
            logger.warning("Модель не найдена: %s", MODEL_PATH)
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            self._model_loaded = True
            logger.info("Модель загружена")
        except Exception as e:
            logger.error("Ошибка загрузки модели: %s", e)

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def _build_features(self, orders: int, couriers: int) -> pd.DataFrame:
        is_peak, is_weekend, is_holiday, weather_severity = 1, 0, 0, 0
        avg_distance_km, active_orders_ratio = 5.0, 0.8
        orders_per_courier = orders / (couriers + 1)
        return pd.DataFrame([{
            "orders": orders, "couriers": couriers,
            "hour_sin": 0.0, "hour_cos": -1.0,
            "is_peak": is_peak, "is_weekend": is_weekend,
            "is_holiday": is_holiday, "city_zone": "center",
            "weather_severity": weather_severity,
            "avg_distance_km": avg_distance_km,
            "active_orders_ratio": active_orders_ratio,
            "orders_per_courier": orders_per_courier,
            "load_index": orders_per_courier * active_orders_ratio,
            "rush_factor": is_peak * orders_per_courier
        }])

    def _fallback_predict(self, orders: int, couriers: int) -> int:
        return max(0, math.ceil(orders / FALLBACK_DIVISOR) - couriers)

    def predict_single(self, orders: int, couriers: int) -> int:
        start = time.perf_counter()
        if self._model_loaded:
            try:
                df = self._build_features(orders, couriers)
                result = max(0, int(round(self._model.predict(df)[0])))
            except Exception:
                result = self._fallback_predict(orders, couriers)
        else:
            result = self._fallback_predict(orders, couriers)
        logger.debug("single | o=%-6d c=%-6d | d=%-4d | %.2fms", orders, couriers, result, (time.perf_counter()-start)*1000)
        return result

    @staticmethod
    def _predict_batch_worker(model_path: str, model_loaded: bool, batch: List[Tuple[int, int]]) -> List[int]:
        if model_loaded and Path(model_path).exists():
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                rows = [{
                    "orders": o, "couriers": c, "hour_sin": 0.0, "hour_cos": -1.0,
                    "is_peak": 1, "is_weekend": 0, "is_holiday": 0,
                    "city_zone": "center", "weather_severity": 0,
                    "avg_distance_km": 5.0, "active_orders_ratio": 0.8,
                    "orders_per_courier": o/(c+1),
                    "load_index": (o/(c+1))*0.8, "rush_factor": 1*(o/(c+1))
                } for o, c in batch]
                predictions = model.predict(pd.DataFrame(rows))
                return [max(0, int(round(p))) for p in predictions]
            except Exception:
                pass
        return [max(0, math.ceil(o/100)-c) for o, c in batch]

    async def predict_batch(self, items: List[Tuple[int, int]]) -> List[int]:
        n = len(items)
        if n < 10:
            return [self.predict_single(o, c) for o, c in items]
        logger.info("batch | size=%d", n)
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future = executor.submit(self._predict_batch_worker, str(MODEL_PATH), self._model_loaded, items)
            results = future.result(timeout=60)
        logger.info("batch | done | size=%d | %.2fms", n, (time.perf_counter()-start)*1000)
        return results


ml_engine = MLEngine()