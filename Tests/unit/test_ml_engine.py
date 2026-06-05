import pytest
import time
from app.ml_engine import MLEngine, PredictionError


class TestMLEngine:
    
    def setup_method(self):
        self.engine = MLEngine()
    
    def test_model_loaded_property(self):
        """Свойство model_loaded — bool"""
        assert isinstance(self.engine.model_loaded, bool)
    
    def test_model_type_property(self):
        """Свойство model_type — str"""
        assert isinstance(self.engine.model_type, str)
        assert self.engine.model_type in ("fallback",) or "model" in self.engine.model_type.lower()
    
    def test_predict_single_returns_int(self):
        result = self.engine.predict_single(100, 1)
        assert isinstance(result, int)
    
    def test_predict_single_nonnegative(self):
        for orders, couriers in [(0, 0), (100, 5), (1000, 0), (1, 100)]:
            assert self.engine.predict_single(orders, couriers) >= 0
    
    def test_predict_single_performance(self):
        """Одно предсказание < 100ms"""
        start = time.perf_counter()
        for _ in range(100):
            self.engine.predict_single(100, 1)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100 предсказаний за {elapsed:.3f}s — слишком медленно"
    
    @pytest.mark.asyncio
    async def test_predict_batch_returns_list(self):
        items = [(100, 1), (300, 2), (50, 5)]
        results = await self.engine.predict_batch(items)
        assert len(results) == 3
        assert all(isinstance(r, int) for r in results)
        assert all(r >= 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_predict_batch_small(self):
        items = [(100, 1)]
        results = await self.engine.predict_batch(items)
        assert results == [self.engine.predict_single(100, 1)]
    
    @pytest.mark.asyncio
    async def test_predict_batch_large(self):
        items = [(i * 10, i) for i in range(1, 51)]
        results = await self.engine.predict_batch(items)
        assert len(results) == 50
    
    @pytest.mark.asyncio
    async def test_predict_batch_empty(self):
        results = await self.engine.predict_batch([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_predict_batch_consistency(self):
        items = [(100, 1), (300, 2), (50, 5)]
        batch_results = await self.engine.predict_batch(items)
        single_results = [self.engine.predict_single(o, c) for o, c in items]
        assert batch_results == single_results