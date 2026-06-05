import pytest
from app.ml_engine import MLEngine


class TestFallbackFormula:
    """Тесты заглушки"""
    
    def setup_method(self):
        self.engine = MLEngine()
    
    # Базовые случаи
    def test_basic_deficit(self):
        """250 заказов, 2 курьера => ceil(250/100) - 2 = 1"""
        assert self.engine._fallback_predict(250, 2) == 1
    
    def test_zero_deficit(self):
        """50 заказов, 5 курьеров => дефицита нет"""
        assert self.engine._fallback_predict(50, 5) == 0
    
    def test_exact_match(self):
        """200 заказов, 2 курьера => ровно 0"""
        assert self.engine._fallback_predict(200, 2) == 0
    
    def test_large_deficit(self):
        """10000 заказов, 0 курьеров => 100"""
        assert self.engine._fallback_predict(10000, 0) == 100
    
    def test_zero_orders(self):
        """0 заказов => дефицит 0"""
        assert self.engine._fallback_predict(0, 10) == 0
    
    def test_zero_both(self):
        """0 заказов и 0 курьеров"""
        assert self.engine._fallback_predict(0, 0) == 0
    
    # Грань случаи
    def test_one_order(self):
        """1 заказ, 0 курьеров => ceil(1/100) - 0 = 1"""
        assert self.engine._fallback_predict(1, 0) == 1
    
    def test_99_orders(self):
        """99 заказов, 0 курьеров => ceil(99/100) = 1"""
        assert self.engine._fallback_predict(99, 0) == 1
    
    def test_101_orders(self):
        """101 заказ, 0 курьеров => ceil(101/100) = 2"""
        assert self.engine._fallback_predict(101, 0) == 2
    
    def test_rounding_up(self):
        """ceil всегда вверх"""
        assert self.engine._fallback_predict(100, 0) == 1
        assert self.engine._fallback_predict(101, 0) == 2
        assert self.engine._fallback_predict(199, 0) == 2
        assert self.engine._fallback_predict(200, 0) == 2
        assert self.engine._fallback_predict(201, 0) == 3
    
    # Крайние значения
    def test_max_values(self):
        """Макс допустимые значения"""
        result = self.engine._fallback_predict(1_000_000, 0)
        assert result == 10_000
    
    def test_many_couriers(self):
        """Курьеров больше чем нужно"""
        assert self.engine._fallback_predict(100, 100) == 0
    
    # Свойства
    def test_result_always_nonnegative(self):
        """Результат всегда >= 0"""
        import random
        random.seed(42)
        for _ in range(100):
            orders = random.randint(0, 10000)
            couriers = random.randint(0, 1000)
            assert self.engine._fallback_predict(orders, couriers) >= 0
    
    def test_monotonic_orders(self):
        """Больше заказов => не меньше дефицит (при фикс. курьерах :) )"""
        for orders in [0, 50, 100, 150, 200, 500, 1000]:
            assert self.engine._fallback_predict(orders, 5) <= self.engine._fallback_predict(orders + 100, 5)
    
    def test_monotonic_couriers(self):
        """Больше курьеров => не больше дефицит (при фикс. заказах ;) )"""
        for couriers in [0, 1, 5, 10, 50]:
            assert self.engine._fallback_predict(500, couriers) >= self.engine._fallback_predict(500, couriers + 1)
    
    # Невалидные входы
    def test_negative_orders_raises(self):
        """Отрицательные заказы => исключение"""
        with pytest.raises(ValueError):
            self.engine._fallback_predict(-10, 5)
    
    def test_negative_couriers_raises(self):
        """Отрицательные курьеры => исключение"""
        with pytest.raises(ValueError):
            self.engine._fallback_predict(100, -5)
    
    def test_both_negative_raises(self):
        """Оба отрицательные => исключение"""
        with pytest.raises(ValueError):
            self.engine._fallback_predict(-10, -5)