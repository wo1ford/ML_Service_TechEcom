import pytest
from pydantic import ValidationError
from app.schemas import CourierInput, BatchInput, PredictionOutput, HealthResponse


class TestCourierInput:
    
    def test_valid(self):
        obj = CourierInput(orders=100, couriers=5)
        assert obj.orders == 100
        assert obj.couriers == 5
    
    def test_min_values(self):
        obj = CourierInput(orders=0, couriers=0)
        assert obj.orders == 0
    
    def test_max_values(self):
        obj = CourierInput(orders=1_000_000, couriers=100_000)
        assert obj.orders == 1_000_000
    
    def test_negative_orders_rejected(self):
        with pytest.raises(ValidationError):
            CourierInput(orders=-1, couriers=5)
    
    def test_negative_couriers_rejected(self):
        with pytest.raises(ValidationError):
            CourierInput(orders=100, couriers=-1)
    
    def test_orders_too_large(self):
        with pytest.raises(ValidationError):
            CourierInput(orders=2_000_000, couriers=5)
    
    def test_string_coercion(self):
        obj = CourierInput(orders="100", couriers="5")
        assert obj.orders == 100
        assert isinstance(obj.orders, int)
    
    def test_to_tuple(self):
        obj = CourierInput(orders=100, couriers=5)
        assert obj.to_tuple() == (100, 5)
    
    def test_missing_orders(self):
        with pytest.raises(ValidationError):
            CourierInput(couriers=5)
    
    def test_missing_couriers(self):
        with pytest.raises(ValidationError):
            CourierInput(orders=100)
    
    def test_empty_object(self):
        with pytest.raises(ValidationError):
            CourierInput()


class TestBatchInput:
    
    def test_valid(self):
        obj = BatchInput(items=[
            CourierInput(orders=100, couriers=1),
            CourierInput(orders=200, couriers=2)
        ])
        assert len(obj.items) == 2
    
    def test_single_item(self):
        obj = BatchInput(items=[CourierInput(orders=100, couriers=1)])
        assert len(obj.items) == 1
    
    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            BatchInput(items=[])
    
    def test_too_many_items(self):
        with pytest.raises(ValidationError):
            BatchInput(items=[CourierInput(orders=1, couriers=1)] * 100_001)
    
    def test_invalid_item(self):
        with pytest.raises(ValidationError):
            BatchInput(items=[{"orders": -1, "couriers": 1}])


class TestPredictionOutput:
    
    def test_valid(self):
        obj = PredictionOutput(deficit=5)
        assert obj.deficit == 5
    
    def test_with_echo(self):
        obj = PredictionOutput(deficit=5, orders=250, couriers=2)
        assert obj.deficit == 5
        assert obj.orders == 250
    
    def test_negative_deficit_rejected(self):
        with pytest.raises(ValidationError):
            PredictionOutput(deficit=-1)


class TestHealthResponse:
    
    def test_valid(self):
        obj = HealthResponse(status="healthy", model_loaded=True)
        assert obj.status == "healthy"
        assert obj.model_loaded is True