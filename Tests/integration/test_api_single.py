import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestSingleAPI:
    
    @pytest.mark.asyncio
    async def test_basic(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": 250, "couriers": 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deficit"] == 1
        assert data["orders"] == 250
        assert data["couriers"] == 2
    
    @pytest.mark.asyncio
    async def test_zero_deficit(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": 50, "couriers": 5}
        )
        assert response.json()["deficit"] == 0
    
    @pytest.mark.asyncio
    async def test_large_numbers(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": 10000, "couriers": 0}
        )
        assert response.json()["deficit"] == 100
    
    @pytest.mark.asyncio
    async def test_zero_both(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": 0, "couriers": 0}
        )
        assert response.json()["deficit"] == 0
    
    @pytest.mark.asyncio
    async def test_negative_orders_rejected(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": -10, "couriers": 2}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_negative_couriers_rejected(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": 100, "couriers": -5}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_missing_orders(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"couriers": 5}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_missing_couriers(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": 100}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_orders_too_large(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": 2_000_000, "couriers": 5}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_string_values_accepted(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/single",
            json={"orders": "100", "couriers": "5"}
        )
        assert response.status_code == 200