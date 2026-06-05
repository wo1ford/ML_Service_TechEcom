import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestBatchAPI:
    
    @pytest.mark.asyncio
    async def test_basic(self, client: AsyncClient, valid_batch):
        response = await client.post("/api/v1/predict/batch", json=valid_batch)
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert data["total"] == 3
        assert "elapsed_ms" in data
    
    @pytest.mark.asyncio
    async def test_single_item(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/batch",
            json={"items": [{"orders": 100, "couriers": 1}]}
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1
    
    @pytest.mark.asyncio
    async def test_large_batch(self, client: AsyncClient, large_batch):
        response = await client.post("/api/v1/predict/batch", json=large_batch)
        assert response.status_code == 200
        assert len(response.json()["results"]) == 100
    
    @pytest.mark.asyncio
    async def test_empty_rejected(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict/batch",
            json={"items": []}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_results_nonnegative(self, client: AsyncClient, valid_batch):
        response = await client.post("/api/v1/predict/batch", json=valid_batch)
        for r in response.json()["results"]:
            assert r["deficit"] >= 0
    
    @pytest.mark.asyncio
    async def test_results_have_echo(self, client: AsyncClient, valid_batch):
        response = await client.post("/api/v1/predict/batch", json=valid_batch)
        for r in response.json()["results"]:
            assert "orders" in r
            assert "couriers" in r