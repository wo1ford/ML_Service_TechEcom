import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestHealthAPI:
    
    @pytest.mark.asyncio
    async def test_health_ok(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_has_model_field(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)
    
    @pytest.mark.asyncio
    async def test_health_has_version(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_health_has_uptime(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0
    
    @pytest.mark.asyncio
    async def test_health_response_time(self, client: AsyncClient):
        import time
        start = time.perf_counter()
        await client.get("/api/v1/health")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
    
    @pytest.mark.asyncio
    async def test_health_content_type(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert "application/json" in response.headers["content-type"]