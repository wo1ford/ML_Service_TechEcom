import io
import pytest
from httpx import AsyncClient


@pytest.mark.e2e
class TestFullFlow:
    
    @pytest.mark.asyncio
    async def test_single_then_batch(self, client: AsyncClient):
        # Single
        r1 = await client.post(
            "/api/v1/predict/single",
            json={"orders": 300, "couriers": 2}
        )
        assert r1.status_code == 200
        single_deficit = r1.json()["deficit"]
        
        # Batch с теми же данными
        r2 = await client.post(
            "/api/v1/predict/batch",
            json={"items": [{"orders": 300, "couriers": 2}]}
        )
        assert r2.status_code == 200
        batch_deficit = r2.json()["results"][0]["deficit"]
        
        # Должны совпадать \(-_-)/
        assert single_deficit == batch_deficit
    
    @pytest.mark.asyncio
    async def test_csv_and_batch_consistency(self, client: AsyncClient):
        import csv
        from io import StringIO
        
        # Через batch
        batch_data = {
            "items": [
                {"orders": 100, "couriers": 1},
                {"orders": 300, "couriers": 2},
                {"orders": 50, "couriers": 5}
            ]
        }
        r1 = await client.post("/api/v1/predict/batch", json=batch_data)
        batch_results = [r["deficit"] for r in r1.json()["results"]]
        
        # Через CSV
        csv_content = "orders,couriers\n100,1\n300,2\n50,5\n"
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        r2 = await client.post("/api/v1/predict/csv", files=files)
        
        reader = csv.DictReader(StringIO(r2.text))
        csv_results = [int(row["deficit"]) for row in reader]
        
        assert batch_results == csv_results
    
    @pytest.mark.asyncio
    async def test_health_before_and_after(self, client: AsyncClient):
        # До
        r1 = await client.get("/api/v1/health")
        uptime_before = r1.json()["uptime_seconds"]
        
        # Нагрузка
        for _ in range(10):
            await client.post(
                "/api/v1/predict/single",
                json={"orders": 100, "couriers": 1}
            )
        
        # После
        r2 = await client.get("/api/v1/health")
        uptime_after = r2.json()["uptime_seconds"]
        
        assert uptime_after >= uptime_before
        assert r2.json()["status"] == "healthy"