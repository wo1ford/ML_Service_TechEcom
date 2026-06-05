import io
import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestCSVAPI:
    
    @pytest.mark.asyncio
    async def test_valid(self, client: AsyncClient, valid_csv):
        files = {"file": ("test.csv", io.BytesIO(valid_csv), "text/csv")}
        response = await client.post("/api/v1/predict/csv", files=files)
        assert response.status_code == 200
        assert "deficit" in response.text
    
    @pytest.mark.asyncio
    async def test_large_csv(self, client: AsyncClient, large_csv):
        files = {"file": ("large.csv", io.BytesIO(large_csv), "text/csv")}
        response = await client.post("/api/v1/predict/csv", files=files)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_no_file(self, client: AsyncClient):
        response = await client.post("/api/v1/predict/csv")
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_wrong_extension(self, client: AsyncClient):
        files = {"file": ("test.txt", io.BytesIO(b"data"), "text/plain")}
        response = await client.post("/api/v1/predict/csv", files=files)
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_empty_csv(self, client: AsyncClient):
        files = {"file": ("empty.csv", io.BytesIO(b"orders,couriers\n"), "text/csv")}
        response = await client.post("/api/v1/predict/csv", files=files)
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_missing_column(self, client: AsyncClient):
        files = {"file": ("bad.csv", io.BytesIO(b"orders\n100\n"), "text/csv")}
        response = await client.post("/api/v1/predict/csv", files=files)
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_csv_with_errors(self, client: AsyncClient, csv_with_errors):
        files = {"file": ("errors.csv", io.BytesIO(csv_with_errors), "text/csv")}
        response = await client.post("/api/v1/predict/csv", files=files)
        # Должен обработать частично (2 хорошие строки из 4)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_response_has_headers(self, client: AsyncClient, valid_csv):
        files = {"file": ("test.csv", io.BytesIO(valid_csv), "text/csv")}
        response = await client.post("/api/v1/predict/csv", files=files)
        assert "X-Processing-Time" in response.headers
        assert "X-Rows-Processed" in response.headers
    
    @pytest.mark.asyncio
    async def test_result_csv_format(self, client: AsyncClient, valid_csv):
        import csv
        from io import StringIO
        
        files = {"file": ("test.csv", io.BytesIO(valid_csv), "text/csv")}
        response = await client.post("/api/v1/predict/csv", files=files)
        
        reader = csv.DictReader(StringIO(response.text))
        rows = list(reader)
        
        assert reader.fieldnames == ["orders", "couriers", "deficit"]
        assert len(rows) == 3
        for row in rows:
            assert int(row["deficit"]) >= 0