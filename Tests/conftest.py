import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def valid_csv():
    return b"orders,couriers\n100,1\n300,2\n50,5\n"

@pytest.fixture
def csv_with_errors():
    return b"orders,couriers\n100,1\n-10,2\nabc,3\n50,5\n"

@pytest.fixture
def valid_batch():
    return {"items": [{"orders": 100, "couriers": 1}, {"orders": 300, "couriers": 2}, {"orders": 50, "couriers": 5}]}

@pytest.fixture
def large_batch():
    return {"items": [{"orders": i*10, "couriers": i} for i in range(1, 101)]}