import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("main", BACKEND_DIR / "main.py")
main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(main)


@pytest.fixture
def client():
    return TestClient(main.app)


def test_predict_rejects_bad_date_format(client):
    response = client.post(
        "/predict",
        json={
            "first_purchase_date": "bad-date",
            "last_purchase_date": "2024-01-01",
            "total_orders": 3,
            "total_spent": 100.0,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "Dates must be in YYYY-MM-DD format"


def test_health_reports_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
