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


def test_segments_endpoint_returns_data(client):
    response = client.get("/segments")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload


def test_predict_accepts_valid_input(client):
    response = client.post(
        "/predict",
        json={
            "first_purchase_date": "2020-01-01",
            "last_purchase_date": "2024-01-01",
            "total_orders": 5,
            "total_spent": 250.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action_code"] in {"retain", "let_go", "nurture", "monitor"}
