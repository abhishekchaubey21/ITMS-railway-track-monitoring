"""Test FastAPI health check endpoint."""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check():
    """Verify that GET /api/health returns 200 OK and expected status payload."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ITMS Backend"
    assert "version" in data
