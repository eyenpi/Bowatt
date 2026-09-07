from fastapi.testclient import TestClient


def test_health_reports_agent_mode(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Bowatt Research Agent API",
        "environment": "test",
        "mode": "agent",
    }
