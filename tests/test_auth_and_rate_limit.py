import os

from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def test_auth_required_rejects_missing_bearer_token(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_required", True)
    monkeypatch.setenv("API_TOKEN", "secret-token")

    response = client.get("/health")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required."


def test_auth_required_accepts_valid_bearer_token(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_required", True)
    monkeypatch.setenv("API_TOKEN", "secret-token")

    response = client.get("/health", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rate_limit_limits_requests_per_ip(monkeypatch):
    monkeypatch.setattr(main.settings, "rate_limit_per_minute", 2)
    monkeypatch.setattr(main.settings, "auth_required", False)

    main.clear_rate_limit_state()

    first = client.get("/health")
    second = client.get("/health")
    third = client.get("/health")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["detail"] == "Rate limit exceeded. Please try again later."
