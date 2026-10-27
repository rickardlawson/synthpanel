"""Innloggingen for demo-miljøet (krever ikke bygget populasjon)."""
from fastapi.testclient import TestClient

from synthpanel.api.main import app

client = TestClient(app)


def test_open_without_password(monkeypatch):
    monkeypatch.delenv("SYNTHPANEL_PASSWORD", raising=False)
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200


def test_requires_login_when_password_set(monkeypatch):
    monkeypatch.setenv("SYNTHPANEL_PASSWORD", "hemmelig")
    monkeypatch.delenv("SYNTHPANEL_USER", raising=False)
    assert client.get("/health").status_code == 200  # driftssjekken er alltid åpen
    resp = client.get("/docs")
    assert resp.status_code == 401
    assert "Basic" in resp.headers["www-authenticate"]
    assert client.get("/docs", auth=("demo", "feil")).status_code == 401
    assert client.get("/docs", auth=("demo", "hemmelig")).status_code == 200
