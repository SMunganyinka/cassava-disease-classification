from pathlib import Path

import dashboard.app as dashboard
from streamlit.testing.v1 import AppTest


def test_api_request_reports_connection_errors() -> None:
    success, payload = dashboard.api_request(
        "GET", "http://127.0.0.1:1", "/health", timeout=0.1
    )

    assert success is False
    assert "API request failed:" in payload


def test_api_request_forwards_long_retraining_timeout(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        ok = True

        def json(self):
            return {"status": "promoted"}

    def fake_request(method, url, *, timeout, **kwargs):
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(dashboard.requests, "request", fake_request)
    success, result = dashboard.api_request(
        "POST", "http://nginx", "/retrain", timeout=1800
    )

    assert success is True
    assert result == {"status": "promoted"}
    assert captured["timeout"] == 1800


def test_dashboard_starts_without_exceptions() -> None:
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    ).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "Cassava Leaf Disease Classifier"


def test_dashboard_hides_training_controls_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_TRAINING_WORKFLOW", "false")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    ).run(timeout=15)

    assert not app.exception
    assert "Trigger retraining" not in [button.label for button in app.button]
