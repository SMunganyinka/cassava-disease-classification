import json
from pathlib import Path

import dashboard.app as dashboard
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _run_dashboard(monkeypatch, tmp_path):
    class FakeResponse:
        ok = True

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def fake_request(method, url, *, timeout, **kwargs):
        if url.endswith("/model/status"):
            return FakeResponse({"loaded": True, "version": "test"})
        if url.endswith("/health"):
            return FakeResponse({"status": "ok"})
        return FakeResponse({"service": {}, "uptime_seconds": 0})

    monkeypatch.setattr(dashboard.requests, "request", fake_request)
    original_directory = Path.cwd()
    try:
        monkeypatch.chdir(tmp_path)
        return AppTest.from_file(APP_PATH).run(timeout=15)
    finally:
        monkeypatch.chdir(original_directory)


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


def test_dashboard_starts_without_exceptions(monkeypatch, tmp_path) -> None:
    app = _run_dashboard(monkeypatch, tmp_path)

    assert not app.exception
    assert app.title[0].value == "Cassava Leaf Disease Classifier"


def test_dashboard_displays_evaluation_metrics_and_recall_chart(
    monkeypatch, tmp_path
) -> None:
    metrics_path = tmp_path / "results" / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "evaluation_split": "held_out_test",
                "test_samples": 10,
                "accuracy": 0.8,
                "balanced_accuracy": 0.7,
                "macro_f1": 0.65,
                "weighted_f1": 0.78,
                "confusion_matrix": [
                    [2, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 2],
                ],
            }
        ),
        encoding="utf-8",
    )
    app = _run_dashboard(monkeypatch, tmp_path)

    assert not app.exception
    metric_labels = [metric.label for metric in app.metric]
    assert "Accuracy" in metric_labels
    assert "Macro F1" in metric_labels
    assert "Confusion matrix" in [item.value for item in app.subheader]
    assert len(app.get("vega_lite_chart")) == 1


def test_dashboard_hides_training_controls_when_disabled(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("ENABLE_TRAINING_WORKFLOW", "false")
    app = _run_dashboard(monkeypatch, tmp_path)

    assert not app.exception
    assert "Trigger retraining" not in [button.label for button in app.button]
