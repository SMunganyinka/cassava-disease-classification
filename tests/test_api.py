import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app


class FakeModel:
    def predict(self, batch, verbose=0):
        return [[0.05, 0.1, 0.7, 0.1, 0.05]]


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), (40, 120, 50)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_model_status(tmp_path: Path) -> None:
    client = TestClient(create_app(upload_dir=tmp_path))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model/status").json()["loaded"] is False


def test_predict_valid_image(tmp_path: Path) -> None:
    client = TestClient(create_app(model=FakeModel(), upload_dir=tmp_path))

    response = client.post(
        "/predict",
        files={"file": ("leaf.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["disease"] == "CGM"
    assert not list(tmp_path.iterdir())


def test_predict_without_model_returns_service_unavailable(tmp_path: Path) -> None:
    client = TestClient(create_app(upload_dir=tmp_path))

    response = client.post(
        "/predict",
        files={"file": ("leaf.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 503


def test_upload_data_requires_labels_and_validates_images(tmp_path: Path) -> None:
    client = TestClient(create_app(upload_dir=tmp_path))

    response = client.post(
        "/upload-data",
        data={"labels": json.dumps([2, 4])},
        files=[
            ("files", ("one.png", image_bytes(), "image/png")),
            ("files", ("two.png", image_bytes(), "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.json()["accepted"] == 2
    assert len(list(tmp_path.glob("labelled-*"))) == 2
    assert (tmp_path / "manifest.csv").is_file()


def test_training_workflow_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_TRAINING_WORKFLOW", "false")
    client = TestClient(create_app(upload_dir=tmp_path))

    upload_response = client.post(
        "/upload-data",
        data={"labels": json.dumps([2])},
        files=[("files", ("leaf.png", image_bytes(), "image/png"))],
    )
    retrain_response = client.post("/retrain")

    assert upload_response.status_code == 501
    assert retrain_response.status_code == 501
    assert not list(tmp_path.iterdir())


def test_docker_configuration_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "Dockerfile").is_file()
    assert (root / "docker-compose.yml").is_file()
    assert (root / "nginx" / "nginx.conf").is_file()
