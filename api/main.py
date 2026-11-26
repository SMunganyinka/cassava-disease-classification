"""FastAPI service for cassava disease model inference and data intake."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.prediction import load_model, predict_image
from src.preprocessing import CLASS_NAMES, validate_image
from src.evaluate import save_metrics
from src.model import build_transfer
from src.retraining import (
    RetrainingConfig,
    append_upload_manifest,
    run_retraining,
)
from src.train import train_model


class RequestMonitor:
    """Thread-safe in-process request counters for basic service monitoring."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0

    def record(self, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += latency_ms
            if error:
                self.error_count += 1

    def snapshot(self) -> dict[str, float | int | None]:
        with self._lock:
            average = (
                self.total_latency_ms / self.request_count
                if self.request_count
                else None
            )
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "average_latency_ms": average,
            }


def _read_metrics(metrics_path: Path | None) -> dict[str, Any]:
    if metrics_path is None or not metrics_path.is_file():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read metrics file: {metrics_path}") from exc


def _configured_retraining_callback(
    *,
    model_path: Path,
    metrics_path: Path | None,
    upload_manifest_path: Path,
) -> Any | None:
    """Build the real TensorFlow retraining callback when paths are configured."""
    base_manifest_value = os.getenv("BASE_MANIFEST_PATH")
    if not base_manifest_value:
        return None
    base_manifest_path = Path(base_manifest_value)
    if not base_manifest_path.is_file():
        return None

    def retrain() -> dict[str, Any]:
        base = pd.read_csv(base_manifest_path)
        uploads = (
            pd.read_csv(upload_manifest_path)
            if upload_manifest_path.is_file()
            else pd.DataFrame(columns=["image_path", "label"])
        )
        combined = pd.concat([base, uploads], ignore_index=True)
        config = RetrainingConfig(
            model_path=model_path,
            history_path=Path(os.getenv("RETRAINING_HISTORY_PATH", "results/retraining_history.json")),
            minimum_samples_per_class=int(os.getenv("MINIMUM_SAMPLES_PER_CLASS", "8")),
            minimum_macro_f1_delta=float(os.getenv("MINIMUM_MACRO_F1_DELTA", "0.0")),
        )
        epochs = int(os.getenv("RETRAINING_EPOCHS", "3"))
        result = run_retraining(
            combined,
            config=config,
            model_factory=lambda: build_transfer(
                backbone=os.getenv("MODEL_BACKBONE", "efficientnet")
            ),
            trainer=lambda candidate, train, validation: train_model(
                candidate,
                train,
                validation,
                epochs=epochs,
                output_path=None,
            ),
            current_metrics=_read_metrics(metrics_path),
        )
        if result["accepted"] and metrics_path is not None:
            save_metrics(result["candidate_metrics"], metrics_path)
        return result

    return retrain


def create_app(
    *,
    model: Any | None = None,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    upload_dir: Path = Path("data/uploaded"),
    upload_manifest_path: Path | None = None,
    retrain_callback: Any | None = None,
) -> FastAPI:
    """Create an app with injectable model and storage paths for testing."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.model is None and application.state.model_path is not None:
            if application.state.model_path.is_file():
                application.state.model = load_model(application.state.model_path)
        yield

    app = FastAPI(
        title="Cassava Leaf Disease API",
        version=os.getenv("MODEL_VERSION", "unversioned"),
        description="Classify cassava leaf images into five disease categories.",
        lifespan=lifespan,
    )
    monitor = RequestMonitor()
    app.state.model = model
    app.state.model_path = Path(model_path) if model_path else None
    app.state.metrics_path = Path(metrics_path) if metrics_path else None
    app.state.upload_dir = Path(upload_dir)
    app.state.upload_manifest_path = (
        Path(upload_manifest_path) if upload_manifest_path else Path(upload_dir) / "manifest.csv"
    )
    app.state.retrain_callback = retrain_callback
    app.state.started_at = time.time()
    app.state.monitor = monitor

    @app.middleware("http")
    async def record_requests(request, call_next):
        started = time.perf_counter()
        error = False
        try:
            response = await call_next(request)
            error = response.status_code >= 400
            return response
        except Exception:
            error = True
            raise
        finally:
            monitor.record((time.perf_counter() - started) * 1000, error)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model/status")
    def model_status() -> dict[str, Any]:
        return {
            "loaded": app.state.model is not None,
            "model_path": str(app.state.model_path) if app.state.model_path else None,
            "version": app.version,
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="An image filename is required")
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Prediction model is unavailable")
        suffix = Path(file.filename).suffix.lower()
        target = app.state.upload_dir / f"prediction-{uuid.uuid4().hex}{suffix}"
        app.state.upload_dir.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("wb") as output:
                shutil.copyfileobj(file.file, output)
            validate_image(target)
            return predict_image(app.state.model, target)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            target.unlink(missing_ok=True)

    @app.post("/upload-data")
    async def upload_data(
        files: list[UploadFile] = File(...),
        labels: str = Form(...),
    ) -> dict[str, Any]:
        try:
            parsed_labels = json.loads(labels)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="labels must be a JSON list") from exc
        if not isinstance(parsed_labels, list) or len(parsed_labels) != len(files):
            raise HTTPException(
                status_code=400, detail="labels must contain one value per uploaded image"
            )
        if any(
            not isinstance(label, int) or isinstance(label, bool)
            or label not in range(len(CLASS_NAMES))
            for label in parsed_labels
        ):
            raise HTTPException(status_code=400, detail="labels must be integers from 0 through 4")

        app.state.upload_dir.mkdir(parents=True, exist_ok=True)
        accepted: list[dict[str, Any]] = []
        created: list[Path] = []
        try:
            for index, file in enumerate(files):
                suffix = Path(file.filename or "").suffix.lower()
                target = app.state.upload_dir / f"labelled-{uuid.uuid4().hex}{suffix}"
                with target.open("wb") as output:
                    shutil.copyfileobj(file.file, output)
                try:
                    validate_image(target)
                except ValueError as exc:
                    target.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                created.append(target)
                accepted.append(
                    {"image_path": str(target), "label": int(parsed_labels[index])}
                )
        except HTTPException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        append_upload_manifest(app.state.upload_manifest_path, accepted)
        return {"accepted": len(accepted), "files": accepted}

    @app.post("/retrain", response_model=None)
    def retrain() -> Any:
        if app.state.retrain_callback is None:
            return JSONResponse(
                status_code=501,
                content={
                    "status": "not_implemented",
                    "detail": "Retraining is not configured for this deployment.",
                },
            )
        try:
            return app.state.retrain_callback()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        result = _read_metrics(app.state.metrics_path)
        result["service"] = monitor.snapshot()
        result["uptime_seconds"] = round(time.time() - app.state.started_at, 2)
        return result

    return app


_model_path = Path(os.getenv("MODEL_PATH", "models/cassava_model.keras"))
_metrics_path = Path(os.getenv("METRICS_PATH", "results/metrics/metrics.json"))
_upload_dir = Path(os.getenv("UPLOAD_DIR", "data/uploaded"))
_upload_manifest_path = _upload_dir / "manifest.csv"
app = create_app(
    model_path=_model_path,
    metrics_path=_metrics_path,
    upload_dir=_upload_dir,
    upload_manifest_path=_upload_manifest_path,
    retrain_callback=_configured_retraining_callback(
        model_path=_model_path,
        metrics_path=_metrics_path,
        upload_manifest_path=_upload_manifest_path,
    ),
)
