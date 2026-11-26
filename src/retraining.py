"""Promotion-safe retraining workflow for labelled uploaded images."""

from __future__ import annotations

import json
import os
from numbers import Integral
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .preprocessing import CLASS_NAMES, stratified_split


@dataclass(frozen=True)
class RetrainingConfig:
    """Files and acceptance criteria for a retraining run."""

    model_path: Path
    history_path: Path
    minimum_samples_per_class: int = 8
    minimum_macro_f1_delta: float = 0.0


def _portable_image_path(value: str | Path) -> Path:
    """Interpret manifest separators correctly on Windows and POSIX systems."""
    return Path(str(value).replace("\\", "/"))


def validate_retraining_manifest(
    manifest: pd.DataFrame, *, minimum_samples_per_class: int | None = 2
) -> None:
    """Validate labels, files, and class coverage before training."""
    if minimum_samples_per_class is not None and minimum_samples_per_class < 1:
        raise ValueError("minimum_samples_per_class must be positive")
    required = {"image_path", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError("Retraining manifest must contain image_path and label columns")
    if manifest.empty:
        raise ValueError("Retraining manifest is empty")
    if any(not isinstance(label, Integral) or isinstance(label, bool) for label in manifest["label"]):
        raise ValueError("Retraining labels must be integer class IDs")
    if not manifest["label"].isin(range(len(CLASS_NAMES))).all():
        raise ValueError("Retraining labels must be integers from 0 through 4")
    missing_files = [
        path for path in manifest["image_path"]
        if not _portable_image_path(path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"Retraining image is missing: {missing_files[0]}")
    counts = manifest["label"].value_counts()
    if minimum_samples_per_class is not None:
        insufficient = [
            class_id for class_id in range(len(CLASS_NAMES))
            if int(counts.get(class_id, 0)) < minimum_samples_per_class
        ]
        if insufficient:
            raise ValueError(
                f"Retraining data needs at least {minimum_samples_per_class} images per class; "
                f"insufficient classes: {insufficient}"
            )


def append_upload_manifest(
    manifest_path: Path, records: list[dict[str, Any]]
) -> pd.DataFrame:
    """Append validated upload records to a CSV manifest."""
    if not records:
        raise ValueError("No upload records were supplied")
    new_records = pd.DataFrame(records)
    if not {"image_path", "label"}.issubset(new_records.columns):
        raise ValueError("Upload records must contain image_path and label")
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        existing = pd.read_csv(manifest_path)
        combined = pd.concat([existing, new_records], ignore_index=True)
    else:
        combined = new_records
    validate_retraining_manifest(
        combined.assign(label=combined["label"].astype("int64")),
        minimum_samples_per_class=None,
    )
    combined.to_csv(manifest_path, index=False)
    return combined


def _record_history(path: Path, entry: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if path.is_file():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Retraining history is invalid JSON: {path}") from exc
    if not isinstance(history, list):
        raise RuntimeError(f"Retraining history must contain a JSON list: {path}")
    history.append(entry)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_retraining(
    manifest: pd.DataFrame,
    *,
    config: RetrainingConfig,
    model_factory: Callable[[], Any],
    trainer: Callable[[Any, pd.DataFrame, pd.DataFrame], tuple[Any, dict[str, Any], dict[str, Any]]],
    current_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a candidate, evaluate it, and atomically promote it if accepted.

    ``trainer`` is injected so the workflow can be tested without TensorFlow;
    production code supplies the Keras training function.
    """
    manifest = manifest.copy()
    manifest["image_path"] = manifest["image_path"].map(
        lambda value: str(_portable_image_path(value))
    )
    validate_retraining_manifest(
        manifest, minimum_samples_per_class=config.minimum_samples_per_class
    )
    splits = stratified_split(manifest)
    candidate = model_factory()
    candidate, training_history, candidate_metrics = trainer(
        candidate, splits.train, splits.validation
    )
    old_score = float((current_metrics or {}).get("macro_f1", float("-inf")))
    new_score = float(candidate_metrics.get("macro_f1", float("-inf")))
    accepted = new_score >= old_score + config.minimum_macro_f1_delta
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "status": "promoted" if accepted else "rejected",
        "accepted": accepted,
        "candidate_metrics": candidate_metrics,
        "previous_metrics": current_metrics or {},
        "training_history": training_history,
        "timestamp": timestamp,
    }
    if accepted:
        model_path = Path(config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_model = model_path.with_suffix(model_path.suffix + ".candidate")
        candidate.save(temporary_model)
        os.replace(temporary_model, model_path)
    _record_history(config.history_path, result)
    return result
