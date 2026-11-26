from pathlib import Path

import pandas as pd
import pytest

from src.retraining import RetrainingConfig, run_retraining, validate_retraining_manifest


class FakeModel:
    def __init__(self, path: Path | None = None):
        self.path = path

    def save(self, path: Path) -> None:
        Path(path).write_text("candidate", encoding="utf-8")


def manifest(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for label in range(5):
        for item in range(8):
            image = tmp_path / f"{label}-{item}.jpg"
            image.write_bytes(b"image")
            rows.append({"image_path": str(image), "label": label})
    return pd.DataFrame(rows)


def test_retraining_requires_all_classes(tmp_path: Path) -> None:
    data = manifest(tmp_path).iloc[:-8]

    with pytest.raises(ValueError, match="insufficient classes"):
        validate_retraining_manifest(data)


def test_retraining_promotes_candidate_and_records_history(tmp_path: Path) -> None:
    model_path = tmp_path / "model.keras"
    history_path = tmp_path / "history.json"

    def trainer(model, train, validation):
        return model, {"loss": [0.2]}, {"macro_f1": 0.8, "accuracy": 0.9}

    result = run_retraining(
        manifest(tmp_path),
        config=RetrainingConfig(model_path, history_path),
        model_factory=FakeModel,
        trainer=trainer,
        current_metrics={"macro_f1": 0.7},
    )

    assert result["accepted"] is True
    assert result["status"] == "promoted"
    assert model_path.read_text(encoding="utf-8") == "candidate"
    assert "promoted" in history_path.read_text(encoding="utf-8")


def test_retraining_rejects_candidate_without_replacing_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model.keras"
    model_path.write_text("production", encoding="utf-8")

    def trainer(model, train, validation):
        return model, {}, {"macro_f1": 0.6}

    result = run_retraining(
        manifest(tmp_path),
        config=RetrainingConfig(model_path, tmp_path / "history.json"),
        model_factory=FakeModel,
        trainer=trainer,
        current_metrics={"macro_f1": 0.7},
    )

    assert result["accepted"] is False
    assert model_path.read_text(encoding="utf-8") == "production"


def test_retraining_accepts_manifest_paths_with_windows_separators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = manifest(tmp_path)
    rows["image_path"] = [
        str(Path(path).relative_to(tmp_path)).replace("/", "\\")
        for path in rows["image_path"]
    ]
    monkeypatch.chdir(tmp_path)

    def trainer(model, train, validation):
        assert all(Path(path).is_file() for path in train["image_path"])
        assert all(Path(path).is_file() for path in validation["image_path"])
        return model, {}, {"macro_f1": 0.8}

    result = run_retraining(
        rows,
        config=RetrainingConfig(tmp_path / "model.keras", tmp_path / "history.json"),
        model_factory=FakeModel,
        trainer=trainer,
        current_metrics={"macro_f1": 0.7},
    )

    assert result["accepted"] is True
