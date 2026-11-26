from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.evaluate import (
    evaluate_probabilities,
    save_confusion_matrix_plot,
    save_metrics,
    save_training_history,
)
from src.prediction import predict_image


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict(self, batch, verbose=0):
        assert batch.shape[0] == 1
        return self.probabilities


def test_evaluate_probabilities_returns_required_metrics(tmp_path: Path) -> None:
    metrics = evaluate_probabilities(
        np.array([0, 1, 2, 0]),
        np.array(
            [
                [0.8, 0.1, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0, 0.0],
                [0.1, 0.1, 0.8, 0.0, 0.0],
                [0.6, 0.2, 0.2, 0.0, 0.0],
            ]
        ),
    )

    assert metrics["accuracy"] == 1.0
    assert "macro_f1" in metrics
    assert len(metrics["confusion_matrix"]) == 5
    save_metrics(metrics, tmp_path / "metrics.json")
    assert (tmp_path / "metrics.json").is_file()


def test_predict_image_returns_class_and_confidence(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.png"
    Image.new("RGB", (10, 10), (30, 120, 40)).save(image_path)

    result = predict_image(
        FakeModel([[0.05, 0.1, 0.7, 0.1, 0.05]]), image_path, image_size=(8, 8)
    )

    assert result["class_id"] == 2
    assert result["disease"] == "CGM"
    assert result["confidence"] == pytest.approx(0.7)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_predict_image_rejects_wrong_model_shape(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.png"
    Image.new("RGB", (10, 10), (30, 120, 40)).save(image_path)

    with pytest.raises(ValueError, match="expected"):
        predict_image(FakeModel([[0.5, 0.5]]), image_path, image_size=(8, 8))


def test_evaluation_artifacts_are_saved(tmp_path: Path) -> None:
    save_training_history(
        {"loss": [1.0, 0.5], "val_loss": [1.1, 0.6], "accuracy": [0.4, 0.8]},
        tmp_path,
    )
    save_confusion_matrix_plot(
        [[2, 0, 0, 0, 0]] * 5,
        ("CBB", "CBSD", "CGM", "CMD", "Healthy"),
        tmp_path / "confusion.png",
    )

    assert (tmp_path / "training_history.json").is_file()
    assert (tmp_path / "training_history.png").is_file()
    assert (tmp_path / "confusion.png").is_file()
