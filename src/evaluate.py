"""Evaluation metrics and serializable evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_count: int = 5,
) -> dict[str, Any]:
    """Calculate classification metrics from a multiclass probability matrix."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != class_count:
        raise ValueError(f"probabilities must have shape (n, {class_count})")
    if len(truth) != len(scores) or len(truth) == 0:
        raise ValueError("y_true and probabilities must have the same non-zero length")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = scores.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("each probability row must have a positive sum")
    scores = scores / row_sums[:, None]
    predicted = scores.argmax(axis=1)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(truth, predicted, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, predicted, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=np.arange(class_count)
        ).tolist(),
    }
    one_hot = label_binarize(truth, classes=np.arange(class_count))
    try:
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(truth, scores, multi_class="ovr", average="macro")
        )
        metrics["pr_auc_macro"] = float(
            average_precision_score(one_hot, scores, average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr"] = None
        metrics["pr_auc_macro"] = None
    return metrics


def evaluate_model(
    model, dataset, labels: np.ndarray, *, class_count: int = 5
) -> dict[str, Any]:
    """Evaluate a Keras model on a dataset using its explicit labels."""
    probabilities = np.asarray(model.predict(dataset, verbose=0))
    return evaluate_probabilities(
        np.asarray(labels)[: len(probabilities)], probabilities, class_count=class_count
    )


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Write metrics as readable JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_training_history(history: dict[str, list[float]], output_dir: Path) -> None:
    """Save training history as JSON and a loss/accuracy plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required to save training plots") from exc

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, axis in (("loss", axes[0]), ("accuracy", axes[1])):
        values = history.get(metric)
        validation_values = history.get(f"val_{metric}")
        if values:
            axis.plot(values, label="train")
        if validation_values:
            axis.plot(validation_values, label="validation")
        axis.set_title(metric.capitalize())
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(figure)


def save_confusion_matrix_plot(
    confusion: list[list[int]], class_names: tuple[str, ...], output_path: Path
) -> None:
    """Save a labelled confusion-matrix plot from evaluated predictions."""
    if len(confusion) != len(class_names) or any(
        len(row) != len(class_names) for row in confusion
    ):
        raise ValueError("confusion matrix dimensions must match class_names")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required to save evaluation plots") from exc

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Held-out test confusion matrix",
    )
    threshold = np.asarray(confusion).max() / 2 if confusion else 0
    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
