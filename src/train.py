"""Keras training utilities for cached cassava images."""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .data_acquisition import download_dataset, load_manifest
from .evaluate import (
    evaluate_model,
    evaluate_probabilities,
    save_confusion_matrix_plot,
    save_metrics,
    save_training_history,
)
from .model import build_transfer, compile_model
from .preprocessing import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    load_image,
    prepare_image_cache,
    set_seed,
    stratified_split,
)

try:
    from tensorflow.keras.utils import Sequence as KerasSequence
except ImportError:
    class KerasSequence:
        """Fallback base so preprocessing tests do not require TensorFlow."""


class ImageSequence(KerasSequence):
    """Small Keras-compatible sequence that loads cached images on demand."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        batch_size: int = 32,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        shuffle: bool = False,
        seed: int = 42,
        normalize: bool = True,
    ) -> None:
        required = {"image_path", "label"}
        if not required.issubset(manifest.columns):
            raise ValueError("manifest must contain image_path and label columns")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.manifest = manifest.reset_index(drop=True)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.manifest))
        super().__init__()
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.manifest) / self.batch_size))

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        selected = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch = self.manifest.iloc[selected]
        images = np.stack([
            load_image(Path(path), self.image_size) for path in batch["image_path"]
        ])
        if not self.normalize:
            images *= 255.0
        return images, batch["label"].to_numpy(dtype=np.int64)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Return balanced weights for all five classes present in training data."""
    labels = np.asarray(labels, dtype=int)
    present = np.unique(labels)
    if len(present) != len(CLASS_NAMES):
        missing = sorted(set(range(len(CLASS_NAMES))) - set(present))
        raise ValueError(f"Training data is missing classes: {missing}")
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=labels
    )
    return {int(index): float(weight) for index, weight in enumerate(weights)}


def train_model(
    model,
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    use_class_weights: bool = True,
    seed: int = 42,
    output_path: Path | None = None,
) -> tuple[object, dict[str, list[float]], dict[str, object]]:
    """Train a model and return it, history, and validation metrics."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed(seed)
    train_sequence = ImageSequence(train_manifest, batch_size=batch_size, shuffle=True, seed=seed)
    validation_sequence = ImageSequence(validation_manifest, batch_size=batch_size)
    compile_model(model, learning_rate)
    callbacks = []
    try:
        from tensorflow import keras
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ]
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for training") from exc
    started = time.perf_counter()
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=epochs,
        class_weight=class_weights(train_manifest["label"]) if use_class_weights else None,
        callbacks=callbacks,
        verbose=1,
    )
    probabilities = np.asarray(model.predict(validation_sequence, verbose=0))
    metrics = evaluate_probabilities(
        validation_manifest["label"].to_numpy(), probabilities, class_count=len(CLASS_NAMES)
    )
    metrics["fit_seconds"] = round(time.perf_counter() - started, 2)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
    return model, {key: [float(value) for value in values] for key, values in history.history.items()}, metrics


def run_training_pipeline(
    *,
    data_dir: Path = Path("data/cassava"),
    cache_dir: Path = Path("data/cassava_cache/224"),
    model_path: Path = Path("models/cassava_model.keras"),
    metrics_path: Path = Path("results/metrics/metrics.json"),
    artifacts_dir: Path = Path("results"),
    backbone: str = "efficientnet",
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, object]:
    """Acquire data, train a model, and evaluate it on the held-out test set."""
    set_seed(seed)
    data_dir = download_dataset(Path(data_dir))
    raw_manifest = load_manifest(data_dir)
    cached_manifest, preprocessing_stats = prepare_image_cache(
        raw_manifest, data_dir / "train_images", Path(cache_dir)
    )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_manifest.to_csv(cache_dir / "manifest.csv", index=False)
    splits = stratified_split(cached_manifest, seed=seed)

    model = build_transfer(backbone=backbone, seed=seed)
    model, history, validation_metrics = train_model(
        model,
        splits.train,
        splits.validation,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        output_path=model_path,
    )
    test_metrics = evaluate_model(
        model,
        ImageSequence(splits.test, batch_size=batch_size),
        splits.test["label"].to_numpy(),
        class_count=len(CLASS_NAMES),
    )

    artifacts_dir = Path(artifacts_dir)
    plots_dir = artifacts_dir / "plots"
    save_training_history(history, plots_dir)
    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"], CLASS_NAMES, plots_dir / "test_confusion_matrix.png"
    )
    result = {
        "validation": validation_metrics,
        "test": test_metrics,
        "preprocessing": preprocessing_stats,
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "model": {
            "backbone": backbone,
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "seed": seed,
        },
    }
    save_metrics(result, Path(metrics_path))
    return result


def main() -> None:
    """Run the reproducible training pipeline from PowerShell or a shell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cassava"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cassava_cache/224"))
    parser.add_argument("--model-path", type=Path, default=Path("models/cassava_model.keras"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/metrics/metrics.json"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("results"))
    parser.add_argument("--backbone", choices=("efficientnet", "mobilenetv2"), default="efficientnet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_training_pipeline(**vars(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
