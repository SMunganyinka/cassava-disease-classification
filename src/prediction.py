"""Single-image prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import CLASS_NAMES, DEFAULT_IMAGE_SIZE, load_image, validate_image


def _expects_normalized_inputs(model: Any) -> bool:
    """Detect models that contain the pipeline's explicit input rescaling layer."""
    return any(layer.name == "rescaling" for layer in getattr(model, "layers", ()))


def predict_image(
    model,
    image_path: Path,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
    """Predict one validated image and return a stable API-friendly result."""
    validate_image(image_path)
    image = load_image(image_path, image_size)
    if not _expects_normalized_inputs(model):
        image *= 255.0
    batch = np.expand_dims(image, axis=0)
    probabilities = np.asarray(model.predict(batch, verbose=0))
    if probabilities.shape != (1, len(CLASS_NAMES)):
        raise ValueError(
            f"Model returned shape {probabilities.shape}; expected (1, {len(CLASS_NAMES)})"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Model returned invalid probabilities")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predicted_index = int(probabilities[0].argmax())
    return {
        "class_id": predicted_index,
        "disease": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[0, predicted_index]),
        "probabilities": {
            name: float(probabilities[0, index])
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def load_model(model_path: Path):
    """Load a saved Keras model with an explicit TensorFlow dependency error."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load a saved model") from exc
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return keras.models.load_model(model_path)
