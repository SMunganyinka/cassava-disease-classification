"""TensorFlow/Keras model builders used by the training pipeline."""

from __future__ import annotations

from typing import Any

from .preprocessing import CLASS_NAMES, DEFAULT_IMAGE_SIZE, build_augmentation


def _keras():
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for model creation. Install the ML dependencies first."
        ) from exc
    return keras, layers


def build_cnn(
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    filters: tuple[int, ...] = (32, 64, 128, 256),
    dense_units: int = 128,
    dropout: float = 0.0,
    batch_norm: bool = False,
    augment: bool = False,
    seed: int = 42,
):
    """Build the notebook's compact CNN baseline."""
    keras, layers = _keras()
    model_layers: list[Any] = [
        layers.Input(shape=(*image_size, 3)),
    ]
    if augment:
        model_layers.append(build_augmentation(seed))
    for filters_count in filters:
        model_layers.append(
            layers.Conv2D(filters_count, 3, padding="same", use_bias=not batch_norm)
        )
        if batch_norm:
            model_layers.append(layers.BatchNormalization())
        model_layers.extend([layers.Activation("relu"), layers.MaxPooling2D()])
    model_layers.extend(
        [
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation="relu"),
        ]
    )
    if dropout:
        model_layers.append(layers.Dropout(dropout))
    model_layers.append(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    return keras.Sequential(model_layers, name="cassava_cnn")


def build_transfer(
    *,
    backbone: str = "efficientnet",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    dropout: float = 0.3,
    augment: bool = True,
    seed: int = 42,
):
    """Build a frozen ImageNet transfer-learning classifier."""
    keras, layers = _keras()
    if backbone not in {"mobilenetv2", "efficientnet"}:
        raise ValueError("backbone must be 'mobilenetv2' or 'efficientnet'")

    inputs = layers.Input(shape=(*image_size, 3))
    features = inputs
    if augment:
        features = build_augmentation(seed)(features)
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        features = layers.Rescaling(1 / 127.5, offset=-1)(features)
    else:
        base = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        features = layers.Rescaling(255.0)(features)
    base.trainable = False
    features = base(features, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(dropout)(features)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(features)
    return keras.Model(inputs, outputs, name=f"cassava_{backbone}")


def compile_model(model, learning_rate: float = 1e-3):
    """Compile a classifier with the loss used by the notebook."""
    keras, _ = _keras()
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
