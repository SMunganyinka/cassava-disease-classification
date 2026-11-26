"""Deterministic image validation, caching, splitting, and augmentation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("CBB", "CBSD", "CGM", "CMD", "Healthy")
VALID_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
RESAMPLE = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class SplitData:
    """Manifest rows and cached image paths for each dataset split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set the Python and NumPy seeds used by preprocessing."""
    random.seed(seed)
    np.random.seed(seed)


def validate_image(path: Path) -> tuple[int, int]:
    """Verify that ``path`` is a supported, decodable image."""
    path = Path(path)
    if path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image format: {path.name}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid or corrupted image: {path}") from exc
    if width < 1 or height < 1:
        raise ValueError(f"Image has invalid dimensions: {path}")
    return width, height


def difference_hash(image: Image.Image, hash_size: int = 16) -> str:
    """Create a perceptual difference hash for duplicate detection."""
    if hash_size < 1:
        raise ValueError("hash_size must be positive")
    grayscale = ImageOps.grayscale(image).resize((hash_size + 1, hash_size))
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return hashlib.sha256(np.packbits(bits).tobytes()).hexdigest()


def _iter_valid_rows(
    manifest: pd.DataFrame, raw_image_dir: Path
) -> Iterator[tuple[int, pd.Series, str]]:
    for index, row in manifest.iterrows():
        path = Path(raw_image_dir) / str(row["image_id"])
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                yield index, row, difference_hash(image)
        except (OSError, ValueError):
            continue


def prepare_image_cache(
    manifest: pd.DataFrame,
    raw_image_dir: Path,
    cache_dir: Path,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate, resize, cache images, and remove unsafe duplicate groups.

    Conflicting labels sharing a perceptual hash are removed entirely. For a
    consistent duplicate group, only the first image is retained. The returned
    manifest contains paths relative to ``cache_dir``.
    """
    raw_image_dir, cache_dir = Path(raw_image_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _, row, image_hash in _iter_valid_rows(manifest, raw_image_dir):
        source = raw_image_dir / str(row["image_id"])
        output = cache_dir / str(row["image_id"])
        try:
            with Image.open(source) as image:
                image.convert("RGB").resize(image_size, RESAMPLE.BILINEAR).save(
                    output, format="JPEG", quality=95
                )
        except (OSError, ValueError):
            continue
        rows.append(
            {"image_id": str(row["image_id"]), "label": int(row["label"]),
             "image_path": str(output), "image_hash": image_hash}
        )

    cleaned = pd.DataFrame(rows)
    if cleaned.empty:
        raise ValueError("No valid images were available for preprocessing")
    conflicting = cleaned.groupby("image_hash")["label"].nunique()
    conflicting_hashes = set(conflicting[conflicting > 1].index)
    conflicting_count = int(cleaned["image_hash"].isin(conflicting_hashes).sum())
    cleaned = cleaned[~cleaned["image_hash"].isin(conflicting_hashes)]
    cleaned = cleaned.drop_duplicates("image_hash", keep="first").reset_index(drop=True)
    stats = {
        "input_images": len(manifest),
        "invalid_images": len(manifest) - len(rows),
        "conflicting_duplicates": conflicting_count,
        "output_images": len(cleaned),
    }
    return cleaned, stats


def stratified_split(
    manifest: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> SplitData:
    """Create deterministic, leakage-safe stratified train/validation/test data."""
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")
    if "label" not in manifest or "image_path" not in manifest:
        raise ValueError("Manifest must contain label and image_path columns")
    indices = np.arange(len(manifest))
    train_indices, remainder = train_test_split(
        indices, test_size=validation_size + test_size,
        stratify=manifest["label"], random_state=seed
    )
    remainder_labels = manifest.iloc[remainder]["label"]
    relative_test_size = test_size / (validation_size + test_size)
    validation_indices, test_indices = train_test_split(
        remainder, test_size=relative_test_size,
        stratify=remainder_labels, random_state=seed
    )
    return SplitData(
        train=manifest.iloc[train_indices].reset_index(drop=True),
        validation=manifest.iloc[validation_indices].reset_index(drop=True),
        test=manifest.iloc[test_indices].reset_index(drop=True),
    )


def load_image(path: Path, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Load an image as a float32 RGB array normalized to ``[0, 1]``."""
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(image_size, RESAMPLE.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def build_augmentation(seed: int = DEFAULT_SEED):
    """Build Keras training augmentation without importing TensorFlow eagerly."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build image augmentation") from exc
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(0.08, seed=seed),
            keras.layers.RandomZoom(0.1, seed=seed),
            keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="cassava_augmentation",
    )
