from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.preprocessing import (
    difference_hash,
    load_image,
    prepare_image_cache,
    stratified_split,
    validate_image,
)


def _write_image(path: Path, color: tuple[int, int, int], seed: int = 0) -> None:
    if seed:
        pixels = np.random.default_rng(seed).integers(0, 256, (8, 12, 3), dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(path)
        return
    Image.new("RGB", (12, 8), color).save(path)


def test_validate_and_load_image_normalizes_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.png"
    _write_image(image_path, (255, 0, 128))

    assert validate_image(image_path) == (12, 8)
    image = load_image(image_path, (4, 4))

    assert image.shape == (4, 4, 3)
    assert image.dtype == np.float32
    assert np.allclose(image[0, 0], [1.0, 0.0, 128 / 255])


def test_corrupt_image_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Invalid or corrupted"):
        validate_image(image_path)


def test_cache_removes_conflicting_duplicates_and_splits_stratified(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    cache = tmp_path / "cache"
    raw.mkdir()
    rows = []
    for label in range(5):
        for item in range(10):
            image_id = f"{label}_{item}.png"
            _write_image(raw / image_id, (label * 40, item * 30, 10), seed=label * 10 + item)
            rows.append({"image_id": image_id, "label": label})
    manifest = pd.DataFrame(rows)

    cleaned, stats = prepare_image_cache(manifest, raw, cache, image_size=(8, 8))
    splits = stratified_split(cleaned, seed=42)

    assert stats["output_images"] == 50
    assert len(splits.train) + len(splits.validation) + len(splits.test) == 50
    assert set(splits.train["image_path"]).isdisjoint(splits.validation["image_path"])
    assert set(splits.train["image_path"]).isdisjoint(splits.test["image_path"])
    assert all(Path(path).is_file() for path in cleaned["image_path"])
    assert difference_hash(Image.open(raw / "0_0.png"))
