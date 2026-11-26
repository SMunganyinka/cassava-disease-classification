"""Cassava dataset acquisition and manifest validation."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


COMPETITION = "cassava-leaf-disease-classification"
REQUIRED_COLUMNS = ("image_id", "label")
ALLOWED_LABELS = frozenset(range(5))


def _safe_zip_members(members: Iterable[str]) -> list[str]:
    """Return dataset members that cannot escape the extraction directory."""
    safe: list[str] = []
    for member in members:
        path = Path(member)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in dataset archive: {member!r}")
        if member == "train.csv" or member == "label_num_to_disease_map.json":
            safe.append(member)
        elif path.parts and path.parts[0] == "train_images":
            safe.append(member)
    return safe


def download_dataset(
    data_dir: Path,
    *,
    competition: str = COMPETITION,
    kaggle_executable: str = "kaggle",
) -> Path:
    """Download and extract the Kaggle competition data into ``data_dir``.

    The Kaggle CLI must already be authenticated. Existing ``train.csv`` and
    ``train_images`` are reused, making repeated runs idempotent.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "train.csv"
    image_dir = data_dir / "train_images"
    if csv_path.is_file() and image_dir.is_dir():
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{competition}.zip"
    if not archive.is_file():
        try:
            result = subprocess.run(
                [
                    kaggle_executable,
                    "competitions",
                    "download",
                    "-c",
                    competition,
                    "-p",
                    str(data_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Kaggle CLI was not found: {kaggle_executable}. "
                "Install the project dependencies or provide data/cassava/train.csv "
                "and data/cassava/train_images first."
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            raise RuntimeError(f"Kaggle download failed for {competition}: {detail}")

    if not archive.is_file():
        raise FileNotFoundError(f"Expected downloaded archive was not found: {archive}")
    with zipfile.ZipFile(archive) as dataset_zip:
        members = _safe_zip_members(dataset_zip.namelist())
        dataset_zip.extractall(data_dir, members=members)
    return data_dir


def load_manifest(data_dir: Path) -> pd.DataFrame:
    """Load and validate the labelled image manifest."""
    manifest_path = Path(data_dir) / "train.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    if frame.empty:
        raise ValueError("Dataset manifest is empty")
    if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
        raise ValueError("Manifest image_id values must be present and unique")
    unsafe_ids = [
        image_id for image_id in frame["image_id"]
        if Path(str(image_id)).is_absolute() or Path(str(image_id)).name != str(image_id)
    ]
    if unsafe_ids:
        raise ValueError(f"Manifest contains unsafe image_id paths: {unsafe_ids[:3]}")
    if not frame["label"].isin(ALLOWED_LABELS).all():
        raise ValueError("Manifest labels must be integer values from 0 through 4")
    frame["label"] = frame["label"].astype("int64")

    image_dir = Path(data_dir) / "train_images"
    missing_images = [
        image_id for image_id in frame["image_id"]
        if not (image_dir / str(image_id)).is_file()
    ]
    if missing_images:
        preview = ", ".join(map(str, missing_images[:3]))
        raise FileNotFoundError(
            f"{len(missing_images)} manifest images are missing from {image_dir}: {preview}"
        )
    return frame


def load_label_map(data_dir: Path) -> dict[int, str]:
    """Load the optional Kaggle label map, returning an empty map if absent."""
    map_path = Path(data_dir) / "label_num_to_disease_map.json"
    if not map_path.is_file():
        return {}
    with map_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): str(value) for key, value in raw.items()}
