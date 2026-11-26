"""Locust workload for image predictions through the Nginx endpoint."""

from __future__ import annotations

import os
from pathlib import Path

from locust import HttpUser, between, task


DEFAULT_IMAGE = Path("data/cassava/train_images/1000015157.jpg")


class CassavaPredictionUser(HttpUser):
    """Send valid image predictions to the configured API host."""

    wait_time = between(1, 2)

    def on_start(self) -> None:
        image_path = Path(os.getenv("CASSAVA_LOCUST_IMAGE", str(DEFAULT_IMAGE)))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Load-test image was not found: {image_path}. "
                "Set CASSAVA_LOCUST_IMAGE to a valid local image."
            )
        self.image_name = image_path.name
        self.image_bytes = image_path.read_bytes()

    @task
    def predict_image(self) -> None:
        with self.client.post(
            "/predict",
            files={
                "file": (
                    self.image_name,
                    self.image_bytes,
                    "image/jpeg",
                )
            },
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Prediction returned HTTP {response.status_code}")
                return
            try:
                payload = response.json()
            except ValueError:
                response.failure("Prediction response was not valid JSON")
                return
            if not {"disease", "confidence", "probabilities"}.issubset(payload):
                response.failure("Prediction response is missing expected fields")
