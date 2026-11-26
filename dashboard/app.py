"""Streamlit dashboard for cassava predictions and service monitoring."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import CLASS_NAMES


DEFAULT_API_URL = "http://localhost:8000"


def api_request(
    method: str,
    api_url: str,
    path: str,
    *,
    timeout: float = 30,
    **kwargs: Any,
) -> tuple[bool, Any]:
    """Call the API and return an explicit success flag and decoded payload."""
    try:
        response = requests.request(method, f"{api_url.rstrip('/')}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        return False, f"API request failed: {exc}"
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if not response.ok:
        return False, {"status_code": response.status_code, "detail": payload}
    return True, payload


def _show_optional_artifacts(st, results_dir: Path) -> None:
    """Display locally generated artifacts when a later training run provides them."""
    metrics_path = results_dir / "metrics" / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read local metrics: {exc}")
        else:
            st.subheader("Latest evaluation metrics")
            st.json(metrics)
    else:
        st.info("No local evaluation metrics are available yet.")

    plots_dir = results_dir / "plots"
    plots = sorted(plots_dir.glob("*.png")) if plots_dir.is_dir() else []
    if plots:
        st.subheader("Generated plots")
        for plot in plots:
            st.image(str(plot), caption=plot.stem, width="stretch")
    else:
        st.info("No generated plots are available yet.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Cassava Disease Classifier", page_icon="🌿", layout="wide")
    st.title("Cassava Leaf Disease Classifier")
    st.caption("Five-class image classification with a FastAPI backend.")
    api_url = st.sidebar.text_input("API URL", value=os.getenv("API_URL", DEFAULT_API_URL))
    results_dir = Path(st.sidebar.text_input("Results directory", "results"))

    status_ok, status = api_request("GET", api_url, "/model/status")
    health_ok, health = api_request("GET", api_url, "/health")
    st.header("Overview")
    overview_left, overview_right = st.columns(2)
    with overview_left:
        st.metric("API status", "Available" if health_ok else "Unavailable")
    with overview_right:
        st.metric("Model status", "Loaded" if status_ok and status.get("loaded") else "Unavailable")
    if status_ok:
        st.write(f"Model version: `{status.get('version', 'unknown')}`")
    elif isinstance(status, str):
        st.warning(status)

    st.header("Predictions")
    prediction_file = st.file_uploader(
        "Upload one cassava leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if prediction_file is not None and st.button("Predict", type="primary"):
        success, result = api_request(
            "POST",
            api_url,
            "/predict",
            files={"file": (prediction_file.name, prediction_file.getvalue(), prediction_file.type)},
        )
        if success:
            st.image(prediction_file, caption=prediction_file.name, width=350)
            st.success(f"{result['disease']} ({result['confidence']:.2%} confidence)")
            st.json(result["probabilities"])
        else:
            st.error(result)

    st.header("Data upload")
    uploaded_files = st.file_uploader(
        "Upload labelled images for future retraining",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="training_files",
    )
    labels = [
        st.selectbox(f"Label for {file.name}", CLASS_NAMES, key=f"label_{index}")
        for index, file in enumerate(uploaded_files)
    ]
    if uploaded_files and st.button("Submit labelled data"):
        label_ids = [CLASS_NAMES.index(label) for label in labels]
        multipart = [
            ("files", (file.name, file.getvalue(), file.type))
            for file in uploaded_files
        ]
        success, result = api_request(
            "POST",
            api_url,
            "/upload-data",
            data={"labels": json.dumps(label_ids)},
            files=multipart,
        )
        if success:
            st.success(f"Accepted {result['accepted']} image(s).")
        else:
            st.error(result)

    st.header("Model retraining")
    st.info("Upload labelled images, then trigger the promotion-safe retraining workflow.")
    if st.button("Trigger retraining"):
        with st.spinner("Retraining the candidate model. This may take several minutes..."):
            success, result = api_request(
                "POST", api_url, "/retrain", timeout=1800
            )
        if success:
            st.json(result)
        else:
            st.warning(result)

    st.header("Data visualizations and metrics")
    _show_optional_artifacts(st, results_dir)

    st.header("Monitoring")
    metrics_ok, metrics = api_request("GET", api_url, "/metrics")
    if metrics_ok:
        st.json(metrics)
    else:
        st.warning(metrics)


if __name__ == "__main__":
    main()
