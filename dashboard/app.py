"""Streamlit dashboard for cassava predictions and service monitoring."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
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


def _deployment_setting(st, name: str, default: str) -> str:
    environment_value = os.getenv(name)
    if environment_value is not None:
        return environment_value
    try:
        return str(st.secrets.get(name, default))
    except FileNotFoundError:
        return default


def _show_evaluation_metrics(st, metrics: dict[str, Any]) -> None:
    """Render the held-out evaluation summary and per-class performance."""
    split = metrics.get("evaluation_split")
    test_samples = metrics.get("test_samples")
    if split or test_samples is not None:
        details = []
        if isinstance(split, str):
            details.append(split.replace("_", " "))
        if isinstance(test_samples, int):
            details.append(f"{test_samples:,} test images")
        st.caption(" · ".join(details))

    metric_fields = (
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
    )
    available_metrics = []
    for label, key in metric_fields:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            st.warning(f"Could not display the evaluation metric `{key}` because it is not numeric.")
            continue
        available_metrics.append((label, value))

    if available_metrics:
        columns = st.columns(len(available_metrics))
        for column, (label, value) in zip(columns, available_metrics):
            column.metric(label, f"{value:.1%}")

    raw_matrix = metrics.get("confusion_matrix")
    if raw_matrix is None:
        st.info("The held-out evaluation report does not include a confusion matrix.")
        return
    if (
        not isinstance(raw_matrix, list)
        or len(raw_matrix) != len(CLASS_NAMES)
        or any(not isinstance(row, list) or len(row) != len(CLASS_NAMES) for row in raw_matrix)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for row in raw_matrix
            for value in row
        )
    ):
        st.warning("The saved confusion matrix is not a valid five-class count matrix.")
        return

    matrix = pd.DataFrame(raw_matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    st.subheader("Confusion matrix")
    st.caption(
        "Rows are the actual disease classes and columns are the model predictions. "
        "Per-class recall shows the share of each actual class that the model identifies correctly."
    )
    st.dataframe(
        matrix,
        alt="Held-out test confusion matrix with actual classes as rows and predicted classes as columns",
    )

    supports = [sum(row) for row in raw_matrix]
    recalls = [
        raw_matrix[index][index] / support if support else None
        for index, support in enumerate(supports)
    ]
    recall_data = pd.DataFrame(
        {"Disease class": CLASS_NAMES, "Recall": recalls}
    )
    st.bar_chart(
        recall_data,
        x="Disease class",
        y="Recall",
        y_label="Recall",
        alt="Per-class recall on the held-out test set; lower bars indicate classes the model misses more often",
    )


def _show_optional_artifacts(st, results_dir: Path) -> None:
    """Display committed evaluation data and any locally generated plots."""
    metrics_path = results_dir / "metrics" / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read local metrics: {exc}")
        else:
            if not isinstance(metrics, dict):
                st.error("Could not display local metrics: the evaluation report must be a JSON object.")
            else:
                st.subheader("Held-out evaluation")
                _show_evaluation_metrics(st, metrics)
    else:
        st.info("No held-out evaluation report is available in this deployment.")

    plots_dir = results_dir / "plots"
    plots = sorted(plots_dir.glob("*.png")) if plots_dir.is_dir() else []
    if plots:
        st.subheader("Additional evaluation plots")
        for plot in plots:
            st.image(
                str(plot),
                caption=plot.stem.replace("_", " "),
                alt=f"Evaluation plot: {plot.stem.replace('_', ' ')}",
                width="stretch",
            )
    else:
        st.caption("No additional saved plot images are available.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Cassava Disease Classifier", page_icon="🌿", layout="wide")
    st.title("Cassava Leaf Disease Classifier")
    st.caption("Five-class image classification with a FastAPI backend.")
    api_url = st.sidebar.text_input(
        "API URL",
        value=_deployment_setting(st, "API_URL", DEFAULT_API_URL),
    )
    results_dir = Path(st.sidebar.text_input("Results directory", "results"))
    training_workflow_enabled = (
        _deployment_setting(st, "ENABLE_TRAINING_WORKFLOW", "true")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )

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

    if training_workflow_enabled:
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
    else:
        st.header("Data upload and retraining")
        st.info("Uploads and retraining are disabled in this prediction-only deployment.")

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
