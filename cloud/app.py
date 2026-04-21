from __future__ import annotations

import csv
from collections import Counter
import os
import pickle
from pathlib import Path
import sys
from threading import Lock
import time
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_config import KEYPOINT_DIM, POSE_LABELS, align_keypoints_dim

try:
    from cloud_train import train_cloud_model
except ImportError:
    from .cloud_train import train_cloud_model

RETRAIN_THRESHOLD = 20

DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "models" / "cloud_model.pkl"
PENDING_CORRECTIONS_PATH = Path(__file__).resolve().parent / "pending_corrections.csv"

app = Flask(__name__)

MODEL_LOCK = Lock()
MODEL: Any = None
MODEL_VERSION: float | None = None
METRICS_LOCK = Lock()
INFER_REQUEST_COUNT = 0
INFER_DEVICE_COUNTS: Counter[str] = Counter()
INFER_LABEL_COUNTS: Counter[str] = Counter()
INFER_LATENCY_TOTAL_MS = 0.0
LAST_INFER: dict[str, Any] = {}


def load_model() -> Any:
    global MODEL_VERSION

    if not MODEL_PATH.exists():
        MODEL_VERSION = None
        return None

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    try:
        MODEL_VERSION = float(MODEL_PATH.stat().st_mtime)
    except OSError:
        MODEL_VERSION = time.time()
    return model


def get_model_version() -> float | None:
    return MODEL_VERSION


def get_model_feature_dim() -> int:
    if MODEL is not None and hasattr(MODEL, "n_features_in_"):
        try:
            return int(MODEL.n_features_in_)
        except (TypeError, ValueError):
            pass
    return KEYPOINT_DIM


def parse_keypoints(payload: dict, target_dim: int) -> np.ndarray:
    keypoints = payload.get("keypoints")
    keypoints_array = np.asarray(keypoints, dtype=np.float32).flatten()
    if keypoints_array.size == 0:
        raise ValueError("keypoints must contain float values")
    return align_keypoints_dim(keypoints_array, target_dim)


def append_pending_correction(
    keypoints: np.ndarray,
    correct_label: str,
    device_id: str,
) -> None:
    PENDING_CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not PENDING_CORRECTIONS_PATH.exists()

    with PENDING_CORRECTIONS_PATH.open("a", newline="") as csv_handle:
        writer = csv.writer(csv_handle)
        if is_new_file:
            writer.writerow(
                ["device_id", "correct_label"]
                + [f"k{i}" for i in range(KEYPOINT_DIM)]
            )
        writer.writerow([device_id, correct_label, *keypoints.tolist()])


def get_pending_count() -> int:
    if not PENDING_CORRECTIONS_PATH.exists():
        return 0
    with PENDING_CORRECTIONS_PATH.open("r", newline="") as csv_handle:
        total_rows = sum(1 for _ in csv_handle)
    return max(total_rows - 1, 0)


def flush_pending_corrections_to_data() -> int:
    if not PENDING_CORRECTIONS_PATH.exists():
        return 0

    grouped_rows: dict[str, list[list[float]]] = {}

    with PENDING_CORRECTIONS_PATH.open("r", newline="") as csv_handle:
        reader = csv.DictReader(csv_handle)
        for row in reader:
            label = str(row.get("correct_label", "")).strip()
            if not label:
                continue
            try:
                keypoints = [float(row[f"k{i}"]) for i in range(KEYPOINT_DIM)]
            except (TypeError, ValueError, KeyError):
                continue
            grouped_rows.setdefault(label, []).append(keypoints)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0

    for label, rows in grouped_rows.items():
        csv_path = DATA_DIR / f"{label}.csv"
        with csv_path.open("a", newline="") as output_handle:
            writer = csv.writer(output_handle)
            writer.writerows(rows)
        moved += len(rows)

    PENDING_CORRECTIONS_PATH.unlink(missing_ok=True)
    return moved


@app.get("/health")
def health() -> Any:
    model_ready = MODEL is not None
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model_ready,
            "expected_feature_dim": get_model_feature_dim(),
            "model_version": get_model_version(),
        }
    )


@app.post("/reload-model")
def reload_model_endpoint() -> Any:
    global MODEL

    with MODEL_LOCK:
        MODEL = load_model()
        model_ready = MODEL is not None

    return jsonify(
        {
            "status": "ok" if model_ready else "missing_model",
            "model_loaded": model_ready,
            "expected_feature_dim": get_model_feature_dim(),
            "model_version": get_model_version(),
        }
    )


@app.post("/infer")
def infer() -> Any:
    global INFER_REQUEST_COUNT, INFER_LATENCY_TOTAL_MS, LAST_INFER

    started_at = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    device_id = str(payload.get("device_id", "unknown-device")).strip() or "unknown-device"

    with MODEL_LOCK:
        model_snapshot = MODEL

    if model_snapshot is None:
        return (
            jsonify(
                {
                    "error": "Cloud model unavailable. Run 'python cloud/cloud_train.py' first."
                }
            ),
            503,
        )

    try:
        model_feature_dim = int(getattr(model_snapshot, "n_features_in_", KEYPOINT_DIM))
        keypoints = parse_keypoints(payload, target_dim=model_feature_dim)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    probabilities = model_snapshot.predict_proba(keypoints.reshape(1, -1))[0]
    best_idx = int(np.argmax(probabilities))
    label = str(model_snapshot.classes_[best_idx])
    confidence = float(probabilities[best_idx])
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    with METRICS_LOCK:
        INFER_REQUEST_COUNT += 1
        INFER_DEVICE_COUNTS[device_id] += 1
        INFER_LABEL_COUNTS[label] += 1
        INFER_LATENCY_TOTAL_MS += latency_ms
        LAST_INFER = {
            "device_id": device_id,
            "label": label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
        }

    print(
        "[CLOUD INFER] "
        f"device={device_id} "
        f"label={label} "
        f"confidence={confidence:.2f} "
        f"latency_ms={latency_ms:.1f}",
        flush=True,
    )

    return jsonify({"meme": label, "confidence": round(confidence, 4)})


@app.get("/metrics")
def metrics() -> Any:
    with METRICS_LOCK:
        infer_count = INFER_REQUEST_COUNT
        avg_latency = (
            round(INFER_LATENCY_TOTAL_MS / infer_count, 2) if infer_count > 0 else 0.0
        )
        return jsonify(
            {
                "infer_request_count": infer_count,
                "infer_device_counts": dict(INFER_DEVICE_COUNTS),
                "infer_label_counts": dict(INFER_LABEL_COUNTS),
                "infer_avg_latency_ms": avg_latency,
                "last_infer": LAST_INFER,
            }
        )


@app.post("/correct")
def correct() -> Any:
    global MODEL

    payload = request.get_json(silent=True) or {}

    correct_label = str(payload.get("correct_label", "")).strip()
    if not correct_label:
        return jsonify({"error": "correct_label is required"}), 400
    if correct_label not in POSE_LABELS:
        return (
            jsonify(
                {
                    "error": "correct_label is not supported",
                    "supported_labels": POSE_LABELS,
                }
            ),
            400,
        )

    device_id = str(payload.get("device_id", "unknown-device")).strip() or "unknown-device"

    try:
        keypoints = parse_keypoints(payload, target_dim=KEYPOINT_DIM)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    with MODEL_LOCK:
        append_pending_correction(keypoints, correct_label, device_id)
        pending_count = get_pending_count()

        retrained = False
        moved_to_training = 0
        retrain_error = ""

        if pending_count >= RETRAIN_THRESHOLD:
            moved_to_training = flush_pending_corrections_to_data()
            try:
                train_cloud_model(DATA_DIR, MODEL_PATH)
                MODEL = load_model()
                retrained = True
                pending_count = get_pending_count()
            except Exception as error:  # noqa: BLE001
                retrain_error = str(error)

    response: dict[str, Any] = {
        "status": "accepted",
        "pending_corrections": pending_count,
        "retrained": retrained,
    }

    if moved_to_training:
        response["moved_to_training"] = moved_to_training
    if retrain_error:
        response["retrain_error"] = retrain_error

    return jsonify(response)


if __name__ == "__main__":
    MODEL = load_model()
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    MODEL = load_model()
