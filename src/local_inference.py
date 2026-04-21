from __future__ import annotations

import argparse
import os
import pickle
import socket
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import requests

from camera_utils import open_camera, probe_available_cameras
from utils import (
    KEYPOINT_DIM,
    align_keypoints_dim,
    expected_model_dim,
    extract_keypoints,
    load_memes,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "local_model.pkl"
MEMES_DIR = ROOT_DIR / "memes"

try:
    CONFIDENCE_THRESHOLD = float(os.getenv("LOCAL_CONFIDENCE_THRESHOLD", "0.75"))
except ValueError:
    CONFIDENCE_THRESHOLD = 0.75

CONFIDENCE_THRESHOLD = max(0.0, min(1.0, CONFIDENCE_THRESHOLD))
CLOUD_URL = os.getenv("CLOUD_URL", "http://localhost:5000").rstrip("/")
DEVICE_ID = os.getenv("DEVICE_ID", socket.gethostname())
NEUTRAL_LABEL = "neutral"

try:
    CLOUD_TIMEOUT_SEC = max(0.1, float(os.getenv("CLOUD_TIMEOUT_SEC", "1.0")))
except ValueError:
    CLOUD_TIMEOUT_SEC = 1.0

try:
    CLOUD_RETRIES = max(1, int(os.getenv("CLOUD_RETRIES", "2")))
except ValueError:
    CLOUD_RETRIES = 2

try:
    CLOUD_MIN_INTERVAL_SEC = max(0.0, float(os.getenv("CLOUD_MIN_INTERVAL_SEC", "0.35")))
except ValueError:
    CLOUD_MIN_INTERVAL_SEC = 0.35

try:
    CLOUD_RESULT_TTL_SEC = max(0.0, float(os.getenv("CLOUD_RESULT_TTL_SEC", "1.5")))
except ValueError:
    CLOUD_RESULT_TTL_SEC = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live local inference with optional cloud fallback."
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=int(os.getenv("CAMERA_ID", "0")),
        help="Preferred camera index (default: 0 or CAMERA_ID env var).",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=8,
        help="How many camera indices to probe when listing/cycling (default: 8).",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List detected camera indices and exit.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=int(os.getenv("FRAME_WIDTH", "640")),
        help="Capture width (default: 640). Lower is faster.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=int(os.getenv("FRAME_HEIGHT", "480")),
        help="Capture height (default: 480). Lower is faster.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=int(os.getenv("HOLISTIC_MODEL_COMPLEXITY", "1")),
        help="MediaPipe Holistic model complexity: 0 (fastest) to 2 (slowest).",
    )
    parser.add_argument(
        "--disable-face-refine",
        action="store_true",
        help="Disable face landmark refinement for better FPS.",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Display FPS in the overlay.",
    )
    return parser.parse_args()


def get_holistic_modules() -> tuple[object, object]:
    """Resolve MediaPipe holistic+drawing modules across package variants."""
    try:
        return mp.solutions.holistic, mp.solutions.drawing_utils
    except AttributeError:
        pass

    try:
        from mediapipe.python.solutions import drawing_utils, holistic

        return holistic, drawing_utils
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            "MediaPipe holistic API is unavailable in this environment. "
            "Use Python 3.10/3.11 and reinstall with 'pip install mediapipe'."
        ) from error


def load_local_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python src/train.py' first."
        )

    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def run_local_prediction(model, keypoints: np.ndarray) -> tuple[str, float]:
    model_feature_dim = expected_model_dim(model)
    aligned_keypoints = align_keypoints_dim(keypoints, model_feature_dim)
    probabilities = model.predict_proba(aligned_keypoints.reshape(1, -1))[0]
    best_idx = int(np.argmax(probabilities))
    label = str(model.classes_[best_idx])
    confidence = float(probabilities[best_idx])
    return label, confidence


def query_cloud(keypoints: np.ndarray) -> tuple[Optional[str], float, float]:
    payload = {
        "keypoints": keypoints.tolist(),
        "device_id": DEVICE_ID,
    }
    for attempt in range(CLOUD_RETRIES):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{CLOUD_URL}/infer",
                json=payload,
                timeout=CLOUD_TIMEOUT_SEC,
            )
            response.raise_for_status()
            data = response.json()
            label = data.get("meme")
            confidence = float(data.get("confidence", 0.0))
            latency_ms = (time.perf_counter() - start) * 1000.0
            if not isinstance(label, str):
                return None, 0.0, latency_ms
            return label, confidence, latency_ms
        except (requests.RequestException, ValueError, TypeError):
            if attempt < CLOUD_RETRIES - 1:
                time.sleep(0.05 * (2**attempt))

    return None, 0.0, 0.0


def overlay_meme(
    frame: np.ndarray,
    meme: np.ndarray,
    margin: int = 12,
    max_width: int = 220,
) -> None:
    if meme is None or meme.size == 0:
        return

    meme_h, meme_w = meme.shape[:2]
    if meme_w == 0 or meme_h == 0:
        return

    scale = min(max_width / meme_w, 1.0)
    target_w = max(1, int(meme_w * scale))
    target_h = max(1, int(meme_h * scale))
    resized = cv2.resize(meme, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    y1, y2 = margin, margin + target_h
    x2 = frame.shape[1] - margin
    x1 = x2 - target_w

    if x1 < 0 or y2 > frame.shape[0]:
        return

    roi = frame[y1:y2, x1:x2]

    if resized.shape[2] == 4:
        alpha = (resized[:, :, 3:4] / 255.0).astype(np.float32)
        overlay_rgb = resized[:, :, :3].astype(np.float32)
        roi_float = roi.astype(np.float32)
        blended = (alpha * overlay_rgb) + ((1.0 - alpha) * roi_float)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = resized[:, :, :3]


def compose_meme_main_view(
    webcam_frame: np.ndarray,
    meme: Optional[np.ndarray],
    predicted_label: Optional[str] = None,
    margin: int = 12,
    webcam_scale: float = 0.28,
) -> np.ndarray:
    """Build a frame where the meme is dominant and webcam is an inset."""
    frame_h, frame_w = webcam_frame.shape[:2]
    canvas = np.full((frame_h, frame_w, 3), 18, dtype=np.uint8)

    if predicted_label == "neutral":
        canvas = webcam_frame.copy()
        cv2.putText(
            canvas,
            "Neutral detected: no meme overlay",
            (12, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    elif meme is not None and meme.size > 0:
        meme_h, meme_w = meme.shape[:2]
        if meme_h > 0 and meme_w > 0:
            scale = min(frame_w / meme_w, frame_h / meme_h)
            target_w = max(1, int(meme_w * scale))
            target_h = max(1, int(meme_h * scale))
            resized = cv2.resize(meme, (target_w, target_h), interpolation=cv2.INTER_AREA)

            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

            y1 = (frame_h - target_h) // 2
            y2 = y1 + target_h
            x1 = (frame_w - target_w) // 2
            x2 = x1 + target_w

            if resized.shape[2] == 4:
                alpha = (resized[:, :, 3:4] / 255.0).astype(np.float32)
                overlay_rgb = resized[:, :, :3].astype(np.float32)
                roi_float = canvas[y1:y2, x1:x2].astype(np.float32)
                blended = (alpha * overlay_rgb) + ((1.0 - alpha) * roi_float)
                canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
            else:
                canvas[y1:y2, x1:x2] = resized[:, :, :3]
    else:
        canvas = cv2.GaussianBlur(webcam_frame, (0, 0), 5)
        cv2.putText(
            canvas,
            "No meme image for current label",
            (12, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    inset_w = max(1, int(frame_w * webcam_scale))
    inset_h = max(1, int(frame_h * webcam_scale))
    inset = cv2.resize(webcam_frame, (inset_w, inset_h), interpolation=cv2.INTER_AREA)

    x2 = frame_w - margin
    x1 = max(0, x2 - inset_w)
    y1 = margin
    y2 = min(frame_h, y1 + inset_h)

    inset_crop_h = max(0, y2 - y1)
    inset_crop_w = max(0, x2 - x1)
    if inset_crop_h > 0 and inset_crop_w > 0:
        canvas[y1:y2, x1:x2] = inset[:inset_crop_h, :inset_crop_w]
        cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
        label_y = y2 + 18 if y2 + 18 < frame_h else max(20, y1 - 8)
        cv2.putText(
            canvas,
            "Webcam",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    return canvas


def main() -> None:
    args = parse_args()

    available_cameras = probe_available_cameras(args.max_cameras)
    if args.list_cameras:
        if available_cameras:
            print("Detected camera IDs:")
            for camera_id in available_cameras:
                print(f"- {camera_id}")
        else:
            print("No working cameras detected in probed range.")
        return

    try:
        model = load_local_model(MODEL_PATH)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    memes = load_memes(MEMES_DIR)
    if not memes:
        print("Warning: no meme images loaded from memes/. Overlay will be skipped.")

    selected_camera_id = args.camera_id
    if available_cameras and selected_camera_id not in available_cameras:
        fallback_id = available_cameras[0]
        print(
            f"Requested camera {selected_camera_id} not detected. "
            f"Using camera {fallback_id} instead."
        )
        selected_camera_id = fallback_id

    try:
        cap = open_camera(selected_camera_id)
    except RuntimeError as error:
        raise SystemExit(f"{error}. Check camera permissions/device ID.") from error

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max(160, args.frame_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max(120, args.frame_height))

    if available_cameras:
        print(f"Detected cameras: {available_cameras}")
    else:
        print("Camera probe found none; using requested camera directly.")
    print(f"Active camera ID: {selected_camera_id}")

    mp_holistic, mp_draw = get_holistic_modules()
    layout_mode = "webcam_main"
    last_label: Optional[str] = None
    last_conf = 0.0
    last_source = "LOCAL"
    cloud_request_count = 0
    cloud_success_count = 0
    last_cloud_latency_ms = 0.0
    last_cloud_label: Optional[str] = None
    last_cloud_conf = 0.0
    last_cloud_response_ts = 0.0
    last_cloud_query_ts = 0.0
    fps = 0.0
    prev_frame_ts = time.perf_counter()

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=args.model_complexity,
        refine_face_landmarks=not args.disable_face_refine,
    ) as holistic:
        while True:
            frame_start_ts = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            has_any_landmarks = any(
                [
                    results.pose_landmarks,
                    results.left_hand_landmarks,
                    results.right_hand_landmarks,
                    results.face_landmarks,
                ]
            )

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                )
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )
            if results.face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                )

            if has_any_landmarks:
                keypoints = extract_keypoints(
                    pose_landmarks=(
                        results.pose_landmarks.landmark
                        if results.pose_landmarks
                        else None
                    ),
                    left_hand_landmarks=(
                        results.left_hand_landmarks.landmark
                        if results.left_hand_landmarks
                        else None
                    ),
                    right_hand_landmarks=(
                        results.right_hand_landmarks.landmark
                        if results.right_hand_landmarks
                        else None
                    ),
                    face_landmarks=(
                        results.face_landmarks.landmark
                        if results.face_landmarks
                        else None
                    ),
                )
                if keypoints.size != KEYPOINT_DIM:
                    continue

                local_label, local_conf = run_local_prediction(model, keypoints)
                final_label = local_label
                final_conf = local_conf
                source = "LOCAL"

                # Neutral is an explicit "no meme" state; avoid cloud overriding it.
                if local_label != NEUTRAL_LABEL and local_conf < CONFIDENCE_THRESHOLD:
                    now_ts = time.monotonic()
                    if now_ts - last_cloud_query_ts >= CLOUD_MIN_INTERVAL_SEC:
                        cloud_request_count += 1
                        cloud_label, cloud_conf, cloud_latency_ms = query_cloud(keypoints)
                        last_cloud_query_ts = now_ts
                        last_cloud_latency_ms = cloud_latency_ms
                        if cloud_label:
                            cloud_success_count += 1
                            last_cloud_label = cloud_label
                            last_cloud_conf = cloud_conf
                            last_cloud_response_ts = now_ts

                    if (
                        last_cloud_label is not None
                        and (now_ts - last_cloud_response_ts) <= CLOUD_RESULT_TTL_SEC
                    ):
                        final_label = last_cloud_label
                        final_conf = last_cloud_conf
                        source = "CLOUD"

                last_label = final_label
                last_conf = final_conf
                last_source = source

            neutral_detected = last_label == NEUTRAL_LABEL
            meme = memes.get(last_label) if last_label else None
            if neutral_detected:
                meme = None

            if layout_mode == "meme_main":
                display_frame = compose_meme_main_view(frame, meme, predicted_label=last_label)
            else:
                display_frame = frame.copy()
                if meme is not None:
                    overlay_meme(display_frame, meme)

            if last_label:
                pose_text = f"Pose: {last_label} ({last_conf:.2f})"
                if not has_any_landmarks:
                    pose_text += " [last]"
                cv2.putText(
                    display_frame,
                    pose_text,
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (40, 255, 40),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Source: {last_source} (threshold={CONFIDENCE_THRESHOLD:.2f})",
                    (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (240, 240, 240),
                    2,
                )
                if neutral_detected:
                    cv2.putText(
                        display_frame,
                        "Neutral detected: no meme",
                        (12, 92),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 215, 255),
                        2,
                    )
            else:
                cv2.putText(
                    display_frame,
                    "No holistic landmarks detected",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (30, 30, 255),
                    2,
                )

            cv2.putText(
                display_frame,
                f"Feature dim: {KEYPOINT_DIM}",
                (12, 122 if neutral_detected else 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
            )

            layout_label = "MEME_MAIN" if layout_mode == "meme_main" else "WEBCAM_MAIN"
            cv2.putText(
                display_frame,
                f"Layout: {layout_label}",
                (12, 152 if neutral_detected else 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
            )

            cv2.putText(
                display_frame,
                f"Camera ID: {selected_camera_id}",
                (12, 182 if neutral_detected else 152),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
            )

            cv2.putText(
                display_frame,
                (
                    f"Cloud req/success: {cloud_request_count}/{cloud_success_count} "
                    f"last={last_cloud_latency_ms:.0f}ms"
                ),
                (12, 212 if neutral_detected else 182),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (240, 240, 240),
                2,
            )

            if args.show_fps:
                delta = max(1e-6, frame_start_ts - prev_frame_ts)
                fps = 1.0 / delta
                prev_frame_ts = frame_start_ts
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (12, 242 if neutral_detected else 212),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (240, 240, 240),
                    2,
                )

            cv2.putText(
                display_frame,
                "Press t layout, c camera, q quit",
                (12, display_frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
            )

            cv2.imshow("Memefieder - Local Inference", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("t"):
                layout_mode = "meme_main" if layout_mode == "webcam_main" else "webcam_main"
            if key == ord("c"):
                refreshed_cameras = probe_available_cameras(args.max_cameras)
                if not refreshed_cameras:
                    print("No cameras available to switch.")
                    continue

                if selected_camera_id in refreshed_cameras:
                    idx = refreshed_cameras.index(selected_camera_id)
                    next_camera_id = refreshed_cameras[(idx + 1) % len(refreshed_cameras)]
                else:
                    next_camera_id = refreshed_cameras[0]

                if next_camera_id == selected_camera_id:
                    print(f"Camera remains on {selected_camera_id} (only one camera detected).")
                    continue

                try:
                    new_cap = open_camera(next_camera_id)
                except RuntimeError:
                    print(f"Failed to switch to camera {next_camera_id}.")
                    continue

                cap.release()
                cap = new_cap
                available_cameras = refreshed_cameras
                selected_camera_id = next_camera_id
                print(f"Switched to camera {selected_camera_id}")
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
