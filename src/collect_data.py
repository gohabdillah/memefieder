from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp

from camera_utils import open_camera, probe_available_cameras
from utils import KEYPOINT_DIM, POSE_LABELS, extract_keypoints

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect labeled holistic keypoints from webcam frames."
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=POSE_LABELS,
        help="Pose label to save samples under (saved to data/{label}.csv).",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID (default: 0).",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=8,
        help="How many camera indices to probe when listing (default: 8).",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List detected camera indices and exit.",
    )
    return parser.parse_args()


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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = DATA_DIR / f"{args.label}.csv"

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

    if available_cameras:
        print(f"Detected cameras: {available_cameras}")
    print(f"Active camera ID: {selected_camera_id}")

    mp_holistic, mp_draw = get_holistic_modules()

    collected = 0
    is_collecting = False

    with output_csv.open("a", newline="") as csv_handle:
        writer = csv.writer(csv_handle)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            refine_face_landmarks=True,
        ) as holistic:
            while True:
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

                if is_collecting and has_any_landmarks:
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
                    writer.writerow(keypoints.tolist())
                    collected += 1

                status_text = "COLLECTING" if is_collecting else "READY"
                status_color = (40, 255, 40) if is_collecting else (0, 215, 255)

                cv2.putText(
                    frame,
                    f"Label: {args.label}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (40, 255, 40),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Samples: {collected}",
                    (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (40, 255, 40),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Status: {status_text}",
                    (12, 94),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Feature dim: {KEYPOINT_DIM}",
                    (12, 126),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (240, 240, 240),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Camera ID: {selected_camera_id}",
                    (12, 156),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (240, 240, 240),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press w to start, q to stop",
                    (12, 186),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (240, 240, 240),
                    2,
                )

                cv2.imshow("Memefieder - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("w"):
                    is_collecting = True
                if key == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {collected} samples to {output_csv}")


if __name__ == "__main__":
    main()
