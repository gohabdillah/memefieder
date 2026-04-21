from __future__ import annotations

import cv2


def _can_read_frame(cap: cv2.VideoCapture, attempts: int = 5) -> bool:
    """Return True if at least one frame can be read from an opened capture."""
    for _ in range(max(1, attempts)):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
    return False


def probe_available_cameras(max_cameras: int = 8) -> list[int]:
    """Probe camera IDs [0..max_cameras-1] and return usable camera indices."""
    available: list[int] = []
    limit = max(1, int(max_cameras))

    for camera_id in range(limit):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            cap.release()
            continue

        if _can_read_frame(cap):
            available.append(camera_id)
        cap.release()

    return available


def open_camera(camera_id: int) -> cv2.VideoCapture:
    """Open a camera by index or raise a runtime error."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera ID {camera_id}")
    return cap
