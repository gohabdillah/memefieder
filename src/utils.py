from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from feature_config import (
    FACE_LANDMARK_COUNT,
    FACE_LANDMARK_INDICES,
    HAND_LANDMARK_COUNT,
    KEYPOINT_DIM,
    POSE_LANDMARK_COUNT,
    POSE_LABELS,
    POSE_TO_MEME_FILE,
    align_keypoints_dim,
)


def _extract_landmark_block(
    landmarks: Optional[Iterable],
    expected_count: int,
    selected_indices: Optional[list[int]] = None,
) -> np.ndarray:
    """Return a fixed-size xyz block from a landmark collection."""
    block_count = len(selected_indices) if selected_indices is not None else expected_count
    block = np.zeros(block_count * 3, dtype=np.float32)
    if landmarks is None:
        return block

    landmark_list = list(landmarks)

    if selected_indices is not None:
        for out_idx, landmark_idx in enumerate(selected_indices):
            if 0 <= landmark_idx < len(landmark_list):
                landmark = landmark_list[landmark_idx]
                block[out_idx * 3 : out_idx * 3 + 3] = [
                    float(landmark.x),
                    float(landmark.y),
                    float(landmark.z),
                ]
        return block

    usable = min(expected_count, len(landmark_list))
    for idx in range(usable):
        landmark = landmark_list[idx]
        block[idx * 3 : idx * 3 + 3] = [
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
        ]
    return block


def extract_keypoints(
    pose_landmarks: Optional[Iterable] = None,
    left_hand_landmarks: Optional[Iterable] = None,
    right_hand_landmarks: Optional[Iterable] = None,
    face_landmarks: Optional[Iterable] = None,
) -> np.ndarray:
    """Return a fixed-length holistic feature vector."""
    pose_vec = _extract_landmark_block(pose_landmarks, POSE_LANDMARK_COUNT)
    left_hand_vec = _extract_landmark_block(left_hand_landmarks, HAND_LANDMARK_COUNT)
    right_hand_vec = _extract_landmark_block(right_hand_landmarks, HAND_LANDMARK_COUNT)
    face_vec = _extract_landmark_block(
        face_landmarks,
        FACE_LANDMARK_COUNT,
        selected_indices=FACE_LANDMARK_INDICES,
    )
    keypoints = np.concatenate([pose_vec, left_hand_vec, right_hand_vec, face_vec])
    return keypoints.astype(np.float32, copy=False)


def expected_model_dim(model: object, default_dim: int = KEYPOINT_DIM) -> int:
    """Read model feature count while keeping a safe fallback."""
    feature_dim = getattr(model, "n_features_in_", default_dim)
    try:
        return int(feature_dim)
    except (TypeError, ValueError):
        return default_dim


def load_memes(memes_dir: str | Path) -> Dict[str, np.ndarray]:
    """Load meme PNGs indexed by pose label."""
    memes_path = Path(memes_dir)
    memes: Dict[str, np.ndarray] = {}

    for label, file_name in POSE_TO_MEME_FILE.items():
        image_path = memes_path / file_name
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            memes[label] = image

    return memes
