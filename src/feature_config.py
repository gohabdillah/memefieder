from __future__ import annotations

import numpy as np

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21

# Focus on expression-heavy facial regions (mouth + eyes + brows).
FACE_LANDMARK_INDICES = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    33,
    133,
    362,
    263,
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    276,
]

FACE_LANDMARK_COUNT = len(FACE_LANDMARK_INDICES)
KEYPOINT_DIM = (POSE_LANDMARK_COUNT + (2 * HAND_LANDMARK_COUNT) + FACE_LANDMARK_COUNT) * 3

POSE_LABELS = [
    "girl_look_fire",
    "monkey-pointing",
    "monkey-thinking-monkey",
    "side-eye",
    "think-better",
    "hmm-eye",
    "neutral",
]

POSE_TO_MEME_FILE = {
    "girl_look_fire": "girl_look_fire.png",
    "monkey-pointing": "monkey-pointing.png",
    "monkey-thinking-monkey": "monkey-thinking-monkey.png",
    "side-eye": "side-eye.png",
    "think-better": "think-better.png",
    "hmm-eye": "hmm-eye.png",
}


def align_keypoints_dim(
    keypoints: np.ndarray | list[float],
    target_dim: int = KEYPOINT_DIM,
) -> np.ndarray:
    """Pad or truncate keypoints so model and sample dimensions always match."""
    keypoints_arr = np.asarray(keypoints, dtype=np.float32).flatten()
    if keypoints_arr.size == target_dim:
        return keypoints_arr

    aligned = np.zeros(target_dim, dtype=np.float32)
    usable = min(target_dim, keypoints_arr.size)
    if usable > 0:
        aligned[:usable] = keypoints_arr[:usable]
    return aligned
