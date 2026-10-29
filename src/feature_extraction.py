"""
feature_extraction.py
---------------------
Converts raw MediaPipe hand landmarks into a fixed-length feature vector
suitable for training/inference with scikit-learn classifiers.

Feature engineering decisions
------------------------------
1. Translate all points relative to the wrist (landmark 0) so the hand
   position in the frame does not affect the classification.
2. Scale by the hand span (wrist → middle MCP) so gesture features are
   size-invariant.
3. Include only x and y coordinates (z is noisy with a standard webcam).
4. Optionally append finger-state bits (extended / curled) for richer signal.

Total feature size:  21 landmarks × 2 (x, y) + 5 finger bits = 47 features.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# MediaPipe landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

THUMB_IP = 3

# ------------------------------------------------------------------
# Finger-tip / MCP pairs  (used for "is finger extended?" heuristic)
# ------------------------------------------------------------------
FINGER_PAIRS: list[tuple[str, int, int]] = [
    ("thumb",  THUMB_TIP,  THUMB_IP),   # thumb special-cased
    ("index",  INDEX_TIP,  INDEX_MCP),
    ("middle", MIDDLE_TIP, MIDDLE_MCP),
    ("ring",   RING_TIP,   RING_MCP),
    ("pinky",  PINKY_TIP,  PINKY_MCP),
]

FEATURE_SIZE = 21 * 2 + 5  # 47


def extract_features(
    landmarks: list[tuple[float, float, float]],
) -> Optional[np.ndarray]:
    """
    Extract a normalised 47-dimensional feature vector from 21 landmarks.

    Parameters
    ----------
    landmarks : list of (x, y, z) tuples (normalised to [0, 1] by MediaPipe)

    Returns
    -------
    np.ndarray of shape (47,) or None if landmarks are invalid.
    """
    if landmarks is None or len(landmarks) != 21:
        logger.debug("Invalid landmarks (expected 21, got %s).", len(landmarks) if landmarks else 0)
        return None

    pts = np.array([[lm[0], lm[1]] for lm in landmarks], dtype=np.float32)  # (21, 2)

    # ---- 1. Translate: subtract wrist position ----
    wrist_xy = pts[WRIST].copy()
    pts -= wrist_xy

    # ---- 2. Scale: divide by wrist→middle-MCP distance ----
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        logger.debug("Hand scale too small, skipping frame.")
        return None
    pts /= scale

    # ---- 3. Flatten to 42 values ----
    flat = pts.flatten()  # (42,)

    # ---- 4. Finger-extension bits (5 values) ----
    finger_bits = _compute_finger_bits(landmarks)

    feature_vector = np.concatenate([flat, finger_bits])
    assert feature_vector.shape == (FEATURE_SIZE,), (
        f"Unexpected feature size: {feature_vector.shape}"
    )
    return feature_vector.astype(np.float32)


def _compute_finger_bits(
    landmarks: list[tuple[float, float, float]],
) -> np.ndarray:
    """
    Returns a 5-element float array indicating whether each finger is
    extended (1.0) or curled (0.0).

    Heuristic: a finger is extended when its tip is farther from the wrist
    than its MCP joint.

    For the thumb we compare tip distance vs. IP joint distance from wrist.
    """
    wrist = np.array(landmarks[WRIST][:2])
    bits = []

    for _, tip_idx, base_idx in FINGER_PAIRS:
        tip  = np.array(landmarks[tip_idx][:2])
        base = np.array(landmarks[base_idx][:2])
        tip_dist  = np.linalg.norm(tip  - wrist)
        base_dist = np.linalg.norm(base - wrist)
        bits.append(1.0 if tip_dist > base_dist else 0.0)

    return np.array(bits, dtype=np.float32)


def describe_features() -> dict:
    """
    Returns a dictionary documenting the feature vector layout.
    Useful for debugging and the README.
    """
    return {
        "total_features": FEATURE_SIZE,
        "layout": {
            "0-41":  "Normalised (x, y) of all 21 landmarks (wrist-centred, size-normalised)",
            "42":    "Thumb extended bit",
            "43":    "Index extended bit",
            "44":    "Middle extended bit",
            "45":    "Ring extended bit",
            "46":    "Pinky extended bit",
        },
        "normalisation": {
            "translation": "Wrist position subtracted from all points",
            "scale":       "Divided by wrist→middle-MCP Euclidean distance",
        },
    }


# ---------------------------------------------------------------------------
# Quick standalone test – run:  python src/feature_extraction.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import pprint
    logging.basicConfig(level=logging.DEBUG)

    print("Feature layout:")
    pprint.pprint(describe_features())

    # Simulate 21 landmarks with random values
    rng = np.random.default_rng(42)
    fake_landmarks = [(float(rng.random()), float(rng.random()), float(rng.random())) for _ in range(21)]
    fv = extract_features(fake_landmarks)
    if fv is not None:
        print(f"\nFeature vector shape : {fv.shape}")
        print(f"Feature vector sample: {fv[:10]}  ...")
        print(f"Finger-extension bits: {fv[42:]}")
    else:
        print("Feature extraction returned None (hand scale too small for random test).")
