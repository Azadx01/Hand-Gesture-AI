"""
test_features.py
----------------
Unit tests for feature_extraction.py.

Run:
    python -m pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_extraction import (
    extract_features,
    _compute_finger_bits,
    FEATURE_SIZE,
    WRIST,
    INDEX_TIP,
    INDEX_MCP,
    MIDDLE_MCP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmarks(n: int = 21) -> list[tuple[float, float, float]]:
    """Create n random landmarks spread across a realistic range."""
    rng = np.random.default_rng(0)
    return [(float(rng.uniform(0.2, 0.8)),
             float(rng.uniform(0.2, 0.8)),
             float(rng.uniform(-0.1, 0.1))) for _ in range(n)]


def _make_open_hand() -> list[tuple[float, float, float]]:
    """
    Handcrafted open-hand landmarks (all fingertips far from wrist).
    Wrist at (0.5, 0.8). Fingertips near (0.5, 0.1).
    """
    lms = [(0.5, 0.8, 0.0)] * 21     # all at wrist by default
    # Wrist (0), MCP joints
    lms[0]  = (0.5, 0.8, 0.0)        # WRIST
    lms[9]  = (0.5, 0.6, 0.0)        # MIDDLE_MCP (scale reference)
    lms[5]  = (0.4, 0.6, 0.0)        # INDEX_MCP
    lms[13] = (0.5, 0.6, 0.0)        # RING_MCP
    lms[17] = (0.6, 0.6, 0.0)        # PINKY_MCP
    # Fingertips far from wrist → fingers extended
    lms[4]  = (0.3, 0.3, 0.0)        # THUMB_TIP
    lms[8]  = (0.4, 0.2, 0.0)        # INDEX_TIP
    lms[12] = (0.5, 0.1, 0.0)        # MIDDLE_TIP
    lms[16] = (0.6, 0.2, 0.0)        # RING_TIP
    lms[20] = (0.7, 0.3, 0.0)        # PINKY_TIP
    # Intermediate joints (roughly between MCP and tip)
    lms[3]  = (0.35, 0.5, 0.0)       # THUMB_IP
    lms[6]  = (0.42, 0.45, 0.0)
    lms[7]  = (0.41, 0.35, 0.0)
    return lms


def _make_fist() -> list[tuple[float, float, float]]:
    """
    Handcrafted closed-fist landmarks (fingertips close to MCPs).
    """
    lms = [(0.5, 0.8, 0.0)] * 21
    lms[0]  = (0.5, 0.8, 0.0)        # WRIST
    lms[9]  = (0.5, 0.6, 0.0)        # MIDDLE_MCP
    lms[5]  = (0.4, 0.62, 0.0)       # INDEX_MCP
    lms[13] = (0.5, 0.62, 0.0)
    lms[17] = (0.6, 0.62, 0.0)
    # Fingertips curled back close to MCP (NOT farther from wrist)
    lms[8]  = (0.42, 0.65, 0.0)      # INDEX_TIP (closer to wrist than MCP)
    lms[12] = (0.5,  0.65, 0.0)
    lms[16] = (0.58, 0.65, 0.0)
    lms[20] = (0.65, 0.65, 0.0)
    lms[4]  = (0.55, 0.68, 0.0)      # THUMB curled
    lms[3]  = (0.52, 0.67, 0.0)      # THUMB_IP
    return lms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractFeatures:

    def test_returns_correct_shape(self):
        lms = _make_landmarks()
        fv  = extract_features(lms)
        assert fv is not None
        assert fv.shape == (FEATURE_SIZE,)

    def test_dtype_is_float32(self):
        fv = extract_features(_make_landmarks())
        assert fv is not None
        assert fv.dtype == np.float32

    def test_none_on_wrong_count(self):
        """Fewer than 21 landmarks should return None."""
        fv = extract_features([(0.5, 0.5, 0.0)] * 10)
        assert fv is None

    def test_none_on_none_input(self):
        fv = extract_features(None)
        assert fv is None

    def test_translation_invariance(self):
        """Shifting all landmarks by the same offset should produce the same features."""
        lms = _make_open_hand()
        shift = 0.15

        lms_shifted = [(x + shift, y + shift, z) for x, y, z in lms]
        fv1 = extract_features(lms)
        fv2 = extract_features(lms_shifted)

        assert fv1 is not None and fv2 is not None
        np.testing.assert_allclose(fv1, fv2, atol=1e-5)

    def test_wrist_feature_is_zero(self):
        """After translation, the wrist (first 2 values) should be (0, 0)."""
        fv = extract_features(_make_open_hand())
        assert fv is not None
        np.testing.assert_allclose(fv[0:2], [0.0, 0.0], atol=1e-5)

    def test_finger_bits_open_hand(self):
        """All 5 fingers should be extended in the open-hand fixture."""
        fv = extract_features(_make_open_hand())
        assert fv is not None
        finger_bits = fv[42:]
        assert len(finger_bits) == 5
        assert all(b == 1.0 for b in finger_bits), f"Expected all 1s, got {finger_bits}"

    def test_finger_bits_fist(self):
        """All 5 fingers should be curled in the fist fixture."""
        fv = extract_features(_make_fist())
        assert fv is not None
        finger_bits = fv[42:]
        assert len(finger_bits) == 5
        assert all(b == 0.0 for b in finger_bits), f"Expected all 0s, got {finger_bits}"

    def test_feature_size_constant_matches(self):
        """FEATURE_SIZE constant must equal the actual vector length."""
        assert FEATURE_SIZE == 21 * 2 + 5


class TestFingerBits:

    def test_index_extended(self):
        lms = _make_fist()
        # Move index tip far from wrist
        lms[8] = (0.4, 0.2, 0.0)
        bits = _compute_finger_bits(lms)
        # Index is bits[1]
        assert bits[1] == 1.0, f"Index bit should be 1, got {bits}"

    def test_all_curled_returns_zeros(self):
        lms = _make_fist()
        bits = _compute_finger_bits(lms)
        assert sum(bits) == 0.0

    def test_returns_5_elements(self):
        bits = _compute_finger_bits(_make_landmarks())
        assert len(bits) == 5
