"""
test_gesture_logic.py
---------------------
Unit tests for gesture classifier and mouse controller logic.

Run:
    python -m pytest tests/ -v
"""

import sys
import os
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gesture_classifier import GestureClassifier
from src.mouse_controller import MouseController


# ===========================================================================
# GestureClassifier Tests
# ===========================================================================

class TestGestureClassifier:

    def _make_mock_pipeline(self, pred_label: int = 0, proba: float = 0.9):
        """Create a mock sklearn pipeline."""
        pipe = MagicMock()
        pipe.predict_proba.return_value = np.array([[proba] + [0.01] * 6])
        pipe.predict.return_value = np.array([pred_label])
        return pipe

    def _make_mock_encoder(self, classes=None):
        """Create a mock label encoder."""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(classes or ["CLICK", "DRAG", "IDLE", "MOVE",
                                            "PAUSE", "RIGHT_CLICK", "SCROLL"])
        le.inverse_transform = lambda x: [le.classes_[i] for i in x]
        return le

    def test_unloaded_predicts_idle(self):
        clf = GestureClassifier()
        fv = np.random.rand(47).astype(np.float32)
        gesture, conf = clf.predict(fv)
        assert gesture == "IDLE"
        assert conf == 0.0

    def test_predict_with_injected_model(self):
        clf = GestureClassifier(confidence_threshold=0.5)
        clf._model = self._make_mock_pipeline(pred_label=3, proba=0.92)
        clf._label_encoder = self._make_mock_encoder()
        clf._loaded = True

        fv = np.ones(47, dtype=np.float32)
        # Run enough predictions to fill smoothing window
        for _ in range(10):
            gesture, conf = clf.predict(fv)

        assert gesture == "MOVE"
        assert conf > 0.5

    def test_low_confidence_returns_idle(self):
        clf = GestureClassifier(confidence_threshold=0.8)
        clf._model = self._make_mock_pipeline(pred_label=0, proba=0.6)
        clf._label_encoder = self._make_mock_encoder()
        clf._loaded = True

        fv = np.ones(47, dtype=np.float32)
        for _ in range(10):
            gesture, _ = clf.predict(fv)
        assert gesture == "IDLE"

    def test_none_feature_vector_returns_idle(self):
        clf = GestureClassifier()
        clf._loaded = True
        gesture, conf = clf.predict(None)
        assert gesture == "IDLE"
        assert conf == 0.0

    def test_reset_history_clears_state(self):
        clf = GestureClassifier(confidence_threshold=0.5)
        clf._model = self._make_mock_pipeline(pred_label=3, proba=0.9)
        clf._label_encoder = self._make_mock_encoder()
        clf._loaded = True

        fv = np.ones(47, dtype=np.float32)
        for _ in range(6):
            clf.predict(fv)

        clf.reset_history()
        assert clf.last_gesture == "IDLE"
        assert clf.last_confidence == 0.0
        assert len(clf._history) == 0

    def test_smoothing_majority_vote(self):
        """Smoothing window should return the most common prediction."""
        clf = GestureClassifier(confidence_threshold=0.5, smoothing_window=5)
        clf._label_encoder = self._make_mock_encoder()
        clf._loaded = True

        # First inject 4 MOVE predictions manually
        from collections import deque
        clf._history = deque(["MOVE", "MOVE", "MOVE", "MOVE"], maxlen=5)

        # Now inject a mock that returns CLICK once
        clf._model = self._make_mock_pipeline(pred_label=0, proba=0.9)  # CLICK
        fv = np.ones(47, dtype=np.float32)
        gesture, _ = clf.predict(fv)

        # MOVE×4 vs CLICK×1 → MOVE wins
        assert gesture == "MOVE"

    def test_is_loaded_false_before_load(self):
        clf = GestureClassifier()
        assert not clf.is_loaded

    def test_load_nonexistent_file_returns_false(self):
        clf = GestureClassifier(model_path="/nonexistent/path/model.joblib")
        result = clf.load()
        assert result is False
        assert not clf.is_loaded


# ===========================================================================
# MouseController Tests (no actual mouse movement)
# ===========================================================================

class TestMouseController:

    def _make_controller(self):
        """MouseController with a small screen and mocked pyautogui."""
        return MouseController(screen_size=(1920, 1080))

    def _fake_landmarks(self, index_x=0.5, index_y=0.5):
        lms = [(0.5, 0.5, 0.0)] * 21
        lms[8] = (index_x, index_y, 0.0)   # index fingertip
        return lms

    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_move_gesture_calls_moveto(self, mock_move):
        mc = self._make_controller()
        lms = self._fake_landmarks(0.5, 0.5)
        mc.update("MOVE", 0.9, lms, (480, 640))
        assert mock_move.called

    @patch("src.mouse_controller.pyautogui.click")
    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_click_gesture_calls_click(self, mock_move, mock_click):
        mc = self._make_controller()
        lms = self._fake_landmarks(0.5, 0.5)
        mc.update("CLICK", 0.9, lms, (480, 640))
        assert mock_click.called

    @patch("src.mouse_controller.pyautogui.click")
    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_click_debouncing_prevents_rapid_clicks(self, mock_move, mock_click):
        mc = self._make_controller()
        mc.config["click_cooldown"] = 0.5
        lms = self._fake_landmarks()
        # Two rapid clicks
        mc.update("CLICK", 0.9, lms, (480, 640))
        mc.update("CLICK", 0.9, lms, (480, 640))
        # Only one click should fire
        assert mock_click.call_count == 1

    @patch("src.mouse_controller.pyautogui.rightClick")
    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_right_click_gesture(self, mock_move, mock_right):
        mc = self._make_controller()
        lms = self._fake_landmarks()
        mc.update("RIGHT_CLICK", 0.9, lms, (480, 640))
        assert mock_right.called

    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_idle_gesture_does_not_move_much(self, mock_move):
        mc = self._make_controller()
        # With IDLE, landmarks=None
        mock_move.reset_mock()
        mc.update("IDLE", 0.0, None, (480, 640))
        assert not mock_move.called

    @patch("src.mouse_controller.pyautogui.mouseDown")
    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_drag_sets_dragging_flag(self, mock_move, mock_down):
        mc = self._make_controller()
        lms = self._fake_landmarks()
        mc.update("DRAG", 0.9, lms, (480, 640))
        assert mc._dragging is True

    @patch("src.mouse_controller.pyautogui.mouseUp")
    @patch("src.mouse_controller.pyautogui.mouseDown")
    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_drag_release_on_pause(self, mock_move, mock_down, mock_up):
        mc = self._make_controller()
        lms = self._fake_landmarks()
        mc.update("DRAG", 0.9, lms, (480, 640))   # start drag
        mc.update("PAUSE", 0.9, None, (480, 640))  # should release
        assert mc._dragging is False
        assert mock_up.called

    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_stop_disables_controller(self, mock_move):
        mc = self._make_controller()
        mc.stop()
        lms = self._fake_landmarks()
        mock_move.reset_mock()
        mc.update("MOVE", 0.9, lms, (480, 640))
        assert not mock_move.called   # stopped, should not move

    @patch("src.mouse_controller.pyautogui.moveTo")
    def test_resume_re_enables_controller(self, mock_move):
        mc = self._make_controller()
        mc.stop()
        mc.resume()
        lms = self._fake_landmarks()
        mc.update("MOVE", 0.9, lms, (480, 640))
        assert mock_move.called

    def test_sensitivity_clamp(self):
        mc = self._make_controller()
        mc.set_sensitivity(-1.0)
        assert mc.config["sensitivity"] == 0.5
        mc.set_sensitivity(100.0)
        assert mc.config["sensitivity"] == 5.0

    def test_smoothing_clamp(self):
        mc = self._make_controller()
        mc.set_smoothing(0.0)
        assert mc.config["smoothing"] == 0.05
        mc.set_smoothing(2.0)
        assert mc.config["smoothing"] == 1.0
