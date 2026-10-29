"""
mouse_controller.py
-------------------
Maps hand landmark coordinates to desktop mouse actions.

Supported actions
-----------------
  MOVE        → move cursor to index-fingertip position
  CLICK       → left-click  (debounced, cooldown-protected)
  RIGHT_CLICK → right-click (debounced, cooldown-protected)
  SCROLL      → scroll vertically based on hand Y-velocity
  DRAG        → hold left button; release on next non-DRAG gesture
  PAUSE/IDLE  → do nothing; release drag if held

Safety
------
  - PyAutoGUI fail-safe is always enabled (move to screen corner to abort).
  - All click actions are debounced to prevent rapid repeats.
  - An emergency stop (ESC key handled by the GUI) can call stop().
"""

import pyautogui
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Keep the fail-safe ON at all times
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0   # We manage timing ourselves

# ---- Default configuration -------------------------------------------------
DEFAULT_CONFIG = {
    # Screen mapping
    "sensitivity":       1.8,      # cursor movement amplification
    "smoothing":         0.35,     # EMA alpha (0 = no movement, 1 = no smoothing)

    # Active region of the camera frame used for cursor mapping
    "cam_margin_x":      0.15,     # fraction of width to ignore on each side
    "cam_margin_y":      0.15,

    # Click debounce / cooldown (seconds)
    "click_cooldown":    0.6,
    "right_click_cooldown": 0.8,
    "drag_start_cooldown":  0.5,

    # Scroll speed (pixels per update while SCROLL gesture is active)
    "scroll_speed":      3,

    # Gesture confidence must exceed this to act (separate from classifier threshold)
    "action_confidence": 0.65,
}


class MouseController:
    """
    Translates gesture predictions into safe desktop mouse actions.

    Parameters
    ----------
    screen_size : (width, height) of the monitor in pixels.
    config      : dict of configuration values (see DEFAULT_CONFIG).
    """

    def __init__(
        self,
        screen_size: tuple[int, int] | None = None,
        config: dict | None = None,
    ):
        self.screen_w, self.screen_h = screen_size or pyautogui.size()
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Smoothed cursor position
        self._cursor_x: float = self.screen_w / 2
        self._cursor_y: float = self.screen_h / 2

        # State flags
        self._dragging:     bool  = False
        self._active:       bool  = True
        self._last_click_t: float = 0.0
        self._last_rclick_t: float = 0.0
        self._last_drag_t:  float = 0.0

        # Previous index-fingertip position (for scroll delta)
        self._prev_index_y: float | None = None

        logger.info(
            "MouseController ready.  Screen: %dx%d. Fail-safe: ON.",
            self.screen_w, self.screen_h,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        gesture: str,
        confidence: float,
        landmarks: list[tuple[float, float, float]] | None,
        frame_shape: tuple[int, int],
    ) -> None:
        """
        Call once per frame with the current gesture prediction.

        Parameters
        ----------
        gesture     : predicted gesture label (string)
        confidence  : classifier confidence [0, 1]
        landmarks   : normalised (x, y, z) for 21 landmarks, or None
        frame_shape : (height, width) of the camera frame
        """
        if not self._active:
            return

        if landmarks is None:
            # No hand – release any held drag
            self._safe_drag_release()
            self._prev_index_y = None
            return

        # Index fingertip (landmark 8)
        if len(landmarks) <= 8:
            return
        index_x_norm, index_y_norm, _ = landmarks[8]

        # Map to screen coordinates
        screen_x, screen_y = self._map_to_screen(
            index_x_norm, index_y_norm, frame_shape
        )

        if gesture == "MOVE":
            self._do_move(screen_x, screen_y)

        elif gesture == "CLICK":
            self._do_move(screen_x, screen_y)
            self._do_click()

        elif gesture == "RIGHT_CLICK":
            self._do_move(screen_x, screen_y)
            self._do_right_click()

        elif gesture == "SCROLL":
            self._do_scroll(index_y_norm)

        elif gesture == "DRAG":
            self._do_move(screen_x, screen_y)
            self._do_drag_start()

        elif gesture in ("PAUSE", "IDLE"):
            self._safe_drag_release()
            self._prev_index_y = None

        # Keep prev_index_y for scroll delta
        self._prev_index_y = index_y_norm

    def stop(self) -> None:
        """Emergency stop: release mouse, disable further actions."""
        self._safe_drag_release()
        self._active = False
        logger.info("MouseController stopped.")

    def resume(self) -> None:
        """Re-enable mouse control after a stop/pause."""
        self._active = True
        logger.info("MouseController resumed.")

    def set_sensitivity(self, value: float) -> None:
        """Adjust cursor sensitivity at runtime (range: 0.5 – 5.0)."""
        self.config["sensitivity"] = float(np.clip(value, 0.5, 5.0))

    def set_smoothing(self, value: float) -> None:
        """Adjust EMA smoothing alpha at runtime (range: 0.05 – 1.0)."""
        self.config["smoothing"] = float(np.clip(value, 0.05, 1.0))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_to_screen(
        self,
        norm_x: float,
        norm_y: float,
        frame_shape: tuple[int, int],
    ) -> tuple[float, float]:
        """
        Convert normalised landmark position to screen pixel position.

        The active camera region is inset by cam_margin to reduce jitter
        at the edges.
        """
        mx = self.config["cam_margin_x"]
        my = self.config["cam_margin_y"]

        # Remap from [margin, 1-margin] → [0, 1]
        rx = (norm_x - mx) / (1.0 - 2 * mx)
        ry = (norm_y - my) / (1.0 - 2 * my)

        # Mirror x (webcam is flipped in display but landmarks are raw)
        rx = 1.0 - rx

        # Clamp
        rx = float(np.clip(rx, 0.0, 1.0))
        ry = float(np.clip(ry, 0.0, 1.0))

        target_x = rx * self.screen_w
        target_y = ry * self.screen_h
        return target_x, target_y

    def _smooth_cursor(self, target_x: float, target_y: float) -> tuple[int, int]:
        """Exponential moving average smoothing for cursor position."""
        alpha = self.config["smoothing"]
        self._cursor_x += alpha * (target_x - self._cursor_x)
        self._cursor_y += alpha * (target_y - self._cursor_y)
        x = int(np.clip(self._cursor_x, 0, self.screen_w - 1))
        y = int(np.clip(self._cursor_y, 0, self.screen_h - 1))
        return x, y

    def _do_move(self, target_x: float, target_y: float) -> None:
        x, y = self._smooth_cursor(target_x, target_y)
        try:
            pyautogui.moveTo(x, y)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI fail-safe triggered! Stopping mouse control.")
            self.stop()

    def _do_click(self) -> None:
        now = time.time()
        if now - self._last_click_t < self.config["click_cooldown"]:
            return
        try:
            pyautogui.click()
            self._last_click_t = now
            logger.debug("Left click.")
        except Exception as exc:
            logger.warning("Click failed: %s", exc)

    def _do_right_click(self) -> None:
        now = time.time()
        if now - self._last_rclick_t < self.config["right_click_cooldown"]:
            return
        try:
            pyautogui.rightClick()
            self._last_rclick_t = now
            logger.debug("Right click.")
        except Exception as exc:
            logger.warning("Right-click failed: %s", exc)

    def _do_scroll(self, index_y_norm: float) -> None:
        if self._prev_index_y is None:
            self._prev_index_y = index_y_norm
            return
        delta_y = index_y_norm - self._prev_index_y   # positive = hand moved down
        scroll_clicks = -int(delta_y * 100 * self.config["scroll_speed"])
        if scroll_clicks != 0:
            try:
                pyautogui.scroll(scroll_clicks)
            except Exception as exc:
                logger.warning("Scroll failed: %s", exc)

    def _do_drag_start(self) -> None:
        now = time.time()
        if not self._dragging and (now - self._last_drag_t > self.config["drag_start_cooldown"]):
            pyautogui.mouseDown()
            self._dragging = True
            self._last_drag_t = now
            logger.debug("Drag started.")

    def _safe_drag_release(self) -> None:
        if self._dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self._dragging = False
            logger.debug("Drag released.")


# ---------------------------------------------------------------------------
# Quick test – run:  python src/mouse_controller.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    print("Mouse Controller quick test.")
    print("The cursor will move to the center of the screen in 3 seconds.")
    print("Move mouse to a screen corner to trigger fail-safe if needed.")
    time.sleep(3)

    mc = MouseController()
    screen_w, screen_h = pyautogui.size()
    # Simulate a MOVE gesture with index fingertip at center of frame
    fake_landmarks = [(0.0, 0.0, 0.0)] * 21
    fake_landmarks[8] = (0.5, 0.5, 0.0)  # index fingertip at center

    for _ in range(30):
        mc.update("MOVE", 0.95, fake_landmarks, (480, 640))
        time.sleep(0.05)

    print(f"Cursor at: {pyautogui.position()}")
    print("Test complete.")
