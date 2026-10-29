"""
hand_tracker.py
---------------
Wraps MediaPipe Hands to detect and track one hand in each frame.
Provides the 21 raw landmark coordinates and a helper to draw them.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)

# MediaPipe landmark index constants (for convenience)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class HandTracker:
    """
    Detects and tracks one hand using MediaPipe Hands.

    Attributes
    ----------
    landmarks : list[tuple[float, float, float]] | None
        Normalised (x, y, z) of each of the 21 landmarks, or None when no
        hand is in the frame.  Coordinates are in [0, 1].
    pixel_landmarks : list[tuple[int, int]] | None
        Pixel-space (x, y) of each landmark for the current frame size.
    """

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

        # Results updated on each process() call
        self.landmarks: list[tuple[float, float, float]] | None = None
        self.pixel_landmarks: list[tuple[int, int]] | None = None
        self._results = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> bool:
        """
        Detect hands in *frame* (BGR, uint8).

        Returns True when at least one hand is found.
        Updates self.landmarks and self.pixel_landmarks.
        """
        h, w = frame.shape[:2]

        # MediaPipe requires RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self._results = self._hands.process(rgb)

        if (
            self._results.multi_hand_landmarks
            and len(self._results.multi_hand_landmarks) > 0
        ):
            hand = self._results.multi_hand_landmarks[0]

            # Normalised landmarks (x, y, z all in [0, 1])
            self.landmarks = [
                (lm.x, lm.y, lm.z) for lm in hand.landmark
            ]

            # Pixel landmarks (clamped to frame bounds)
            self.pixel_landmarks = [
                (
                    min(max(int(lm.x * w), 0), w - 1),
                    min(max(int(lm.y * h), 0), h - 1),
                )
                for lm in hand.landmark
            ]
            return True

        # No hand detected – clear previous data
        self.landmarks = None
        self.pixel_landmarks = None
        return False

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the 21 landmarks and their connections on *frame* (in-place).

        Returns the annotated frame for convenience.
        """
        if self._results is None or not self._results.multi_hand_landmarks:
            return frame

        for hand_landmarks in self._results.multi_hand_landmarks:
            # Coloured landmark dots + connections
            self._mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
        return frame

    def get_landmark(self, index: int) -> tuple[float, float, float] | None:
        """Return a single normalised landmark by MediaPipe index."""
        if self.landmarks and 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    def get_pixel_landmark(self, index: int) -> tuple[int, int] | None:
        """Return a single pixel-space landmark by MediaPipe index."""
        if self.pixel_landmarks and 0 <= index < len(self.pixel_landmarks):
            return self.pixel_landmarks[index]
        return None

    def fingertip_pixel(self, finger: str) -> tuple[int, int] | None:
        """
        Convenience: return the pixel tip of a named finger.

        Parameters
        ----------
        finger : 'index' | 'middle' | 'ring' | 'pinky' | 'thumb'
        """
        mapping = {
            "index": INDEX_TIP,
            "middle": MIDDLE_TIP,
            "ring": RING_TIP,
            "pinky": PINKY_TIP,
            "thumb": THUMB_TIP,
        }
        idx = mapping.get(finger.lower())
        if idx is None:
            return None
        return self.get_pixel_landmark(idx)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Quick standalone test – run:  python src/hand_tracker.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from camera import Camera

    logging.basicConfig(level=logging.DEBUG)
    print("Hand Tracker Test  –  press 'q' to quit")

    cam = Camera(index=0)
    if not cam.open():
        print("ERROR: Could not open the webcam.")
        raise SystemExit(1)

    tracker = HandTracker()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        hand_found = tracker.process(frame)
        tracker.draw(frame)

        status = "Hand detected" if hand_found else "No hand"
        color = (0, 255, 0) if hand_found else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(
            frame,
            f"FPS: {cam.fps}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        if hand_found and tracker.pixel_landmarks:
            tip = tracker.fingertip_pixel("index")
            if tip:
                cv2.circle(frame, tip, 10, (255, 0, 0), -1)

        cv2.imshow("Hand Tracker Test – press q", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()
    print("Hand tracker test complete.")
