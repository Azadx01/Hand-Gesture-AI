"""
camera.py
---------
Handles webcam initialization, frame capture, FPS measurement,
and graceful shutdown. All other modules consume frames from this one.
"""

import cv2
import time
import logging

logger = logging.getLogger(__name__)


class Camera:
    """
    Wraps OpenCV VideoCapture with FPS tracking and error handling.

    Usage:
        cam = Camera(index=0)
        if cam.open():
            ret, frame = cam.read()
            cam.release()
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        """
        Parameters
        ----------
        index  : Camera device index (0 = default webcam).
        width  : Desired frame width in pixels.
        height : Desired frame height in pixels.
        """
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

        # FPS tracking
        self._prev_time: float = 0.0
        self._fps: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """
        Opens the camera.

        Returns True on success, False on failure.
        """
        try:
            # On Windows, DirectShow (CAP_DSHOW) or MSMF avoids default backend device locking issues
            self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.index)

            if not self._cap.isOpened():
                logger.error("Cannot open camera at index %d.", self.index)
                self._cap = None
                return False

            # Apply resolution hints (camera driver may override these)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            logger.info(
                "Camera %d opened at %dx%d.",
                self.index,
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            self._prev_time = time.time()
            return True

        except Exception as exc:
            logger.error("Error opening camera: %s", exc)
            return False

    def read(self) -> tuple[bool, object]:
        """
        Reads the next frame and updates the FPS counter.

        Returns (success: bool, frame: np.ndarray | None).
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame from camera %d.", self.index)
            return False, None

        # Update FPS
        now = time.time()
        elapsed = now - self._prev_time
        self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._prev_time = now

        return True, frame

    def release(self) -> None:
        """Releases the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera %d released.", self.index)

    def is_opened(self) -> bool:
        """Returns True if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        """Most-recently measured frames-per-second."""
        return round(self._fps, 1)

    # ------------------------------------------------------------------
    # Context-manager support  (with Camera() as cam:)
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()


# ---------------------------------------------------------------------------
# Quick standalone test – run:  python src/camera.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("Press 'q' to quit the camera test.")

    cam = Camera(index=0)
    if not cam.open():
        print("ERROR: Could not open the webcam.")
        print("Troubleshooting:")
        print("  1. Check that no other app is using the camera.")
        print("  2. Try a different index: Camera(index=1).")
        print("  3. Ensure OpenCV is installed: pip install opencv-python")
        raise SystemExit(1)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Frame read failed – exiting.")
            break

        # Overlay FPS on the frame
        cv2.putText(
            frame,
            f"FPS: {cam.fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Camera Test – press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Camera test complete.")
