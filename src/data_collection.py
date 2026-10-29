"""
data_collection.py
------------------
Interactive script for collecting labelled hand-gesture samples.

Usage (from the project root with .venv activated):
    python src/data_collection.py --gesture MOVE --samples 200
    python src/data_collection.py --gesture CLICK --samples 200
    ... (repeat for every gesture class)

Each run appends rows to  data/gestures.csv.
"""

import cv2
import csv
import argparse
import logging
import os
import sys
import time

# Make sure the src/ package is importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.camera import Camera
from src.hand_tracker import HandTracker
from src.feature_extraction import extract_features, FEATURE_SIZE

logger = logging.getLogger(__name__)

# ---- Gesture classes -------------------------------------------------------
GESTURE_CLASSES = [
    "MOVE",
    "CLICK",
    "RIGHT_CLICK",
    "SCROLL",
    "PAUSE",
    "DRAG",
    "IDLE",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gestures.csv")

# ---- CSV column header -----------------------------------------------------
def _build_header() -> list[str]:
    cols = [f"f{i}" for i in range(FEATURE_SIZE)]
    cols.append("label")
    return cols


def _ensure_csv(path: str) -> None:
    """Create the CSV with a header row if it does not exist yet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_build_header())
        logger.info("Created %s", path)


def collect(gesture: str, n_samples: int, camera_index: int = 0) -> None:
    """
    Opens the webcam and collects *n_samples* feature vectors labelled
    with *gesture*, saving them to the shared CSV file.
    """
    if gesture not in GESTURE_CLASSES:
        print(
            f"ERROR: '{gesture}' is not a recognised gesture.\n"
            f"Valid gestures: {', '.join(GESTURE_CLASSES)}"
        )
        return

    _ensure_csv(DATA_PATH)

    cam = Camera(index=camera_index)
    if not cam.open():
        print("ERROR: Could not open the webcam.")
        return

    tracker = HandTracker()
    collected = 0

    print(f"\n{'='*60}")
    print(f"  Collecting gesture: {gesture}")
    print(f"  Target samples   : {n_samples}")
    print(f"  Output file      : {DATA_PATH}")
    print(f"{'='*60}")
    print("\nPose your hand for this gesture and press SPACE to start.")
    print("Press 'q' to quit early.\n")

    collecting = False
    last_save_time = 0.0
    SAVE_INTERVAL = 0.05  # ~20 samples/sec max

    with open(DATA_PATH, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        while collected < n_samples:
            ret, frame = cam.read()
            if not ret:
                print("Warning: frame read failed, retrying...")
                time.sleep(0.05)
                continue

            # Flip so the display mirrors the user
            frame = cv2.flip(frame, 1)

            hand_found = tracker.process(frame)
            tracker.draw(frame)

            if collecting and hand_found and tracker.landmarks:
                now = time.time()
                if now - last_save_time >= SAVE_INTERVAL:
                    fv = extract_features(tracker.landmarks)
                    if fv is not None:
                        row = list(fv) + [gesture]
                        writer.writerow(row)
                        collected += 1
                        last_save_time = now

            # ---- On-screen overlay ----
            h, w = frame.shape[:2]
            status_color = (0, 200, 0) if collecting else (0, 120, 255)
            status_text  = "COLLECTING" if collecting else "READY – press SPACE"

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, f"Gesture: {gesture}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {collected} / {n_samples}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (w - 320, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"FPS: {cam.fps}", (w - 120, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if not hand_found:
                cv2.putText(frame, "No hand detected", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(f"Data Collection – {gesture}", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                collecting = not collecting
                print("  Recording..." if collecting else "  Paused.")
            elif key == ord("q"):
                print(f"\nQuit early. Collected {collected} samples.")
                break

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Saved {collected} samples for '{gesture}' → {DATA_PATH}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect hand-gesture training samples."
    )
    parser.add_argument(
        "--gesture",
        choices=GESTURE_CLASSES,
        required=True,
        help="Gesture label to collect.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to collect (default: 200).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    collect(
        gesture=args.gesture,
        n_samples=args.samples,
        camera_index=args.camera,
    )
