"""
gesture_classifier.py
---------------------
Loads a trained model bundle from disk and provides real-time predictions
with confidence thresholding and temporal smoothing.

Model bundle format (created by train_model.py):
    {"model": sklearn.pipeline.Pipeline, "label_encoder": LabelEncoder}
"""

import os
import logging
import numpy as np
import joblib
from collections import deque

logger = logging.getLogger(__name__)

# ---- Default paths ---------------------------------------------------------
BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "gesture_model.joblib")

# ---- Prediction settings ---------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.60   # below this → IDLE
SMOOTHING_WINDOW = 5                  # vote over last N predictions


class GestureClassifier:
    """
    Wraps a trained scikit-learn pipeline to provide:
    - confidence-gated predictions
    - temporal smoothing via majority vote
    - safe fallback to 'IDLE' when uncertain
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        smoothing_window: int = SMOOTHING_WINDOW,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        self._model = None
        self._label_encoder = None
        self._loaded = False

        # Rolling window of recent raw predictions
        self._history: deque[str] = deque(maxlen=smoothing_window)

        # Last smoothed result exposed to callers
        self.last_gesture: str = "IDLE"
        self.last_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load the model bundle from disk.

        Returns True on success, False on failure.
        """
        if not os.path.isfile(self.model_path):
            logger.error(
                "Model file not found: %s\n"
                "Run 'python src/train_model.py' to train the model first.",
                self.model_path,
            )
            return False

        try:
            bundle = joblib.load(self.model_path)
            # Validate bundle structure
            if "model" not in bundle or "label_encoder" not in bundle:
                logger.error(
                    "Invalid model bundle. Expected keys 'model' and 'label_encoder'."
                )
                return False

            self._model = bundle["model"]
            self._label_encoder = bundle["label_encoder"]
            self._loaded = True
            logger.info(
                "Model loaded from %s (classes: %s)",
                self.model_path,
                list(self._label_encoder.classes_),
            )
            return True

        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def classes(self) -> list[str]:
        if self._label_encoder is not None:
            return list(self._label_encoder.classes_)
        return []

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_vector: np.ndarray,
    ) -> tuple[str, float]:
        """
        Predict a gesture from a 47-dimensional feature vector.

        Returns
        -------
        (gesture, confidence)
            gesture    : class name string (or 'IDLE' when uncertain / not loaded)
            confidence : float in [0, 1]
        """
        if not self._loaded:
            return "IDLE", 0.0

        if feature_vector is None or len(feature_vector) == 0:
            return "IDLE", 0.0

        try:
            fv = feature_vector.reshape(1, -1)
            proba = self._model.predict_proba(fv)[0]   # shape: (n_classes,)
            best_idx  = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            raw_label  = self._label_encoder.inverse_transform([best_idx])[0]

            # ---- Confidence gate ----
            if confidence < self.confidence_threshold:
                raw_label = "IDLE"

        except Exception as exc:
            logger.warning("Prediction error: %s", exc)
            return "IDLE", 0.0

        # ---- Temporal smoothing ----
        self._history.append(raw_label)
        smoothed = self._majority_vote()

        self.last_gesture    = smoothed
        self.last_confidence = confidence
        return smoothed, confidence

    def _majority_vote(self) -> str:
        """Return the most common label in the smoothing window."""
        if not self._history:
            return "IDLE"
        from collections import Counter
        return Counter(self._history).most_common(1)[0][0]

    def reset_history(self) -> None:
        """Clear the temporal smoothing buffer."""
        self._history.clear()
        self.last_gesture    = "IDLE"
        self.last_confidence = 0.0

    # ------------------------------------------------------------------
    # Context-manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        pass   # scikit-learn models don't need explicit teardown


# ---------------------------------------------------------------------------
# Quick standalone test – run:  python src/gesture_classifier.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    clf = GestureClassifier()
    if not clf.load():
        print(
            "\nNo trained model found.\n"
            "Run:  python src/train_model.py   first.\n"
            "Then re-run this test."
        )
        sys.exit(1)

    print(f"\nModel loaded.  Classes: {clf.classes}")

    # Feed random feature vectors
    rng = np.random.default_rng(0)
    for i in range(10):
        fv = rng.random(47).astype(np.float32)
        gesture, conf = clf.predict(fv)
        print(f"  Sample {i+1:2d}: {gesture:<12s} (confidence={conf:.3f})")
