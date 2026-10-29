"""
train_model.py
--------------
Trains, evaluates, and saves a gesture classifier.

Two classifiers are trained and compared:
  1. Random Forest  (default saved model)
  2. Support Vector Machine

Usage (from project root with .venv activated):
    python src/train_model.py

Outputs
-------
  models/gesture_model.joblib       – best/default model (Random Forest)
  models/gesture_model_svm.joblib   – SVM model
  Printed evaluation report + confusion matrix
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---- Paths -----------------------------------------------------------------
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH  = os.path.join(BASE_DIR, "data", "gestures.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RF_MODEL_PATH  = os.path.join(MODEL_DIR, "gesture_model.joblib")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model_svm.joblib")

MIN_SAMPLES_PER_CLASS = 10  # safety check


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load and validate the CSV dataset.

    Returns
    -------
    X  : feature matrix  (n_samples, n_features)
    y  : integer labels  (n_samples,)
    le : fitted LabelEncoder (stored inside model bundle)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run data_collection.py first to collect training samples."
        )

    df = pd.read_csv(path)

    # ---- Validate columns --------------------------------------------------
    if "label" not in df.columns:
        raise ValueError("CSV has no 'label' column. Is it the right file?")

    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in the dataset.")

    # ---- Drop rows with NaN ------------------------------------------------
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        logger.warning("Dropped %d rows with NaN values.", before - after)

    # ---- Per-class sample count check --------------------------------------
    counts = df["label"].value_counts()
    print("\nSample counts per class:")
    print(counts.to_string())
    sparse_classes = counts[counts < MIN_SAMPLES_PER_CLASS]
    if len(sparse_classes) > 0:
        raise ValueError(
            f"The following classes have fewer than {MIN_SAMPLES_PER_CLASS} samples:\n"
            f"{sparse_classes.to_string()}\n"
            "Collect more data before training."
        )

    X = df[feature_cols].values.astype(np.float32)
    raw_labels = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(raw_labels)

    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(le.classes_)} classes: {list(le.classes_)}")

    return X, y, le


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> tuple[Pipeline, Pipeline]:
    """
    Train Random Forest and SVM classifiers, print evaluation metrics,
    save confusion-matrix plots, and return both trained pipelines.
    """
    # ---- Train / Validation / Test split (70 / 15 / 15) -------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval
        # 0.176 of 0.85 ≈ 0.15 of the full set
    )
    print(f"\nSplit → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---- 1. Random Forest --------------------------------------------------
    print("\n" + "="*60)
    print("  Training Random Forest ...")
    rf_pipeline = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    _report(rf_pipeline, X_val, y_val, X_test, y_test, le, "Random Forest")

    # ---- 2. SVM ------------------------------------------------------------
    print("\n" + "="*60)
    print("  Training SVM (RBF kernel) ...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ))
    ])
    svm_pipeline.fit(X_train, y_train)
    _report(svm_pipeline, X_val, y_val, X_test, y_test, le, "SVM")

    # ---- Cross-validation on full training set (RF only) -------------------
    print("\n" + "="*60)
    print("  5-fold Cross-Validation (Random Forest on train+val) ...")
    cv_scores = cross_val_score(
        rf_pipeline, X_trainval, y_trainval,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ---- Confusion matrix plot (RF, test set) ------------------------------
    _plot_confusion_matrix(rf_pipeline, X_test, y_test, le, "Random_Forest")
    _plot_confusion_matrix(svm_pipeline, X_test, y_test, le, "SVM")

    return rf_pipeline, svm_pipeline


def _report(
    model: Pipeline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    name: str,
) -> None:
    val_acc  = accuracy_score(y_val,  model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n  [{name}]  Val Acc: {val_acc:.4f}   Test Acc: {test_acc:.4f}")
    print("\n  Classification Report (test set):")
    print(
        classification_report(
            y_test,
            model.predict(X_test),
            target_names=le.classes_,
            zero_division=0,
        )
    )


def _plot_confusion_matrix(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    model_name: str,
) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def save_models(
    rf_pipeline: Pipeline,
    svm_pipeline: Pipeline,
    le: LabelEncoder,
) -> None:
    """Persist both model pipelines and the label encoder."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    rf_bundle  = {"model": rf_pipeline,  "label_encoder": le}
    svm_bundle = {"model": svm_pipeline, "label_encoder": le}

    joblib.dump(rf_bundle,  RF_MODEL_PATH)
    joblib.dump(svm_bundle, SVM_MODEL_PATH)

    print(f"\nModels saved:")
    print(f"  Random Forest → {RF_MODEL_PATH}")
    print(f"  SVM           → {SVM_MODEL_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Hand Gesture AI – Model Training")
    print("="*60)

    try:
        X, y, le = load_dataset(DATA_PATH)
        rf_pipeline, svm_pipeline = train_and_evaluate(X, y, le)
        save_models(rf_pipeline, svm_pipeline, le)
        print("\nTraining complete.  You can now start the application.")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nDATA ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during training.")
        sys.exit(1)
