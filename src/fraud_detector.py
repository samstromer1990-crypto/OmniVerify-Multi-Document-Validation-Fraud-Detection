"""
Fraud detection: train a RandomForest on synthetic data and predict
risk level for a new document.

Risk levels:
  LOW      - looks consistent with the genuine-document distribution
  MEDIUM   - some unusual features; warrants manual review
  HIGH     - multiple features consistent with manipulation
"""

import os
import json
import pickle
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score)

from src.features import extract_features, FEATURE_NAMES
from src.dataset_generator import generate_dataset, label_for_path
from src.preprocessing import load_image
from src.ocr import extract_text
from src.qr_detector import detect_qr_codes


# Class ordering: lower risk -> higher risk
RISK_CLASSES = ["LOW", "MEDIUM", "HIGH"]

# Map dataset folder labels to risk levels
LABEL_TO_RISK = {
    "GENUINE": "LOW",
    "MANIPULATED": "HIGH",
}


def _build_dataset(dataset_dir: str) -> tuple:
    """
    Walk the dataset directory, extract features for every image,
    and produce X (features) and y (labels in RISK_CLASSES space).
    """
    X, y = [], []
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            full = os.path.join(root, fname)
            try:
                img = load_image(full)
                feats = extract_features(img)
                X.append([feats[k] for k in FEATURE_NAMES])
                # Map dataset label to risk level
                raw_label = label_for_path(full)
                risk_label = LABEL_TO_RISK.get(raw_label, "MEDIUM")
                y.append(risk_label)
            except Exception as e:
                print(f"[!] Skipping {full}: {e}")
    return np.array(X), np.array(y)


def _synthesize_medium_examples(X_low: np.ndarray, X_high: np.ndarray,
                                 n: int = 20, random_state: int = 42) -> np.ndarray:
    """
    Create MEDIUM examples by interpolating between LOW and HIGH feature
    vectors. This gives the model a meaningful middle class.
    """
    rng = np.random.default_rng(random_state)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(X_low))
        j = rng.integers(0, len(X_high))
        alpha = rng.uniform(0.3, 0.7)  # mid-range mixing
        mixed = X_low[i] * alpha + X_high[j] * (1 - alpha)
        out.append(mixed)
    return np.array(out)


def train_fraud_model(dataset_dir: str, model_output_path: str,
                      n_genuine: int = 40, n_manipulated: int = 40,
                      n_medium: int = 20, random_state: int = 42) -> dict:
    """
    Generate synthetic data, train a RandomForest, and save it.
    """
    print(f"Generating synthetic dataset in {dataset_dir}...")
    generate_dataset(dataset_dir, n_genuine=n_genuine,
                     n_manipulated=n_manipulated)

    X, y = _build_dataset(dataset_dir)
    print(f"Dataset size: {len(X)} samples (raw)")

    # Split by label so we can synthesize MEDIUM examples properly
    X_low = X[y == "LOW"]
    X_high = X[y == "HIGH"]

    if len(X_low) > 0 and len(X_high) > 0:
        X_medium = _synthesize_medium_examples(
            X_low, X_high, n=n_medium, random_state=random_state
        )
        y_medium = np.array(["MEDIUM"] * len(X_medium))
        X = np.vstack([X, X_medium])
        y = np.concatenate([y, y_medium])
        print(f"Added {len(X_medium)} synthetic MEDIUM examples.")

    print(f"Total dataset size: {len(X)} samples")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Only include classes that actually appear in y_test
    present_labels = sorted(set(y_test) | set(y_pred),
                            key=lambda c: RISK_CLASSES.index(c)
                            if c in RISK_CLASSES else 99)
    cm = confusion_matrix(y_test, y_pred, labels=present_labels).tolist()

    try:
        proba_test = clf.predict_proba(X_test)
        classes = list(clf.classes_)
        if "HIGH" in classes:
            high_idx = classes.index("HIGH")
            high_bin = (y_test == "HIGH").astype(int)
            auc = float(roc_auc_score(high_bin, proba_test[:, high_idx]))
        else:
            auc = None
    except Exception:
        auc = None

    importances = {
        FEATURE_NAMES[i]: float(clf.feature_importances_[i])
        for i in range(len(FEATURE_NAMES))
    }

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump({"model": clf, "feature_names": FEATURE_NAMES}, f)

    summary = {
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": present_labels,
        "roc_auc_high_class": auc,
        "feature_importances": importances,
        "model_path": os.path.abspath(model_output_path),
    }
    return summary, clf


def load_fraud_model(model_path: str):
    """Load a previously-trained fraud model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_names"]


def predict_fraud_risk(image: np.ndarray, model, feature_names: list,
                       ocr_result: dict = None,
                       qr_result: dict = None) -> dict:
    """
    Predict fraud risk for a single image.
    """
    feats = extract_features(image, ocr_result=ocr_result, qr_result=qr_result)
    x = np.array([[feats[k] for k in feature_names]])
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    prob_dict = {classes[i]: float(proba[i]) for i in range(len(classes))}
    return {
        "predicted_risk": pred,
        "class_probabilities": prob_dict,
        "features_used": feats,
    }


def explain_with_shap(model, feature_names: list,
                      background_features: np.ndarray,
                      instance_features: np.ndarray,
                      top_k: int = 5) -> list:
    """
    Use SHAP to explain a single prediction.
    """
    try:
        import shap
    except ImportError:
        return []

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance_features)

    if isinstance(shap_values, list):
        predicted_proba = model.predict_proba(instance_features)[0]
        predicted_class_idx = int(np.argmax(predicted_proba))
        sv = shap_values[predicted_class_idx][0]
    else:
        sv = shap_values[0]

    # Flatten sv in case it's a 2D array from multi-output SHAP
    sv_flat = np.array(sv).flatten()
    # Pair with feature names (cycle if there are more SHAP values than features)
    pairs = []
    for i, val in enumerate(sv_flat):
        fname = feature_names[i % len(feature_names)]
        pairs.append((fname, float(val)))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:top_k]
