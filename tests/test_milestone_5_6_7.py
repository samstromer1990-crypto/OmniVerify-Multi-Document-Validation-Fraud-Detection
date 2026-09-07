"""
Milestone 5 + 6 + 7 test: train fraud model, predict on test image,
and explain with SHAP.

Run from the project root:
    python tests/test_milestone_5_6_7.py
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image  # noqa: E402
from src.ocr import extract_text  # noqa: E402
from src.qr_detector import detect_qr_codes  # noqa: E402
from src.features import extract_features, FEATURE_NAMES  # noqa: E402
from src.fraud_detector import (  # noqa: E402
    train_fraud_model,
    load_fraud_model,
    predict_fraud_risk,
    explain_with_shap,
)


DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_docs")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_model.pkl")
TEST_IMAGE = os.path.join(PROJECT_ROOT, "data", "samples",
                          "sample_document.jpg")


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    print_section("MILESTONES 5, 6, 7: FRAUD DETECTION + EXPLAINABILITY")

    # 1. Train
    print("\n[1/3] Training fraud model on synthetic data...")
    summary, model = train_fraud_model(
        DATASET_DIR, MODEL_PATH,
        n_genuine=40, n_manipulated=40
    )

    print(f"\n  Samples: {summary['n_samples']} "
          f"(train={summary['n_train']}, test={summary['n_test']})")
    print(f"  Confusion Matrix (rows=true, cols=pred, "
          f"order=[LOW,MEDIUM,HIGH]):")
    for row in summary["confusion_matrix"]:
        print(f"    {row}")
    auc = summary.get("roc_auc_high_class")
    if auc is not None:
        print(f"  ROC-AUC (HIGH class): {auc:.3f}")
    print("\n  Top 5 features by importance:")
    sorted_imp = sorted(summary["feature_importances"].items(),
                        key=lambda x: x[1], reverse=True)
    for name, val in sorted_imp[:5]:
        print(f"    {name:<22}: {val:.4f}")

    # 2. Predict on the test image
    print_section("[2/3] Predicting fraud risk on test image")
    if not os.path.exists(TEST_IMAGE):
        print(f"[!] No test image at {TEST_IMAGE}")
        return

    image = load_image(TEST_IMAGE)
    ocr = extract_text(image)
    qr = detect_qr_codes(image)

    result = predict_fraud_risk(image, model, FEATURE_NAMES,
                                ocr_result=ocr, qr_result=qr)

    print(f"  Predicted risk     : {result['predicted_risk']}")
    print(f"  Class probabilities:")
    for cls, p in result["class_probabilities"].items():
        print(f"    {cls:<8}: {p:.3f}")

    # 3. SHAP explanation
    print_section("[3/3] SHAP explanation (why?)")
    feats = extract_features(image, ocr_result=ocr, qr_result=qr)
    x = np.array([[feats[k] for k in FEATURE_NAMES]])

    # Build a small background set from the synthetic data
    background = []
    for cls_dir in ("genuine", "manipulated"):
        cls_path = os.path.join(DATASET_DIR, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path))[:5]:
            img = load_image(os.path.join(cls_path, fname))
            f = extract_features(img)
            background.append([f[k] for k in FEATURE_NAMES])
    if not background:
        background = x.tolist()  # fallback
    bg = np.array(background)

    explanations = explain_with_shap(model, FEATURE_NAMES, bg, x, top_k=5)
    if explanations:
        print("  Top features pushing this prediction:")
        for fname, sval in explanations:
            direction = "up risk" if sval > 0 else "down risk"
            print(f"    {fname:<22}: {sval:+.4f}  ({direction})")
    else:
        print("  SHAP not installed. Run: python -m pip install shap")

    print_section("MILESTONES 5, 6, 7 COMPLETE")
    print("Fraud-risk model trained, evaluated, and used to predict "
          "with explanation.")


if __name__ == "__main__":
    main()
