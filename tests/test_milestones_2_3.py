"""
Milestone 2 + 3 test: run OCR and QR detection on a document image.

Run from the project root:
    python tests/test_milestones_2_3.py
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image  # noqa: E402
from src.ocr import extract_text, visualize_ocr, save_ocr_result  # noqa: E402
from src.qr_detector import detect_qr_codes, visualize_qr  # noqa: E402


IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "samples", "sample_document.jpg")

OCR_VIZ_PATH = os.path.join(PROJECT_ROOT, "reports",
                            "milestone_2_ocr_visualization.png")
OCR_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "processed",
                             "ocr_result.json")
QR_VIZ_PATH = os.path.join(PROJECT_ROOT, "reports",
                           "milestone_3_qr_visualization.png")
COMBINED_VIZ_PATH = os.path.join(PROJECT_ROOT, "reports",
                                 "milestone_2_3_combined.png")


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    print_section("MILESTONES 2 & 3: OCR + QR DETECTION")

    if not os.path.exists(IMAGE_PATH):
        print(f"\n[!] No test image at: {IMAGE_PATH}")
        print("    Run: python tests/generate_test_image.py")
        return

    image = load_image(IMAGE_PATH)
    print(f"[OK] Loaded image: {IMAGE_PATH}")

    # ----- OCR -----
    print_section("Running OCR")
    ocr_result = extract_text(image)

    print(f"  Detected words : {ocr_result['word_count']}")
    print(f"  Mean confidence: {ocr_result['mean_confidence']}")
    print(f"  Text preview   : {ocr_result['text'][:200]}...")

    json_path = save_ocr_result(ocr_result, OCR_JSON_PATH)
    print(f"  JSON saved to  : {json_path}")

    ocr_viz_path = visualize_ocr(image, ocr_result, OCR_VIZ_PATH)
    print(f"  OCR viz saved  : {ocr_viz_path}")

    # ----- QR DETECTION -----
    print_section("Running QR Detection")
    qr_result = detect_qr_codes(image)

    print(f"  QR count       : {qr_result['count']}")
    for idx, qr in enumerate(qr_result["qrcodes"]):
        print(f"  QR #{idx + 1}:")
        print(f"    BBox         : {qr['bbox']}")
        print(f"    Readable     : {qr['readable']}")
        print(f"    Decoded len  : {qr['decoded_length']}")
        if qr["decoded_text"]:
            print(f"    Decoded text : {qr['decoded_text'][:100]}")

    qr_viz_path = visualize_qr(image, qr_result, QR_VIZ_PATH)
    print(f"  QR viz saved   : {qr_viz_path}")

    # ----- COMBINED VISUALIZATION -----
    print_section("Building combined visualization")
    combined = image.copy()

    for word in ocr_result["words"]:
        b = word["bbox"]
        cv2.rectangle(combined, (b["x"], b["y"]),
                      (b["x"] + b["w"], b["y"] + b["h"]),
                      (0, 255, 0), 2)

    for idx, qr in enumerate(qr_result["qrcodes"]):
        polygon = qr["polygon"]
        pts = [(p[0], p[1]) for p in polygon]
        for i in range(len(pts)):
            cv2.line(combined, pts[i], pts[(i + 1) % len(pts)], (255, 0, 0), 3)
        b = qr["bbox"]
        cv2.putText(combined, f"QR {idx + 1}", (b["x"], b["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imwrite(COMBINED_VIZ_PATH, combined)
    print(f"  Combined viz   : {COMBINED_VIZ_PATH}")

    print_section("Displaying results")
    print("A matplotlib window will open. Close it to finish.")

    if len(combined.shape) == 3:
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    else:
        combined_rgb = combined

    plt.figure(figsize=(14, 8))
    plt.imshow(combined_rgb)
    plt.title("OCR (green) + QR (blue) Detection")
    plt.axis("on")
    plt.show()

    print_section("MILESTONES 2 & 3 COMPLETE")


if __name__ == "__main__":
    main()
