"""
Milestone 4 test: Aadhaar validation prototype.

Run from the project root:
    python tests/test_milestone_4.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image, check_image_quality  # noqa: E402
from src.ocr import extract_text  # noqa: E402
from src.qr_detector import detect_qr_codes  # noqa: E402
from src.validator import validate_document, print_report  # noqa: E402


IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "samples",
                           "sample_document.jpg")


def main() -> None:
    if not os.path.exists(IMAGE_PATH):
        print(f"[!] No test image at: {IMAGE_PATH}")
        return

    print("=" * 60)
    print("MILESTONE 4: DOCUMENT VALIDATION PROTOTYPE")
    print("=" * 60)

    image = load_image(IMAGE_PATH)
    quality = check_image_quality(image)
    ocr = extract_text(image)
    qr = detect_qr_codes(image)

    report = validate_document(ocr, quality, qr)

    # Pretty print
    print()
    print(print_report(report))

    # Also dump as JSON so it's easy to copy into your report
    json_path = os.path.join(PROJECT_ROOT, "data", "processed",
                             "validation_report.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved to: {json_path}")

    print("\nMILESTONE 4 COMPLETE")


if __name__ == "__main__":
    main()
