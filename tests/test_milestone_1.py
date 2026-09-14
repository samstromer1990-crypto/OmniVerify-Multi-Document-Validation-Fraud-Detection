"""
Milestone 1 test: load an image, inspect it, check quality, save a copy.

Run from the project root:
    python tests/test_milestone_1.py
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt

# Make the 'src' folder importable regardless of where this script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (  # noqa: E402
    load_image,
    get_image_info,
    check_image_quality,
    save_processed_copy,
)


IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "samples", "sample_document.jpg")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "sample_document_gray.png")


def print_section(title: str) -> None:
    """Print a clear section header to the console."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    print_section("MILESTONE 1: IMAGE INSPECTION")

    # 1. Make sure the test image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"\n[!] No test image found at: {IMAGE_PATH}")
        print("    Run this first:")
        print("        python tests/generate_test_image.py")
        print("    Or place your own image at that path.")
        return

    # 2. Load
    print(f"\nLoading image: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
    print("[OK] Image loaded successfully")

    # 3. Info
    print_section("Image Information")
    info = get_image_info(image)
    for key, value in info.items():
        print(f"  {key:<14}: {value}")

    # 4. Quality
    print_section("Image Quality")
    quality = check_image_quality(image)
    for key, value in quality.items():
        print(f"  {key:<18}: {value}")

    # 5. Save processed copy
    print_section("Saving Processed Copy")
    saved_path = save_processed_copy(image, OUTPUT_PATH)
    print(f"[OK] Grayscale copy saved to: {saved_path}")

    # 6. Display
    print_section("Displaying Image")
    print("A matplotlib window will open. Close it to finish.")

    # OpenCV uses BGR; matplotlib expects RGB
    if len(image.shape) == 3:
        image_to_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cmap = None
    else:
        image_to_show = image
        cmap = "gray"

    plt.figure(figsize=(10, 6))
    plt.imshow(image_to_show, cmap=cmap)
    plt.title("Loaded Document Image")
    plt.axis("on")
    plt.show()

    print_section("MILESTONE 1 COMPLETE")
    print("Everything worked. Ready for Milestone 2 when you are.")


if __name__ == "__main__":
    main()
