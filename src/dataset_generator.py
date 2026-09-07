"""
Generate a synthetic dataset of 'genuine' and 'manipulated' document images.

This avoids using any real personal documents. We simulate:
  - Genuine: clean synthetic document with text and field placeholders.
  - Manipulated: a copy-pasted/overwritten region that creates suspicious
    visual artifacts (mismatched noise, sharp boundaries, etc.).
"""

import os
import cv2
import numpy as np


def _blank_document(width: int = 600, height: int = 400) -> np.ndarray:
    """Create a clean white document with subtle noise (mimics scan noise)."""
    base = np.ones((height, width, 3), dtype=np.uint8) * 245
    # Add subtle Gaussian noise (mimics scanner/print noise)
    noise = np.random.normal(0, 4, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Border
    cv2.rectangle(base, (8, 8), (width - 8, height - 8), (30, 30, 30), 2)
    # Header text
    cv2.putText(base, "AADHAAR", (180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(base, "Name: Test Person",
                (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(base, "Number: 123456789012",
                (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(base, "DOB: 01/01/1990",
                (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(base, "Gender: M",
                (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return base


def _manipulate_document(image: np.ndarray) -> np.ndarray:
    """Simulate tampering by overwriting a region with no noise + sharp edges."""
    h, w = image.shape[:2]
    manipulated = image.copy()
    # Choose a region (over the number or DOB usually)
    x1 = np.random.randint(40, w - 200)
    y1 = np.random.randint(120, h - 80)
    x2 = x1 + np.random.randint(120, 200)
    y2 = y1 + np.random.randint(30, 50)
    # Overwrite with a flat patch (NO noise — looks artificially clean)
    manipulated[y1:y2, x1:x2] = 240
    # Add a sharp black border around the patch (a common tampering tell)
    cv2.rectangle(manipulated, (x1, y1), (x2, y2), (0, 0, 0), 2)
    # Optionally add some random artifacts
    if np.random.random() < 0.3:
        manipulated[y1:y2, x1:x2] = cv2.GaussianBlur(
            manipulated[y1:y2, x1:x2], (3, 3), 0
        )
    return manipulated


def generate_dataset(output_dir: str, n_genuine: int = 40,
                     n_manipulated: int = 40) -> dict:
    """
    Create a synthetic dataset of document images with labels.

    Returns:
        Dict with 'genuine' and 'manipulated' lists of file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    genuine_dir = os.path.join(output_dir, "genuine")
    manipulated_dir = os.path.join(output_dir, "manipulated")
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(manipulated_dir, exist_ok=True)

    paths = {"genuine": [], "manipulated": []}

    for i in range(n_genuine):
        img = _blank_document()
        h, w = img.shape[:2]
        M = np.float32([[1, 0, np.random.randint(-3, 3)],
                        [0, 1, np.random.randint(-3, 3)]])
        img = cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 245))
        path = os.path.join(genuine_dir, f"genuine_{i:03d}.png")
        cv2.imwrite(path, img)
        paths["genuine"].append(path)

    for i in range(n_manipulated):
        img = _blank_document()
        img = _manipulate_document(img)
        path = os.path.join(manipulated_dir, f"manipulated_{i:03d}.png")
        cv2.imwrite(path, img)
        paths["manipulated"].append(path)

    return paths


def label_for_path(image_path: str) -> str:
    """Helper: derive label from the parent folder name."""
    parent = os.path.basename(os.path.dirname(image_path))
    if parent == "genuine":
        return "GENUINE"
    if parent == "manipulated":
        return "MANIPULATED"
    return "UNKNOWN"

