"""
Generate a synthetic dataset of 'genuine' and 'manipulated' document images.

Supports all core document types: Aadhaar, PAN, and Driving Licence,
simulating realistic card layouts, scanner noise, and digital tampering artifacts.
"""

import os
import cv2
import numpy as np


def _blank_document(doc_type: str = "aadhaar", width: int = 600, height: int = 400) -> np.ndarray:
    """Create a clean document image with subtle scan/print noise."""
    bg_val = np.random.randint(235, 255)
    base = np.ones((height, width, 3), dtype=np.uint8) * bg_val
    noise = np.random.normal(0, np.random.uniform(1.0, 3.5), base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    border_thick = max(1, int(width / 300))
    cv2.rectangle(base, (6, 6), (width - 6, height - 6), (30, 30, 30), border_thick)

    scale = width / 600.0
    fs = max(0.4, 0.5 * scale)
    th = max(1, int(scale))

    if doc_type == "aadhaar":
        cv2.putText(base, "AADHAAR", (int(width * 0.3), int(height * 0.18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (20, 20, 20), max(1, int(2 * scale)))
        cv2.putText(base, "Name: Test Person", (int(width * 0.08), int(height * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "Number: 1234 5678 9012", (int(width * 0.08), int(height * 0.50)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "DOB: 01/01/1990", (int(width * 0.08), int(height * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "Gender: M", (int(width * 0.08), int(height * 0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        qr_size = int(height * 0.32)
        cv2.rectangle(base, (width - qr_size - 15, height - qr_size - 15),
                      (width - 15, height - 15), (0, 0, 0), -1)

    elif doc_type == "pan":
        cv2.putText(base, "INCOME TAX DEPARTMENT - PAN", (int(width * 0.10), int(height * 0.18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (20, 20, 20), max(1, int(2 * scale)))
        cv2.putText(base, "Name: Jane Smith", (int(width * 0.08), int(height * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "Father Name: Robert Smith", (int(width * 0.08), int(height * 0.50)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "PAN: ABCDE1234F", (int(width * 0.08), int(height * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "DOB: 05/05/1985", (int(width * 0.08), int(height * 0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)

    elif doc_type == "license":
        cv2.putText(base, "UNION OF INDIA Driving Licence", (int(width * 0.08), int(height * 0.18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (20, 20, 20), max(1, int(2 * scale)))
        cv2.putText(base, "HP-32 20140003137", (int(width * 0.08), int(height * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "Validity: 06-03-2034", (int(width * 0.08), int(height * 0.50)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "Date of Birth: 12-08-1990", (int(width * 0.08), int(height * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        cv2.putText(base, "Name: CHUNNI LAL", (int(width * 0.08), int(height * 0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
        pw, ph = int(width * 0.2), int(height * 0.5)
        cv2.rectangle(base, (width - pw - 15, int(height * 0.28)),
                      (width - 15, int(height * 0.28) + ph), (120, 120, 120), -1)

    return base


def _manipulate_document(image: np.ndarray) -> np.ndarray:
    """Simulate tampering by overwriting a region with no noise + sharp edges."""
    h, w = image.shape[:2]
    manipulated = image.copy()
    
    x1 = np.random.randint(int(w * 0.08), int(w * 0.5))
    y1 = np.random.randint(int(h * 0.25), int(h * 0.7))
    x2 = min(w - 10, x1 + np.random.randint(int(w * 0.2), int(w * 0.4)))
    y2 = min(h - 10, y1 + np.random.randint(int(h * 0.08), int(h * 0.18)))
    
    manipulated[y1:y2, x1:x2] = np.random.randint(220, 255)
    cv2.rectangle(manipulated, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    if np.random.random() < 0.3:
        manipulated[y1:y2, x1:x2] = cv2.GaussianBlur(
            manipulated[y1:y2, x1:x2], (3, 3), 0
        )
    return manipulated


def generate_dataset(output_dir: str, n_genuine: int = 45,
                     n_manipulated: int = 45) -> dict:
    """
    Create a balanced multi-document synthetic dataset with labels.

    Returns:
        Dict with 'genuine' and 'manipulated' lists of file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    genuine_dir = os.path.join(output_dir, "genuine")
    manipulated_dir = os.path.join(output_dir, "manipulated")
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(manipulated_dir, exist_ok=True)

    paths = {"genuine": [], "manipulated": []}
    doc_types = ["aadhaar", "pan", "license"]
    resolutions = [(600, 400), (333, 248), (800, 500)]

    for i in range(n_genuine):
        dtype = doc_types[i % len(doc_types)]
        res = resolutions[(i // len(doc_types)) % len(resolutions)]
        img = _blank_document(doc_type=dtype, width=res[0], height=res[1])
        
        h, w = img.shape[:2]
        M = np.float32([[1, 0, np.random.randint(-2, 3)],
                        [0, 1, np.random.randint(-2, 3)]])
        img = cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 245))
        
        path = os.path.join(genuine_dir, f"genuine_{i:03d}.png")
        cv2.imwrite(path, img)
        paths["genuine"].append(path)

    for i in range(n_manipulated):
        dtype = doc_types[i % len(doc_types)]
        res = resolutions[(i // len(doc_types)) % len(resolutions)]
        img = _blank_document(doc_type=dtype, width=res[0], height=res[1])
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

