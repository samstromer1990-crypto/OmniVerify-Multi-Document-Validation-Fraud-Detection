"""
Image feature extraction for fraud detection.

Extracts numerical features from a document image that an ML model
can use to predict tampering risk.

All features are simple, deterministic, and explainable. We avoid
deep features because we want a student-level system that is easy
to defend in viva.
"""

import cv2
import numpy as np


def extract_features(image: np.ndarray, ocr_result: dict = None,
                     qr_result: dict = None) -> dict:
    """
    Compute fraud-detection features for an image.
    """
    if image is None:
        raise ValueError("Image is None")

    # Always work on grayscale for most features
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape

    # 1. Edge density (Canny edges / total pixels).
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    # 2. Laplacian variance (sharpness proxy).
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3. Noise estimate (median absolute deviation of high-pass residual).
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.median(np.abs(gray.astype(np.int16)
                                    - blurred.astype(np.int16))))

    # 4. Brightness & contrast.
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # 5. Color consistency (only meaningful for color images).
    if len(image.shape) == 3:
        b, g, r = (float(np.mean(image[:, :, c])) for c in range(3))
        color_mean_diff = float(np.std([b, g, r]))
    else:
        color_mean_diff = 0.0

    # 6. Resolution.
    resolution = float(min(h, w))

    # 7. OCR-derived features (if available).
    if ocr_result:
        word_count = float(ocr_result.get("word_count", 0))
        mean_conf = float(ocr_result.get("mean_confidence", 0.0))
        words = ocr_result.get("words", [])
        if words:
            low_conf_ratio = float(
                sum(1 for w in words if w["confidence"] < 60)
            ) / float(len(words))
        else:
            low_conf_ratio = 1.0
    else:
        word_count = 0.0
        mean_conf = 0.0
        low_conf_ratio = 1.0

    # 8. QR-derived feature (if available).
    if qr_result:
        qr_detected = float(qr_result.get("count", 0) > 0)
        qr_readable = float(
            any(q.get("readable") for q in qr_result.get("qrcodes", []))
        )
    else:
        qr_detected = 0.0
        qr_readable = 0.0

    return {
        "edge_density": edge_density,
        "laplacian_variance": lap_var,
        "noise_estimate": noise,
        "brightness": brightness,
        "contrast": contrast,
        "color_mean_diff": color_mean_diff,
        "resolution": resolution,
        "word_count": word_count,
        "ocr_mean_confidence": mean_conf,
        "ocr_low_conf_ratio": low_conf_ratio,
        "qr_detected": qr_detected,
        "qr_readable": qr_readable,
    }


FEATURE_NAMES = [
    "edge_density",
    "laplacian_variance",
    "noise_estimate",
    "brightness",
    "contrast",
    "color_mean_diff",
    "resolution",
    "word_count",
    "ocr_mean_confidence",
    "ocr_low_conf_ratio",
    "qr_detected",
    "qr_readable",
]
