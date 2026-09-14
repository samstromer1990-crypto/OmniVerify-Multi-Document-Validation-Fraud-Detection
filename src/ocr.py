"""
OCR module.

Uses Tesseract through pytesseract.
Multiple preprocessing methods are tried so that
OCR is more reliable on document images.
"""

import os
import json
import cv2
import numpy as np
import pytesseract
import platform


# Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


def _run_tesseract(image: np.ndarray) -> dict:
    """Run Tesseract and return structured OCR data."""

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 6"
    )

    words = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if conf < 0:
            continue

        words.append({
            "text": text,
            "confidence": round(conf, 2),
            "bbox": {
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
            },
        })

    full_text = " ".join(
        w["text"] for w in words
    )

    mean_conf = (
        round(
            float(
                np.mean([
                    w["confidence"]
                    for w in words
                ])
            ),
            2
        )
        if words
        else 0.0
    )

    return {
        "text": full_text,
        "word_count": len(words),
        "mean_confidence": mean_conf,
        "words": words,
    }


def extract_text(image: np.ndarray) -> dict:
    """
    Run OCR using multiple preprocessing methods.
    The strongest OCR result is returned.
    """

    if image is None:
        raise ValueError("Image is None")

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    results = []

    # --------------------------------------------------
    # Method 1: Original grayscale
    # --------------------------------------------------
    results.append(
        _run_tesseract(gray)
    )

    # --------------------------------------------------
    # Method 2: Adaptive threshold
    # --------------------------------------------------
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    results.append(
        _run_tesseract(adaptive)
    )

    # --------------------------------------------------
    # Method 3: Otsu threshold
    # --------------------------------------------------
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    results.append(
        _run_tesseract(otsu)
    )

    # Choose result with most words,
    # then use confidence as a tie-breaker.
    best = max(
        results,
        key=lambda r: (
            r["word_count"],
            r["mean_confidence"]
        )
    )

    return best


def visualize_ocr(
    image: np.ndarray,
    ocr_result: dict,
    output_path: str
) -> str:

    if image is None:
        raise ValueError("Image is None")

    annotated = image.copy()

    for word in ocr_result["words"]:

        b = word["bbox"]

        x = b["x"]
        y = b["y"]
        w = b["w"]
        h = b["h"]

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            word["text"],
            (x, max(y - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    cv2.imwrite(
        output_path,
        annotated
    )

    return os.path.abspath(output_path)


def save_ocr_result(
    ocr_result: dict,
    output_path: str
) -> str:

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            ocr_result,
            f,
            indent=2,
            ensure_ascii=False
        )

    return os.path.abspath(output_path)
