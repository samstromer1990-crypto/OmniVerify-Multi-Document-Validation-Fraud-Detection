"""
OCR module: extract text, bounding boxes, and confidence from a document image.

Uses Tesseract via pytesseract. Designed to return *structured* results
(bounding boxes preserved) because later modules need to know WHERE on
the document each piece of text was found.
"""

import os
import json
import cv2
import numpy as np
import pytesseract

import platform

# Smart path detection:
if platform.system() == "Windows":
    # Use the Windows path for your local computer
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # On Linux (Render/Docker), Tesseract is already in the system PATH
    # so we don't need to set a manual path.
    pass


def extract_text(image: np.ndarray) -> dict:
    """
    Run OCR on an image and return a structured result.
    Includes adaptive thresholding to handle AI-generated images with gradients.
    """
    if image is None:
        raise ValueError("Image is None")

    # 1. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Apply Adaptive Thresholding 
    # This removes gradients and shadows, making text "pop" for Tesseract
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Tesseract will now read the high-contrast black-and-white image
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        # Tesseract returns -1 for confidence on blocks it didn't process
        if not text or data["conf"][i] == "-1":
            continue

        try:
            conf = float(data["conf"][i])
        except ValueError:
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

    full_text = " ".join(w["text"] for w in words)
    mean_conf = (
        round(float(np.mean([w["confidence"] for w in words])), 2)
        if words else 0.0
    )

    return {
        "text": full_text,
        "word_count": len(words),
        "mean_confidence": mean_conf,
        "words": words,
    }


def visualize_ocr(image: np.ndarray, ocr_result: dict, output_path: str) -> str:
    """
    Draw bounding boxes around detected words and save the visualization.
    """
    if image is None:
        raise ValueError("Image is None")

    annotated = image.copy()

    for word in ocr_result["words"]:
        b = word["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = word["text"]
        cv2.putText(
            annotated, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return os.path.abspath(output_path)


def save_ocr_result(ocr_result: dict, output_path: str) -> str:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_result, f, indent=2, ensure_ascii=False)
    return os.path.abspath(output_path)
