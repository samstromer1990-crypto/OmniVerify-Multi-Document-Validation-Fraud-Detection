"""
Robust Tesseract OCR for document images.

Several OCR configurations are tried and the most useful
document-oriented result is selected.
"""

import os
import json
import cv2
import numpy as np
import pytesseract
import platform
import re


# Windows
if platform.system() == "Windows":

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


def _ensure_readable_size(gray: np.ndarray) -> np.ndarray:
    """
    Tesseract needs characters roughly 20+ pixels tall.
    Phone screenshots and compressed uploads are often ~250-400px,
    which is too small for a Driving Licence number line.
    """
    height, width = gray.shape[:2]
    min_side = min(height, width)
    target = 900

    if min_side >= target:
        return gray

    scale = target / float(min_side)
    return cv2.resize(
        gray,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )


def _run_tesseract(
    image: np.ndarray,
    psm: int
) -> dict:

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config=f"--psm {psm}",
    )

    words = []

    for i, raw_text in enumerate(
        data["text"]
    ):

        text = raw_text.strip()

        if not text:
            continue

        try:
            confidence = float(
                data["conf"][i]
            )
        except (
            ValueError,
            TypeError
        ):
            continue

        if confidence < 0:
            continue

        words.append({
            "text": text,
            "confidence": round(
                confidence,
                2
            ),
            "bbox": {
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
            },
        })

    full_text = " ".join(
        word["text"]
        for word in words
    )

    mean_confidence = (
        round(
            float(
                np.mean([
                    word["confidence"]
                    for word in words
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
        "mean_confidence": mean_confidence,
        "words": words,
        "_psm": psm,
    }


def _score_result(
    result: dict
) -> float:
    """
    Score OCR results using document signals
    rather than simply counting words.
    """

    text = result["text"]

    lower = text.lower()

    score = (
        result["mean_confidence"]
        * 0.5
    )

    # Aadhaar
    if re.search(
        r"\b(?:aadhaar|aadhar|adhaar|uidai)\b",
        lower
    ):
        score += 40

    if re.search(
        r"(?<!\d)(?:\d{4}\s*){3}(?!\d)",
        text
    ):
        score += 35

    # PAN
    if (
        "income tax" in lower
        or re.search(
            r"\bpan\b",
            lower
        )
    ):
        score += 35

    if re.search(
        r"\bpan\s*[:\-]?\s*[a-z]{5}\d{4}[a-z]\b",
        lower,
        re.IGNORECASE
    ):
        score += 35

    # Driving Licence
    if (
        "driving licence" in lower
        or "driving license" in lower
    ):
        score += 45

    if re.search(
        r"\b\d{16}\b",
        text
    ):
        score += 25

    # Typical Indian DL: HP-32 20140003137 / UP14 20240042764
    # OCR often turns the hyphen into a period: HP.32
    if re.search(
        r"\b[A-Z]{2}[\s.\-/]*\d{1,4}[\s.\-/]*\d{6,13}\b",
        text,
        re.IGNORECASE,
    ):
        score += 30

    # Useful document fields
    for field in (
        "name",
        "dob",
        "gender",
        "father",
        "validity",
    ):

        if field in lower:
            score += 4

    # Penalize extremely noisy OCR
    if result["word_count"] > 100:

        score -= (
            result["word_count"] - 100
        ) * 0.35

    return score


def extract_text(
    image: np.ndarray
) -> dict:

    if image is None:
        raise ValueError(
            "Image is None"
        )

    # Convert to grayscale
    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image.copy()

    gray = _ensure_readable_size(gray)

    candidates = []

    # Original grayscale
    for psm in (
        6,
        11,
        12,
        3,
    ):

        candidates.append(
            _run_tesseract(
                gray,
                psm
            )
        )

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    candidates.append(
        _run_tesseract(
            adaptive,
            11
        )
    )

    # Otsu threshold
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    candidates.append(
        _run_tesseract(
            otsu,
            11
        )
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    candidates.append(
        _run_tesseract(
            clahe.apply(gray),
            6
        )
    )

    # Select the most useful result
    best = max(
        candidates,
        key=_score_result
    )

    best.pop(
        "_psm",
        None
    )

    return best


def visualize_ocr(
    image: np.ndarray,
    ocr_result: dict,
    output_path: str
) -> str:

    if image is None:
        raise ValueError(
            "Image is None"
        )

    annotated = image.copy()

    for word in ocr_result["words"]:

        bbox = word["bbox"]

        x = bbox["x"]
        y = bbox["y"]
        w = bbox["w"]
        h = bbox["h"]

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
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    directory = os.path.dirname(
        output_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    cv2.imwrite(
        output_path,
        annotated
    )

    return os.path.abspath(
        output_path
    )


def save_ocr_result(
    ocr_result: dict,
    output_path: str
) -> str:

    directory = os.path.dirname(
        output_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            ocr_result,
            file,
            indent=2,
            ensure_ascii=False
        )

    return os.path.abspath(
        output_path
    )