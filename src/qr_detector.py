"""
QR code detection and decoding.

Uses OpenCV's built-in QRCodeDetector. Each detected QR is reported
with its location, decoded contents, and a readability flag.

We never claim a QR proves authenticity — only that it is present,
located somewhere, and decodable. Content verification is a separate step.
"""

import os
import cv2
import numpy as np


def detect_qr_codes(image: np.ndarray) -> dict:
    """
    Detect and decode QR codes in the image.

    Args:
        image: BGR or grayscale image as a NumPy array.

    Returns:
        Dictionary with:
            - count (int): number of QR codes detected
            - qrcodes (list): one entry per QR, with bbox, decoded text,
                              readability, and the straight (decoded) image
    """
    if image is None:
        raise ValueError("Image is None")

    detector = cv2.QRCodeDetector()

    qrcodes = []

    try:
        retval, decoded_info, points, straight_qrcodes = (
            detector.detectAndDecodeMulti(image)
        )
    except cv2.error as e:
        return {
            "count": 0,
            "qrcodes": [],
            "warning": f"QR detection failed: {e}",
        }

    if points is None or len(points) == 0:
        return {"count": 0, "qrcodes": []}

    if decoded_info is None:
        decoded_info = ["" for _ in range(len(points))]

    for i, point_set in enumerate(points):
        try:
            x_coords = [float(p[0]) for p in point_set]
            y_coords = [float(p[1]) for p in point_set]
        except (TypeError, IndexError):
            continue

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        decoded_text = decoded_info[i] if i < len(decoded_info) else ""
        readable = bool(decoded_text)

        qrcodes.append({
            "bbox": {
                "x": x_min,
                "y": y_min,
                "w": x_max - x_min,
                "h": y_max - y_min,
            },
            "polygon": [
                [int(p[0]), int(p[1])] for p in point_set
            ],
            "decoded_text": decoded_text,
            "readable": readable,
            "decoded_length": len(decoded_text),
        })

    return {
        "count": len(qrcodes),
        "qrcodes": qrcodes,
    }


def visualize_qr(image: np.ndarray, qr_result: dict, output_path: str) -> str:
    """
    Draw bounding boxes around detected QR codes and label them.

    Args:
        image: Original image.
        qr_result: Output of detect_qr_codes().
        output_path: Where to save the annotated image.

    Returns:
        Absolute path of the saved file.
    """
    if image is None:
        raise ValueError("Image is None")

    annotated = image.copy()

    for idx, qr in enumerate(qr_result["qrcodes"]):
        b = qr["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        polygon = np.array(qr["polygon"], dtype=np.int32)
        cv2.polylines(annotated, [polygon], isClosed=True,
                      color=(255, 0, 0), thickness=2)

        label = f"QR #{idx + 1} ({'READABLE' if qr['readable'] else 'UNREADABLE'})"
        cv2.putText(
            annotated, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return os.path.abspath(output_path)
