"""
QR code detection and decoding.

Uses OpenCV's built-in QRCodeDetector. Each detected QR is reported
with its location, decoded contents, and a readability flag.

OpenCV's detector is unreliable on small/scaled-down QR regions (common
in screenshots or compressed uploads), so detection is retried at a
few scales/color modes before giving up.

We never claim a QR proves authenticity — only that it is present,
located somewhere, and decodable. Content verification is a separate step.
"""

import os
import cv2
import numpy as np


def _attempts(image: np.ndarray):
    """Yield (candidate_image, scale_factor) pairs to try detection on."""
    yield image, 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    yield gray, 1.0
    for scale in (2.0, 3.0):
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        yield resized, scale


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

    decoded_info, points, used_scale = None, None, 1.0

    for candidate, scale in _attempts(image):
        try:
            retval, d_info, pts, _ = detector.detectAndDecodeMulti(candidate)
        except cv2.error:
            continue
        if pts is not None and len(pts) > 0:
            decoded_info, points, used_scale = d_info, pts, scale
            break

    if points is None or len(points) == 0:
        return {"count": 0, "qrcodes": []}

    if decoded_info is None:
        decoded_info = ["" for _ in range(len(points))]

    qrcodes = []
    for i, point_set in enumerate(points):
        try:
            # Map coordinates back to the original image scale.
            x_coords = [float(p[0]) / used_scale for p in point_set]
            y_coords = [float(p[1]) / used_scale for p in point_set]
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
                [int(x / used_scale), int(y / used_scale)]
                for x, y in zip(
                    [p[0] for p in point_set], [p[1] for p in point_set]
                )
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
