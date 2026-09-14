"""
Image loading and basic quality inspection.

This module is the foundation: every later module (OCR, QR detection,
fraud analysis) will assume the image has already been loaded and
validated through these functions.
"""

import os
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and validate that it opened correctly.

    Args:
        image_path: Path to the image file.

    Returns:
        The image as a NumPy array (BGR format if color).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If OpenCV cannot decode the file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"OpenCV could not read the image. "
            f"File may be corrupted or in an unsupported format: {image_path}"
        )

    return image


def get_image_info(image: np.ndarray) -> dict:
    """
    Return basic information about an image.

    Args:
        image: NumPy array representing the image.

    Returns:
        Dictionary with width, height, channels, color format, etc.
    """
    if image is None:
        raise ValueError("Image is None")

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    if channels == 3:
        color_format = "BGR (color, OpenCV default order)"
    elif channels == 1:
        color_format = "Grayscale"
    elif channels == 4:
        color_format = "BGRA (color with alpha)"
    else:
        color_format = f"Unknown ({channels} channels)"

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "color_format": color_format,
        "total_pixels": width * height,
        "aspect_ratio": round(width / height, 3),
    }


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Internal helper: convert any image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")


def check_image_quality(image: np.ndarray) -> dict:
    """
    Run basic image-quality checks.

    Checks performed:
        - Blur (via Laplacian variance)
        - Brightness (mean pixel intensity)
        - Contrast (std dev of pixel intensities)
        - Resolution (minimum dimension)

    Args:
        image: NumPy array representing the image.

    Returns:
        Dictionary of quality measurements and a status for each,
        plus an overall verdict and a list of issues.
    """
    if image is None:
        raise ValueError("Image is None")

    gray = _to_grayscale(image)

    # Blur: variance of the Laplacian. High = sharp, low = blurry.
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness and contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Resolution
    height, width = gray.shape
    min_resolution_threshold = 200

    # Classify each metric
    if laplacian_var < 100:
        blur_status = "BLURRY"
    elif laplacian_var < 300:
        blur_status = "SLIGHTLY_BLURRY"
    else:
        blur_status = "SHARP"

    if brightness < 50:
        brightness_status = "TOO_DARK"
    elif brightness > 210:
        brightness_status = "TOO_BRIGHT"
    else:
        brightness_status = "OK"

    if contrast < 40:
        contrast_status = "LOW_CONTRAST"
    else:
        contrast_status = "OK"

    if min(width, height) < min_resolution_threshold:
        resolution_status = "LOW_RESOLUTION"
    else:
        resolution_status = "OK"

    # Overall summary.
    # SLIGHTLY_BLURRY is tolerated as a pass condition — real-world
    # phone/screenshot photos are rarely tack-sharp, and treating any
    # non-"SHARP" image as an automatic fail was too strict.
    issues = []
    if blur_status == "BLURRY":
        issues.append(f"Blur: {blur_status}")
    if brightness_status != "OK":
        issues.append(f"Brightness: {brightness_status}")
    if contrast_status != "OK":
        issues.append(f"Contrast: {contrast_status}")
    if resolution_status != "OK":
        issues.append(f"Resolution: {resolution_status}")

    overall = "GOOD" if not issues else "ISSUES_FOUND"

    return {
        "laplacian_variance": round(laplacian_var, 2),
        "blur_status": blur_status,
        "brightness": round(brightness, 2),
        "brightness_status": brightness_status,
        "contrast": round(contrast, 2),
        "contrast_status": contrast_status,
        "resolution": f"{width}x{height}",
        "resolution_status": resolution_status,
        "issues": issues,
        "overall_quality": overall,
    }


def save_processed_copy(image: np.ndarray, output_path: str) -> str:
    """
    Save a grayscale copy of the image for downstream use.

    Args:
        image: The original image.
        output_path: Where to save the grayscale copy.

    Returns:
        The absolute path of the saved file.
    """
    gray = _to_grayscale(image)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, gray)
    return os.path.abspath(output_path)
