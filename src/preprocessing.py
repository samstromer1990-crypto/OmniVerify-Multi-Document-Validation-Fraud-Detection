"""
Image loading and basic quality inspection.
"""

import os

import cv2
import numpy as np


def load_image(
    image_path: str
) -> np.ndarray:

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found at: "
            f"{image_path}"
        )

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "OpenCV could not read "
            f"the image: {image_path}"
        )

    return image


def get_image_info(
    image: np.ndarray
) -> dict:

    if image is None:
        raise ValueError(
            "Image is None"
        )

    height, width = (
        image.shape[:2]
    )

    channels = (
        image.shape[2]
        if len(image.shape) == 3
        else 1
    )

    if channels == 3:

        color_format = (
            "BGR (color, "
            "OpenCV default order)"
        )

    elif channels == 1:

        color_format = (
            "Grayscale"
        )

    elif channels == 4:

        color_format = (
            "BGRA "
            "(color with alpha)"
        )

    else:

        color_format = (
            f"Unknown ({channels} channels)"
        )

    return {

        "width":
            width,

        "height":
            height,

        "channels":
            channels,

        "color_format":
            color_format,

        "total_pixels":
            width * height,

        "aspect_ratio":
            round(
                width / height,
                3
            ),
    }


def _to_grayscale(
    image: np.ndarray
) -> np.ndarray:

    if len(image.shape) == 2:
        return image

    if image.shape[2] == 3:

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    if image.shape[2] == 4:

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY
        )

    raise ValueError(
        "Unsupported number of "
        f"channels: {image.shape[2]}"
    )


def check_image_quality(
    image: np.ndarray
) -> dict:

    if image is None:
        raise ValueError(
            "Image is None"
        )

    gray = _to_grayscale(
        image
    )

    # Blur
    laplacian_variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # Brightness
    brightness = float(
        np.mean(gray)
    )

    # Contrast
    contrast = float(
        np.std(gray)
    )

    # Resolution
    height, width = (
        gray.shape
    )

    minimum_resolution = 200

    # Blur classification
    if laplacian_variance < 100:

        blur_status = "BLURRY"

    elif laplacian_variance < 300:

        blur_status = (
            "SLIGHTLY_BLURRY"
        )

    else:

        blur_status = "SHARP"

    # Brightness classification
    #
    # Document backgrounds are naturally bright.
    # 210 was too strict for synthetic documents.
    if brightness < 50:

        brightness_status = (
            "TOO_DARK"
        )

    elif brightness > 250:

        brightness_status = (
            "TOO_BRIGHT"
        )

    else:

        brightness_status = "OK"

    # Contrast
    if contrast < 40:

        contrast_status = (
            "LOW_CONTRAST"
        )

    else:

        contrast_status = "OK"

    # Resolution
    if min(
        width,
        height
    ) < minimum_resolution:

        resolution_status = (
            "LOW_RESOLUTION"
        )

    else:

        resolution_status = "OK"

    issues = []

    if blur_status == "BLURRY":

        issues.append(
            f"Blur: {blur_status}"
        )

    if brightness_status != "OK":

        issues.append(
            f"Brightness: "
            f"{brightness_status}"
        )

    if contrast_status != "OK":

        issues.append(
            f"Contrast: "
            f"{contrast_status}"
        )

    if resolution_status != "OK":

        issues.append(
            f"Resolution: "
            f"{resolution_status}"
        )

    overall = (
        "GOOD"
        if not issues
        else "ISSUES_FOUND"
    )

    return {

        "laplacian_variance":
            round(
                laplacian_variance,
                2
            ),

        "blur_status":
            blur_status,

        "brightness":
            round(
                brightness,
                2
            ),

        "brightness_status":
            brightness_status,

        "contrast":
            round(
                contrast,
                2
            ),

        "contrast_status":
            contrast_status,

        "resolution":
            f"{width}x{height}",

        "resolution_status":
            resolution_status,

        "issues":
            issues,

        "overall_quality":
            overall,
    }


def save_processed_copy(
    image: np.ndarray,
    output_path: str
) -> str:

    gray = _to_grayscale(
        image
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
        gray
    )

    return os.path.abspath(
        output_path
    )