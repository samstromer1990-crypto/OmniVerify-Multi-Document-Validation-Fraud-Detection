import re
import numpy as np

from src.document_types import (
    DOCUMENT_CONFIG,
    detect_document_type
)


def _find_id_number(
    words: list,
    expected_len: int,
    tolerance: int = 3
) -> dict:

    candidates = []

    # --------------------------------------------------
    # 1. Check individual OCR words
    # --------------------------------------------------
    for w in words:

        raw_text = str(
            w.get("text", "")
        )

        digits = re.sub(
            r"\D",
            "",
            raw_text
        )

        if len(digits) == expected_len:

            candidates.append({
                "digits": digits,
                "source_word": raw_text,
                "bbox": w.get("bbox"),
                "confidence": w.get("confidence"),
            })

    if candidates:

        candidates.sort(
            key=lambda x: x["confidence"] or 0,
            reverse=True
        )

        return {
            "found": True,
            "count": len(candidates),
            "best": candidates[0],
        }

    # --------------------------------------------------
    # 2. Combine numeric OCR words
    #
    # Example:
    # 9999 8888 7777
    #
    # becomes:
    # 999988887777
    # --------------------------------------------------

    numeric_words = []

    for w in words:

        raw_text = str(
            w.get("text", "")
        )

        digits = re.sub(
            r"\D",
            "",
            raw_text
        )

        if len(digits) >= 2:

            numeric_words.append({
                "digits": digits,
                "bbox": w.get("bbox"),
                "confidence": w.get("confidence"),
                "source_word": raw_text,
            })

    # Sort by location
    numeric_words.sort(
        key=lambda w: (
            w["bbox"]["y"]
            if w["bbox"]
            else 0,

            w["bbox"]["x"]
            if w["bbox"]
            else 0,
        )
    )

    # Try consecutive groups
    for start in range(
        len(numeric_words)
    ):

        combined = ""
        selected = []

        for current in range(
            start,
            len(numeric_words)
        ):

            combined += (
                numeric_words[current]["digits"]
            )

            selected.append(
                numeric_words[current]
            )

            # Exact length found
            if len(combined) == expected_len:

                confidences = [
                    x["confidence"]
                    for x in selected
                    if x["confidence"] is not None
                ]

                confidence = (
                    round(
                        float(
                            np.mean(confidences)
                        ),
                        2
                    )
                    if confidences
                    else None
                )

                return {
                    "found": True,
                    "count": 1,
                    "best": {
                        "digits": combined,
                        "source_word": " ".join(
                            x["source_word"]
                            for x in selected
                        ),
                        "bbox": None,
                        "confidence": confidence,
                    },
                }

            # Stop if already too long
            if len(combined) > expected_len:
                break

    # --------------------------------------------------
    # 3. Conservative full-text fallback
    # --------------------------------------------------

    all_digits = re.sub(
        r"\D",
        "",
        " ".join(
            str(w.get("text", ""))
            for w in words
        )
    )

    if (
        len(all_digits) > 0
        and abs(
            len(all_digits) - expected_len
        ) <= tolerance
    ):

        return {
            "found": True,
            "count": 1,
            "best": {
                "digits": all_digits,
                "source_word": "(concatenated)",
                "bbox": None,
                "confidence": None,
            },
        }

    return {
        "found": False,
        "count": 0,
        "best": None,
    }


FIELD_SYNONYMS = {

    "dob": [
        "dob",
        "date of birth",
        "birth"
    ],

    "name": [
        "name"
    ],

    "gender": [
        "gender",
        "male",
        "female"
    ],

    "father's name": [
        "father's name",
        "father name",
        "father"
    ],

    "validity": [
        "validity",
        "valid till",
        "valid upto",
        "valid up to",
        "validity upto",
        "validity up to"
    ],
}


def _detect_required_fields(
    words: list,
    required_fields: list
) -> dict:

    text_lower = " ".join(
        str(w.get("text", "")).lower()
        for w in words
    )

    found = {}

    for field in required_fields:

        options = FIELD_SYNONYMS.get(
            field,
            [field]
        )

        found[field] = any(
            option in text_lower
            for option in options
        )

    return {
        "fields": found,
        "all_present": all(
            found.values()
        ),
    }


def validate_document(
    ocr_result: dict,
    quality_result: dict,
    qr_result: dict,
    image_path: str = ""
) -> dict:

    words = ocr_result.get(
        "words",
        []
    )

    # --------------------------------------------------
    # Document type detection
    # --------------------------------------------------

    doc_type = detect_document_type(
        words
    )

    # --------------------------------------------------
    # Quality
    #
    # Slight blur is allowed.
    # Only serious problems fail the quality check.
    # --------------------------------------------------

    image_ok = (
        quality_result.get(
            "brightness_status"
        ) == "OK"

        and quality_result.get(
            "contrast_status"
        ) == "OK"

        and quality_result.get(
            "resolution_status"
        ) == "OK"

        and quality_result.get(
            "blur_status"
        ) != "BLURRY"
    )

    ocr_ok = (
        ocr_result.get(
            "word_count",
            0
        ) > 0
    )

    qr_detected = (
        qr_result.get(
            "count",
            0
        ) > 0
    )

    # --------------------------------------------------
    # Fallback document detection
    # --------------------------------------------------

    if doc_type == "unknown":

        img_name = (
            image_path.lower()
            if image_path
            else ""
        )

        full_text = " ".join(
            str(w.get("text", ""))
            for w in words
        ).lower()

        hint_map = {
            "aadhaar": (
                "aadhaar",
                "aadhar",
                "uidai",
                "unique identification",
            ),

            "license": (
                "license",
                "licence",
                "driving",
                "dl no",
            ),

            "pan": (
                "pan",
                "income tax",
                "permanent account",
            ),
        }

        for key, hints in hint_map.items():

            if any(
                hint in img_name
                or hint in full_text
                for hint in hints
            ):

                doc_type = key
                break

    # --------------------------------------------------
    # Unknown document
    # --------------------------------------------------

    if doc_type == "unknown":

        return {
            "document_type": "Unknown",

            "validation_result": "FAIL",

            "checks": {
                "image_quality_acceptable": image_ok,
                "ocr_successful": ocr_ok,
                "number_region_detected": False,
                "expected_digit_count": 0,
                "detected_digit_count": 0,
                "required_fields_present": False,
                "qr_code_detected": qr_detected,
            },
        }

    # --------------------------------------------------
    # Document-specific validation
    # --------------------------------------------------

    config = DOCUMENT_CONFIG[
        doc_type
    ]

    number_check = _find_id_number(
        words,
        config["expected_digits"]
    )

    fields = _detect_required_fields(
        words,
        config["required_fields"]
    )

    number_ok = number_check[
        "found"
    ]

    detected_digits = 0

    if number_check["best"]:

        detected_digits = len(
            number_check["best"]["digits"]
        )

    checks = {

        "image_quality_acceptable":
            image_ok,

        "ocr_successful":
            ocr_ok,

        "number_region_detected":
            number_ok,

        "expected_digit_count":
            config["expected_digits"],

        "detected_digit_count":
            detected_digits,

        "required_fields_present":
            fields["all_present"],

        "qr_code_detected":
            qr_detected,
    }

    result = (
        "PASS"

        if (
            image_ok
            and ocr_ok
            and number_ok
            and fields["all_present"]
        )

        else "FAIL"
    )

    return {
        "document_type":
            config["name"],

        "checks":
            checks,

        "validation_result":
            result,
    }


def print_report(
    report: dict,
    fraud_result: dict = None
) -> str:

    if "checks" not in report:
        return "Invalid Report"

    c = report["checks"]

    final_decision = "FAILED"

    if report[
        "validation_result"
    ] == "PASS":

        if (
            fraud_result
            and fraud_result.get(
                "predicted_risk"
            ) == "LOW"
        ):

            final_decision = "VERIFIED"

        elif (
            fraud_result
            and fraud_result.get(
                "predicted_risk"
            ) == "MEDIUM"
        ):

            final_decision = (
                "MANUAL REVIEW REQUIRED"
            )

        else:

            final_decision = (
                "FRAUD SUSPECTED"
            )

    lines = [

        "=" * 60,

        "DOCUMENT VERIFICATION REPORT",

        "=" * 60,

        f"Detected Type       : "
        f"{report.get('document_type', 'Unknown')}",

        f"OCR Successful      : "
        f"{'YES' if c.get('ocr_successful') else 'NO'}",

        f"Image Quality OK    : "
        f"{'YES' if c.get('image_quality_acceptable') else 'NO'}",

        f"ID Number Detected  : "
        f"{'YES' if c.get('number_region_detected') else 'NO'}",

        f"Expected Digits     : "
        f"{c.get('expected_digit_count', 0)}",

        f"Detected Digits     : "
        f"{c.get('detected_digit_count', 0)}",

        f"Required Fields OK  : "
        f"{'YES' if c.get('required_fields_present') else 'NO'}",

        f"QR Code Detected    : "
        f"{'YES' if c.get('qr_code_detected') else 'NO'}",

        "-" * 60,

        f"Validation Result   : "
        f"{report.get('validation_result', 'FAIL')}",
    ]

    if fraud_result:

        lines.append(
            f"Fraud Risk          : "
            f"{fraud_result.get('predicted_risk', 'N/A')}"
        )

    lines.extend([
        "=" * 60,

        f"FINAL DECISION      : "
        f"{final_decision}",

        "=" * 60,
    ])

    return "\n".join(lines)
