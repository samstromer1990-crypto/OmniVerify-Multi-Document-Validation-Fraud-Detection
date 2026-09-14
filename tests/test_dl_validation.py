"""Unit tests for Driving Licence ID / field extraction (no Tesseract)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.validator import _find_id_number, _detect_required_fields, validate_document
from src.document_types import DOCUMENT_CONFIG


def words(*parts):
    return [{"text": p, "confidence": 80, "bbox": None} for p in parts]


def assert_id(tokens, expected):
    result = _find_id_number(words(*tokens), None, doc_type="license")
    assert result["found"], f"ID not found in {tokens}"
    assert result["best"]["digits"] == expected, result


def main():
    assert_id(["Driving", "Licence", "HP-32", "20140003137"], "HP3220140003137")
    assert_id(["Driving", "Licence", "HP.32", "20140003137"], "HP3220140003137")
    assert_id(["HP", ".", "32", "20140003137"], "HP3220140003137")
    assert_id(["UP14", "20240042764"], "UP1420240042764")
    assert_id(["DL", "No:", "1234567890123456"], "1234567890123456")

    # Aadhaar must still require 12 digits and not use the DL regex.
    aadhaar = _find_id_number(
        words("Aadhaar", "1234", "5678", "9012", "Name", "DOB"),
        12,
        doc_type="aadhaar",
    )
    assert aadhaar["found"] and aadhaar["best"]["digits"] == "123456789012"

    hp_fields = _detect_required_fields(
        words(
            "Driving", "Licence", "HP.32", "20140003137",
            "Date", "of", "issue", "Valisty", "07-03-2014", "06-03-2034",
            "Date", "of", "Ben", "12-08-1990",
            "Nwre", "CHUNNI", "LAL", "Famer", "BALE", "RAM",
        ),
        DOCUMENT_CONFIG["license"]["required_fields"],
        doc_type="license",
    )
    assert hp_fields["all_present"], hp_fields

    report = validate_document(
        {
            "words": words(
                "Driving", "Licence", "HP.32", "20140003137",
                "Date", "of", "issue", "07-03-2014", "06-03-2034",
                "Date", "of", "Ben", "12-08-1990",
                "Nwre", "CHUNNI", "LAL",
            ),
            "word_count": 16,
        },
        {"overall_quality": "ISSUES_FOUND"},
        {"count": 0},
    )
    assert report["document_type"] == "Driving Licence"
    assert report["validation_result"] == "PASS", report
    assert report["checks"]["detected_id"] == "HP3220140003137"

    print("DL validation tests passed")


if __name__ == "__main__":
    main()
