import re


DOCUMENT_CONFIG = {
    "aadhaar": {
        "name": "Aadhaar Card",
        "keywords": [
            "aadhaar",
            "aadhar",
            "adhaar",
            "uidai",
            "unique identification",
        ],
        "expected_digits": 12,
        "required_fields": ["name", "dob", "gender"],
    },

    "pan": {
        "name": "PAN Card",
        "keywords": [
            "income tax",
            "permanent account number",
            "permanent account",
            "pan",
        ],
        "expected_digits": 10,
        "required_fields": ["name", "father's name", "dob"],
    },

    "license": {
        "name": "Driving Licence",
        "keywords": [
            "driving licence",
            "driving license",
            "driver licence",
            "driver license",
            "dl no",
            "dlno",
        ],
        "expected_digits": None,
        "required_fields": ["name", "dob", "validity"],
    },
}


def _normalise_text(text: str) -> str:
    text = text.lower()

    text = text.replace("licence", "license")

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def detect_document_type(words: list) -> str:
    """
    Detect the document type from OCR words.

    Returns:
        aadhaar
        pan
        license
        unknown
    """

    if not words:
        return "unknown"

    raw_text = " ".join(
        str(w.get("text", ""))
        for w in words
    )

    text = _normalise_text(raw_text)

    scores = {
        "aadhaar": 0,
        "pan": 0,
        "license": 0,
    }

    # -----------------------------
    # Aadhaar signals
    # -----------------------------

    if re.search(
        r"\b(?:aadhaar|aadhar|adhaar|uidai)\b",
        text
    ):
        scores["aadhaar"] += 10

    if "unique identification" in text:
        scores["aadhaar"] += 8

    # Aadhaar number
    if re.search(
        r"(?<!\d)(?:\d{4}\s*){3}(?!\d)",
        raw_text
    ):
        scores["aadhaar"] += 8

    # -----------------------------
    # PAN signals
    # -----------------------------

    if re.search(
        r"\bpan\b",
        text
    ):
        scores["pan"] += 7

    if (
        "income tax" in text
        or "permanent account" in text
    ):
        scores["pan"] += 10

    if re.search(
        r"\b[a-z]{5}\d{4}[a-z]\b",
        text,
        re.IGNORECASE
    ):
        scores["pan"] += 8

    # -----------------------------
    # Driving Licence signals
    # -----------------------------

    if (
        "driving license" in text
        or "driving licence" in raw_text.lower()
    ):
        scores["license"] += 12

    if re.search(
        r"\bdl\s*(?:no|number)\b",
        text
    ):
        scores["license"] += 8

    if "validity" in text:
        scores["license"] += 3

    # -----------------------------
    # Final decision
    # -----------------------------

    best_type = max(
        scores,
        key=scores.get
    )

    if scores[best_type] == 0:
        return "unknown"

    return best_type