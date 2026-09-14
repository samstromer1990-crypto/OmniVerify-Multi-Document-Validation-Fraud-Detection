import re


DOCUMENT_CONFIG = {
    "aadhaar": {
        "name": "Aadhaar Card",
        "keywords": [
            "aadhaar",
            "aadhar",
            "uidai",
            "unique identification",
            "unique identification authority",
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
            "transport",
        ],
        "expected_digits": 16,
        "required_fields": ["name", "dob", "validity"],
    },
}


def _normalise_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_document_type(words: list) -> str:
    """
    Determine the document type from OCR text.
    Returns: aadhaar, pan, license, or unknown.
    """

    if not words:
        return "unknown"

    text_blob = " ".join(
        str(w.get("text", "")) for w in words
    )

    text_blob = _normalise_text(text_blob)

    scores = {
        "aadhaar": 0,
        "pan": 0,
        "license": 0,
    }

    for doc_id, config in DOCUMENT_CONFIG.items():

        for keyword in config["keywords"]:

            keyword_normalised = _normalise_text(keyword)

            if keyword_normalised in text_blob:
                scores[doc_id] += max(
                    1,
                    len(keyword_normalised.split())
                )

    best_type = max(scores, key=scores.get)

    if scores[best_type] == 0:
        return "unknown"

    return best_type
