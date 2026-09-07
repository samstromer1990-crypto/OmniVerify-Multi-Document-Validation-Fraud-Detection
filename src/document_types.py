DOCUMENT_CONFIG = {
    "aadhaar": {
        "name": "Aadhaar Card",
        "keywords": ["aadhaar", "uidai", "unique identification"],
        "expected_digits": 12,
        "required_fields": ["name", "dob", "gender"],
    },
    "pan": {
        "name": "PAN Card",
        "keywords": ["pan", "income tax", "permanent account"],
        "expected_digits": 10,
        "required_fields": ["name", "father's name", "dob"],
    },
    "license": {
        "name": "Driving Licence",
        "keywords": ["driving licence", "license", "dl"],
        "expected_digits": 16,
        "required_fields": ["name", "dob", "validity"],
    },
}

def detect_document_type(words: list) -> str:
    """
    Analyze OCR words to determine the document type.
    Returns the key (e.g., 'aadhaar') or 'unknown'.
    """
    text_blob = " ".join(w["text"].lower() for w in words)
    
    for doc_id, config in DOCUMENT_CONFIG.items():
        for kw in config["keywords"]:
            if kw in text_blob:
                return doc_id
    return "unknown"
