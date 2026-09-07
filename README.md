# OmniVerify: AI-Driven Framework for Multi-Document Validation & Fraud Detection

## 📌 Project Overview
OmniVerify is an intelligent document analysis system designed to automate the process of identity document validation and forensic fraud detection. Unlike simple OCR tools, this system implements a three-tiered analysis approach: **Validation**, **Verification**, and **Fraud Risk Assessment**.

The system is designed to be document-agnostic, supporting various ID types (Aadhaar, PAN, Driving Licence) through a flexible, configuration-driven architecture.

## 🛠️ Core Capabilities

### 1. Document Validation (Structural)
Checks if the uploaded document meets the technical and structural requirements of its type.
- **Document Type Detection:** Automatically identifies the ID type using OCR keyword analysis.
- **Field Extraction:** Extracts critical fields (Name, DOB, ID Number) using Tesseract OCR.
- **Format Validation:** Verifies that ID numbers match expected lengths and patterns (e.g., 12 digits for Aadhaar).

### 2. Document Verification (Consistency)
Performs internal consistency checks to ensure the document is not logically flawed.
- **Cross-Field Analysis:** Compares extracted text against expected formats.
- **QR Analysis:** Detects and decodes QR codes to check for presence and readability.

### 3. Fraud Risk Detection (Forensic ML)
Analyzes the image for visual signatures of digital tampering.
- **Feature Engineering:** Extracts 12 forensic features, including Edge Density, Laplacian Variance, Noise Estimation, and OCR Confidence.
- **Machine Learning:** Uses a **RandomForest Classifier** trained on a synthetic dataset of genuine and manipulated documents to predict a risk score (**LOW, MEDIUM, HIGH**).
- **Explainable AI (XAI):** Integrates **SHAP (SHapley Additive exPlanations)** to provide a transparent reason for every fraud prediction, highlighting which features pushed the risk score up or down.

---

## ⚙️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **Computer Vision** | OpenCV |
| **OCR Engine** | Tesseract OCR (pytesseract) |
| **Machine Learning** | Scikit-Learn (RandomForest) |
| **Explainability** | SHAP |
| **User Interface** | Streamlit |
| **Deployment** | Docker & Render |

---

## 📁 Project Structure
```text
document_verification_system/
│
├── app.py                  # Main Streamlit Application
├── Dockerfile              # Containerization for Cloud Deployment
├── requirements.txt       # Project Dependencies
│
├── src/                    # Core Logic Modules
│   ├── preprocessing.py    # Image loading & Quality analysis
│   ├── ocr.py              # Tesseract OCR wrapper
│   ├── qr_detector.py      # QR code analysis
│   ├── document_types.py   # ID configuration & Dispatcher
│   ├── validator.py        # Validation & Reporting engine
│   ├── features.py         # Forensic feature extraction
│   ├── fraud_detector.py    # ML Model & SHAP explanations
│   └── dataset_generator.py# Synthetic data engine
│
├── models/                 # Saved ML Models (.pkl)
├── data/                   # Raw, Processed, and Sample images
├── reports/                # Output visualizations
└── tests/                  # Milestone test scripts
