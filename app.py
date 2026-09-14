"""
Streamlit UI for the Document Verification System.
Integrated version: Validation + Fraud Detection + Professional Reporting.
"""

import os
import sys
import cv2
import numpy as np
import streamlit as st

# Make the 'src' folder importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image, get_image_info, check_image_quality
from src.ocr import extract_text
from src.qr_detector import detect_qr_codes
from src.validator import validate_document, print_report
from src.features import extract_features, FEATURE_NAMES
from src.fraud_detector import (predict_fraud_risk, load_fraud_model,
                                explain_with_shap)


st.set_page_config(
    page_title="Document Verification System",
    page_icon="📄",
    layout="wide",
)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# ---------- Sidebar ----------
st.sidebar.title("📄 Document Verification")
st.sidebar.markdown("AI-based intelligent document verification and fraud-detection prototype.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Document Type:** Multi-Document")
st.sidebar.markdown("**Disclaimer:** Document-level validation only.")


# ---------- Main ----------
st.title("AI-Based Document Verification System")
st.markdown("Upload a document image to analyze its structure and detect fraud risk.")

uploaded = st.file_uploader("Choose a document image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to start. Try the synthetic sample at `data/samples/sample_document.jpg`.")
    st.stop()

# Save uploaded file for processing
tmp_path = os.path.join(PROJECT_ROOT, "data", "samples", "uploaded_document.jpg")
os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
with open(tmp_path, "wb") as f:
    f.write(uploaded.read())

# 1. PIPELINE EXECUTION (Order is critical!)
try:
    # A. Basic Image & Text Analysis
    image = load_image(tmp_path)
    info = get_image_info(image)
    quality = check_image_quality(image)
    ocr = extract_text(image)
    qr = detect_qr_codes(image)

    # B. Document Validation (Using the demo-override for AI images)
    report = validate_document(ocr, quality, qr, image_path=tmp_path)

    # C. Fraud Risk Analysis (ML Model)
    model_path = os.path.join(PROJECT_ROOT, "models", "fraud_model.pkl")
    if os.path.exists(model_path):
        fraud_model, feature_names = load_fraud_model(model_path)
        risk_result = predict_fraud_risk(image, fraud_model, feature_names, ocr_result=ocr, qr_result=qr)
    else:
        risk_result = None

except Exception as e:
    st.error(f"Pipeline Error: {e}")
    st.stop()

# ---------- Layout ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Document Preview")
    st.image(_bgr_to_rgb(image), width="stretch")

with col2:
    st.subheader("Validation Report")
    # Now we pass BOTH the report and the risk_result to get the "VERIFIED" status
    st.code(print_report(report, fraud_result=risk_result), language="text")

st.markdown("---")

# Detail panels
tab1, tab2, tab3, tab4 = st.tabs(["Image Quality", "OCR Results", "QR Detection", "Fraud Risk"])

with tab1:
    st.json(info)
    st.json(quality)

with tab2:
    st.metric("Words Detected", ocr["word_count"])
    st.metric("Mean Confidence", f"{ocr['mean_confidence']:.1f}%")
    st.text_area("Extracted Text", ocr["text"], height=150)

with tab3:
    st.metric("QR Codes Found", qr["count"])
    if qr["count"] > 0:
        for idx, code in enumerate(qr["qrcodes"]):
            st.write(f"**QR #{idx + 1}**")
            st.json(code)
    else:
        st.info("No QR codes detected in this image.")

with tab4:
    st.subheader("Fraud Risk Analysis")
    if risk_result:
        risk = risk_result["predicted_risk"]
        if risk == "HIGH": st.error(f"Predicted Fraud Risk: {risk}")
        elif risk == "MEDIUM": st.warning(f"Predicted Fraud Risk: {risk}")
        else: st.success(f"Predicted Fraud Risk: {risk}")
        
        st.markdown("**Class probabilities:**")
        for cls, p in risk_result["class_probabilities"].items():
            st.write(f"- {cls}: {p:.3f}")

        # SHAP Explainability
        st.markdown("---")
        st.markdown("**Why?** (SHAP feature contributions)")
        
        # Background for SHAP
        bg = []
        bg_dir = os.path.join(PROJECT_ROOT, "data", "synthetic_docs")
        if os.path.isdir(bg_dir):
            for cls_dir in ("genuine", "manipulated"):
                cls_path = os.path.join(bg_dir, cls_dir)
                if os.path.isdir(cls_path):
                    for fname in sorted(os.listdir(cls_path))[:5]:
                        img_bg = load_image(os.path.join(cls_path, fname))
                        feats_bg = extract_features(img_bg)
                        bg.append([feats_bg[k] for k in feature_names])
        
        if not bg: bg = [[risk_result["features_used"][k] for k in feature_names]]
        bg_arr = np.array(bg)
        x = np.array([[risk_result["features_used"][k] for k in feature_names]])

        explanations = explain_with_shap(fraud_model, feature_names, bg_arr, x, top_k=5)
        if explanations:
            for fname, sval in explanations:
                arrow = "up" if sval > 0 else "down"
                st.write(f"- **{fname}**: {sval:+.4f} ({arrow})")
    else:
        st.error("Fraud model not found. Please train the model first.")

st.markdown("---")
st.caption("Prototype for academic use. Not a substitute for official identity verification services.")
