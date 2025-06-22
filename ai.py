import streamlit as st
import easyocr
import numpy as np
import pandas as pd
import cv2
import pickle
import plotly.express as px
import re

# ✅ MUST BE FIRST
st.set_page_config(page_title="Stock Image Predictor", layout="wide")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

import streamlit as st

def run_ai_agent_tab():
    st.subheader("🧠 Running AI Agent from ai.py")
    st.write("This is a placeholder for your AI agent. Replace with real functionality.")
    # You can build a separate assistant/chatbot here if needed


# Correlation matrix
correlation_matrix = {
    "P/E Ratio": {"P/E Ratio": 1.0, "P/B Ratio": 0.0045, "D/E Ratio": -0.0035, "ROE": -0.0178, "ROA": -0.0105},
    "P/B Ratio": {"P/E Ratio": 0.0045, "P/B Ratio": 1.0, "D/E Ratio": 0.8175, "ROE": 0.0281, "ROA": 0.0153},
    "D/E Ratio": {"P/E Ratio": -0.0035, "P/B Ratio": 0.8175, "D/E Ratio": 1.0, "ROE": 0.0678, "ROA": -0.0069},
    "ROE": {"P/E Ratio": -0.0178, "P/B Ratio": 0.0281, "D/E Ratio": 0.0678, "ROE": 1.0, "ROA": 0.0771},
    "ROA": {"P/E Ratio": -0.0105, "P/B Ratio": 0.0153, "D/E Ratio": -0.0069, "ROE": 0.0771, "ROA": 1.0}
}

# Adjust feature function
def adjust_features(features, corr_matrix):
    adjusted = features.copy()
    for col in features.columns:
        adjustment = sum(features[other] * corr_matrix[col].get(other, 0) for other in features.columns)
        adjusted[col] = features[col] + adjustment
    return adjusted

# EasyOCR reader
reader = easyocr.Reader(['en'])

# UI
st.title("📸 Stock Ratio Extraction & Auto Recommendation")
st.write("Upload an image of stock financials (P/E, P/B, D/E, ROE, ROA), and get auto recommendation.")

# Upload image
uploaded_file = st.file_uploader("Upload Stock Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="📷 Uploaded Screenshot", use_column_width=True)

    # OCR
    results = reader.readtext(image, detail=0)
    full_text = " ".join(results)
    st.text_area("📃 OCR Extracted Text", full_text, height=200)

    def extract_ratio(labels, fallback=0.0):
        for label in labels:
            match = re.search(rf"{label}[\s:/]*([\d.]+)", full_text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return fallback

    # Extract
    pe = extract_ratio(["P/E", "Stock P/E", "Stock PE"])
    pb = extract_ratio(["P/B", "Price to book value", "Book Value"])
    de = extract_ratio(["D/E", "Debt to equity"])
    roe = extract_ratio(["ROE", "Return on equity"])
    roa = extract_ratio(["ROA", "Return on assets"])

    # Editable UI
    st.subheader("📝 Edit Financial Ratios (if needed)")
    pe = st.number_input("P/E Ratio", value=pe, format="%.2f")
    pb = st.number_input("P/B Ratio", value=pb, format="%.2f")
    de = st.number_input("D/E Ratio", value=de, format="%.2f")
    roe = st.number_input("ROE", value=roe, format="%.2f")
    roa = st.number_input("ROA", value=roa, format="%.2f")

    # Predict
    features = pd.DataFrame([{
        "P/E Ratio": pe,
        "P/B Ratio": pb,
        "D/E Ratio": de,
        "ROE": roe,
        "ROA": roa
    }])

    adjusted = adjust_features(features, correlation_matrix)
    prediction = model.predict(adjusted)[0]
    confidence = model.predict_proba(adjusted)[0][prediction] * 100

    # Show result
    recommendation = "✅ Buy the Stock!" if prediction == 1 else "❌ Do Not Buy the Stock"
    color = "green" if prediction == 1 else "red"

    st.markdown(f"<h3 style='color:{color}'>{recommendation}</h3>", unsafe_allow_html=True)
    st.write(f"📈 Confidence: **{confidence:.2f}%**")

    # Chart
    chart_df = pd.DataFrame({
        "Metric": ["P/E Ratio", "P/B Ratio", "D/E Ratio", "ROE", "ROA"],
        "Value": [pe, pb, de, roe, roa]
    })
    fig = px.bar(chart_df, x="Metric", y="Value", title="📊 Financial Metrics from Image")
    st.plotly_chart(fig)
