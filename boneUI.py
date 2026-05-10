import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="BoneScan:-Bone fracture Detection",
    page_icon="🦴",
    layout="centered"
)

# -----------------------------
# Custom CSS (🔥 makes UI better)
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: white;
    }
    .subtitle {
        text-align: center;
        color: #bbbbbb;
        margin-bottom: 30px;
    }
    .card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = load_model("DLmodel.h5")

# -----------------------------
# Preprocess
# -----------------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Predict
# -----------------------------
def predict_image(uploaded_file):
    img = preprocess_image(uploaded_file)
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return "Not Fractured", pred
    else:
        return "Fractured", 1 - pred

# -----------------------------
# UI Layout
# -----------------------------
st.markdown('<div class="title">🦴 BoneScan:Bone Fracture Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an X-ray to detect fractures</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("🔍 Analyze X-ray"):
        label, confidence = predict_image(uploaded_file)

        # Result Styling
        if label == "Fractured":
            st.error(f"🚨 {label}")
        else:
            st.success(f"✅ {label}")

        # Confidence Bar
        st.write(f"Confidence: {confidence:.4f}")
        st.progress(float(confidence))

    st.markdown('</div>', unsafe_allow_html=True)
