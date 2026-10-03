import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="AgriVision AI | Mango Disease Diagnostic",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light Green Theme Styling for Native Elements
st.markdown("""
    <style>
    /* Accent green highlights */
    .stProgress > div > div > div > div {
        background-color: #10b981 !important;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

# Cache Model Loading
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_my_model()

CLASS_NAMES = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

DISEASE_INFO = {
    'Anthracnose': 'Fungal infection causing dark sunken spots. Apply copper fungicides.',
    'Bacterial Canker': 'Water-soaked bacterial lesions. Prune infected parts immediately.',
    'Cutting Weevil': 'Pest cutting young leaf stalks. Apply targeted foliage spray.',
    'Die Back': 'Fungal drying of branches from top down. Trim dried twigs.',
    'Gall Midge': 'Insect galls formed on leaf surface. Use systemic insecticides.',
    'Healthy': 'Optimal foliage condition! Maintain regular hydration & nutrition.',
    'Powdery Mildew': 'White powdery fungal coating. Apply sulfur-based spray.',
    'Sooty Mould': 'Black fungal layer. Clear honey-dew insects using neem spray.'
}

# Sidebar
with st.sidebar:
    st.title("🌿 AgriVision AI")
    st.caption("Mango Leaf Disease Diagnostic System")
    st.divider()
    st.markdown("### 📊 Model Architecture")
    st.write("**Model:** Custom Deep CNN")
    st.write("**Parameters:** ~0.45M")
    st.write("**Validation Accuracy:** 99.67%")
    st.divider()
    st.markdown("### 🏷️ Detectable Classes")
    for name in CLASS_NAMES:
        st.write(f"- {name}")

# Main Title Header
st.title("🌿 Mango Pathology Diagnostic Dashboard")
st.caption("Automated real-time leaf disease classification powered by Deep Learning")

st.divider()

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Total Classes", value="8")
m2.metric(label="Validation Accuracy", value="99.67%")
m3.metric(label="Model Size", value="1.75 MB")
m4.metric(label="Inference Latency", value="< 0.2s")

st.divider()

# Main Workspace
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📷 Specimen Ingestion")
    uploaded_file = st.file_uploader("Upload leaf photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col2:
    st.subheader("🔬 Diagnostic Analysis")
    if uploaded_file is not None:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Executing neural inference..."):
            predictions = model.predict(img_array)[0]
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)
            
        if predicted_class == 'Healthy':
            st.success(f"### 🎉 Diagnosis: {predicted_class}")
        else:
            st.warning(f"### ⚠️ Detected: {predicted_class}")
            
        st.markdown(f"**Confidence Level: {confidence:.2f}%**")
        st.progress(confidence / 100.0)
        
        st.divider()
        st.markdown("#### 💡 Agronomic Recommendation")
        st.info(DISEASE_INFO.get(predicted_class, "Standard maintenance recommended."))
    else:
        st.info("👈 Upload a leaf image from the left panel to run analysis.")
