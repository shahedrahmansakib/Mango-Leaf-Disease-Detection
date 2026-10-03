import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Mango Leaf Disease Diagnostic System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header-title {
        color: #38bdf8;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Result Metric Boxes */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #38bdf8;
        margin-top: 1rem;
    }

    /* Custom File Uploader Style */
    .stFileUploader {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #475569;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="header-card">
        <div class="header-title">🌿 Mango Leaf Health Diagnostic Center</div>
        <div class="header-subtitle">Automated Pathology Identification & Diagnostic Intelligence Platform using Deep Learning</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Setup
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628324.png", width=80)
    st.title("📌 System Overview")
    st.info("""
    **Model Architecture:** Custom Deep CNN (~0.45M Params)
    \n**Accuracy:** 99.67% Validation Accuracy
    \n**Supported Classes:** 8 Health Categories
    """)
    st.markdown("---")
    st.markdown("### 🏷️ Detectable Diseases:")
    st.markdown("""
    - Anthracnose
    - Bacterial Canker
    - Cutting Weevil
    - Die Back
    - Gall Midge
    - Healthy
    - Powdery Mildew
    - Sooty Mould
    """)
    st.markdown("---")
    st.caption("Developed for Academic & Portfolio Showcase")

# Cache Model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_my_model()

CLASS_NAMES = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

# Disease Information & Management Suggestions
DISEASE_INFO = {
    'Anthracnose': 'Fungal disease causing dark lesions. Apply copper-based fungicides.',
    'Bacterial Canker': 'Bacterial infection causing water-soaked spots. Prune infected parts.',
    'Cutting Weevil': 'Pest infestation causing leaf drops. Use recommended insecticides.',
    'Die Back': 'Fungal drying of twigs from top downwards. Trim dried branches and apply fungicide.',
    'Gall Midge': 'Insect infestation causing gall formation. Use systemic pest control.',
    'Healthy': 'Leaf is healthy and free from noticeable diseases! Maintain regular care.',
    'Powdery Mildew': 'White powdery fungal growth. Apply sulfur-based fungicides.',
    'Sooty Mould': 'Black fungal layer caused by insect secretion. Control honeydew-producing insects.'
}

# Main Grid Layout (2 Columns)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Leaf Image")
    uploaded_file = st.file_uploader("Drop a high-resolution leaf photo here...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col2:
    st.subheader("🔬 Diagnostic Analysis")
    
    if uploaded_file is not None:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Processing neural network inference..."):
            predictions = model.predict(img_array)[0]
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)
            
        # Success Badge or Healthy Indicator
        if predicted_class == 'Healthy':
            st.success(f"### 🎉 Diagnosis: {predicted_class}")
        else:
            st.warning(f"### ⚠️ Detected Pathology: {predicted_class}")
            
        # Confidence Progress Bar
        st.markdown(f"**Confidence Level: {confidence:.2f}%**")
        st.progress(confidence / 100.0)
        
        # Recommendation Card
        st.markdown("---")
        st.markdown("### 💡 Recommended Action / Note:")
        st.info(DISEASE_INFO.get(predicted_class, "No specific recommendations available."))
    else:
        st.info("👈 Please upload a leaf image from the panel on the left to perform real-time analysis.")
