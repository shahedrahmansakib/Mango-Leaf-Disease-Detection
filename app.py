import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="AgriVision AI | Mango Pathology",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS with Light Green Accent
st.markdown("""
    <style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Top Glass Card Header */
    .glass-header {
        background: rgba(18, 24, 38, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    
    .status-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }

    .title-text {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 8px;
    }

    /* Glass Cards for Metrics & Containers */
    .glass-card {
        background: rgba(18, 24, 38, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Light Green Highlight Accent for Result Box */
    .result-box-green {
        background: rgba(16, 185, 129, 0.1);
        border-left: 5px solid #10b981;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    .result-box-warning {
        background: rgba(245, 158, 11, 0.1);
        border-left: 5px solid #f59e0b;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    /* Customizing Streamlit Progress Bar Color to Light Green */
    .stProgress > div > div > div > div {
        background-color: #34d399;
    }
    </style>
""", unsafe_allow_html=True)

# Top Glass Header Banner
st.markdown("""
    <div class="glass-header">
        <span class="status-badge">● Engine Online & Ready</span>
        <div class="title-text">🌿 AgriVision AI Diagnostic Platform</div>
        <div class="subtitle-text">Real-time Automated Mango Leaf Pathology Categorization using Deep Neural Networks</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Setup
with st.sidebar:
    st.markdown("### ⚙️ System Metrics")
    st.markdown("""
    - **Architecture:** Custom Deep CNN
    - **Total Params:** ~0.45M (1.75 MB)
    - **Val Accuracy:** `99.67%`
    - **Classes:** 8 Categories
    """)
    st.divider()
    st.markdown("### 🏷️ Target Classes")
    classes_list = [
        "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
        "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
    ]
    for c in classes_list:
        st.markdown(f"- **{c}**")

# Cache Model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_my_model()

CLASS_NAMES = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

DISEASE_ACTION = {
    'Anthracnose': 'Apply copper-based fungicide. Avoid overhead irrigation.',
    'Bacterial Canker': 'Prune affected foliage and apply bactericide spray.',
    'Cutting Weevil': 'Inspect undersides of leaves and apply recommended insecticide.',
    'Die Back': 'Trim infected twigs 2-3 inches below affected portion and seal with fungicide.',
    'Gall Midge': 'Spray systemic insecticides during fresh foliage emergence.',
    'Healthy': 'Plant specimen shows optimal health! Continue standard nutrition & hydration.',
    'Powdery Mildew': 'Apply wettable sulfur or systemic fungicides at first sign.',
    'Sooty Mould': 'Spray mild soapy water or neem oil to clear mold and eliminate honeydew insects.'
}

# Main Layout (2 Columns)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div class="glass-card">
            <h4 style="margin:0 0 10px 0; color:#34d399;">📤 Image Upload</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">Drop a leaf specimen image to analyze</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col2:
    st.markdown("""
        <div class="glass-card">
            <h4 style="margin:0 0 10px 0; color:#34d399;">🔬 Analysis & Inference</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">Model classification & action recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Executing neural network inference..."):
            predictions = model.predict(img_array)[0]
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)
            
        # Result Card Display
        if predicted_class == 'Healthy':
            st.markdown(f"""
                <div class="result-box-green">
                    <h3 style="margin:0; color:#34d399;">Diagnosis: {predicted_class}</h3>
                    <p style="margin:5px 0 0 0; color:#cbd5e1;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box-warning">
                    <h3 style="margin:0; color:#fbbf24;">Detected: {predicted_class}</h3>
                    <p style="margin:5px 0 0 0; color:#cbd5e1;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.markdown(f"**Confidence Meter:**")
        st.progress(confidence / 100.0)
        
        # Actionable Recommendation Card
        st.markdown("---")
        st.markdown("##### 💡 Agronomic Recommendation:")
        st.info(DISEASE_ACTION.get(predicted_class, "Maintain standard plant care."))
    else:
        st.info("👈 Upload a leaf image on the left panel to display diagnostic analysis.")
