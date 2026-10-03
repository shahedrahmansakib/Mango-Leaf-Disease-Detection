import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Clasy AI | Pathology Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS matching screenshot layout with Light Green Tone
st.markdown("""
    <style>
    /* Dark Background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Hide Default Streamlit Header */
    header {visibility: hidden;}

    /* Top Glass Navbar */
    .top-navbar {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .nav-brand {
        font-size: 1.3rem;
        font-weight: 800;
        color: #10b981;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Glass Stat Cards */
    .stat-card {
        background: rgba(22, 27, 34, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #8b949e;
        font-weight: 500;
    }
    
    .stat-badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid rgba(52, 211, 153, 0.2);
    }

    /* Main Diagnostic Window Card */
    .main-card {
        background: rgba(22, 27, 34, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
    }

    .result-pill-green {
        background: rgba(16, 185, 129, 0.12);
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 15px;
    }

    .result-pill-warning {
        background: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 15px;
    }

    /* Custom Streamlit Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #10b981;
    }
    </style>
""", unsafe_allow_html=True)

# Cache Model
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

# Left Navigation Sidebar
with st.sidebar:
    st.markdown("### 🌿 **AgriVision Admin**")
    st.caption("AI-Powered Pathology Suite")
    st.divider()
    
    st.markdown("#### ⚙️ Navigation")
    st.markdown("- 📊 **Dashboard Overview**")
    st.markdown("- 🔬 **Live Diagnostic Engine**")
    st.markdown("- 📁 **Model Metrics & Specs**")
    st.markdown("- 📄 **System Documentation**")
    st.divider()
    
    st.markdown("#### 🎯 Model Information")
    st.markdown("""
    * **Architecture:** Custom Deep CNN
    * **Params:** ~0.45M (1.75 MB)
    * **Val Accuracy:** `99.67%`
    * **Supported Classes:** 8
    """)
    st.divider()
    st.caption("v2.4 Glassmorphism Release")

# Top Glass Navbar Header
st.markdown("""
    <div class="top-navbar">
        <div class="nav-brand">🌿 Dashboard Overview</div>
        <div style="color: #8b949e; font-size: 0.9rem;">
            System Status: <span class="stat-badge-green">● AI Engine Active</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Top Stat Cards (4 Grid Cards like Screenshot)
s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Total Classes</div>
            <div class="stat-number">8</div>
            <span class="stat-badge-green">Stratified Dataset</span>
        </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Model Accuracy</div>
            <div class="stat-number">99.67%</div>
            <span class="stat-badge-green">Validation Score</span>
        </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Model Parameters</div>
            <div class="stat-number">0.45M</div>
            <span class="stat-badge-green">Lightweight CNN</span>
        </div>
    """, unsafe_allow_html=True)

with s4:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Inference Speed</div>
            <div class="stat-number">&lt; 0.2s</div>
            <span class="stat-badge-green">Real-Time Processing</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Dashboard Workspace (Dual Column Grid)
m1, m2 = st.columns([1, 1], gap="large")

with m1:
    st.markdown("""
        <div class="main-card">
            <h4 style="margin:0 0 8px 0; color:#10b981;">📷 Image Ingestion Window</h4>
            <p style="color:#8b949e; font-size:0.88rem; margin-bottom:15px;">
                Upload a mango leaf photo to initiate deep learning pathology inference.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Target Specimen", use_container_width=True)

with m2:
    st.markdown("""
        <div class="main-card">
            <h4 style="margin:0 0 8px 0; color:#10b981;">🔬 Diagnostic Analytics</h4>
            <p style="color:#8b949e; font-size:0.88rem; margin-bottom:15px;">
                Real-time neural network classification and confidence assessment.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Processing deep network inference..."):
            predictions = model.predict(img_array)[0]
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)
            
        if predicted_class == 'Healthy':
            st.markdown(f"""
                <div class="result-pill-green">
                    <h3 style="margin:0; color:#34d399;">Diagnosis: {predicted_class}</h3>
                    <p style="margin:4px 0 0 0; color:#e6edf3;">Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-pill-warning">
                    <h3 style="margin:0; color:#fbbf24;">Detected Pathology: {predicted_class}</h3>
                    <p style="margin:4px 0 0 0; color:#e6edf3;">Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.markdown("**Confidence Probability:**")
        st.progress(confidence / 100.0)
        
        st.markdown("---")
        st.markdown("##### 💡 Agronomic Treatment & Care Guideline:")
        st.info(DISEASE_INFO.get(predicted_class, "Standard plant maintenance recommended."))
    else:
        st.info("👈 Upload a leaf image from the left ingestion panel to view real-time diagnostic analytics.")
