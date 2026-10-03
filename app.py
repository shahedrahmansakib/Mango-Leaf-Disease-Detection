import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Clasy AI | Pathology Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Glassmorphism CSS Inject (Matching Next.js Dashboard Template)
st.markdown("""
    <style>
    /* Main Background Override */
    .stApp {
        background: #0d1117 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Native UI Elements */
    header, footer, #MainMenu {visibility: hidden !important;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    
    /* Top Navbar Glass Card */
    .top-nav {
        background: rgba(22, 27, 34, 0.8) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Stat Cards Glassmorphism */
    .glass-card {
        background: rgba(22, 27, 34, 0.7) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
    }
    
    .stat-val {
        font-size: 2rem;
        font-weight: 800;
        color: #f0f6fc;
        margin: 4px 0;
    }
    .stat-title {
        font-size: 0.85rem;
        color: #8b949e;
        font-weight: 500;
    }
    
    /* Neon Green Badges */
    .green-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Result Cards */
    .res-green {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid #10b981 !important;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
    }
    .res-warn {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid #f59e0b !important;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
    }
    
    /* Progress bar green accent */
    .stProgress > div > div > div > div {
        background-color: #10b981 !important;
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
    st.markdown("<h3 style='color:#10b981; margin-bottom:0;'>🌿 AgriVision AI</h3>", unsafe_allow_html=True)
    st.caption("Glassmorphism Admin Suite")
    st.markdown("---")
    
    st.markdown("#### ⚙️ Navigation")
    st.markdown("• 📊 Dashboard Overview")
    st.markdown("• 🔬 Diagnostic Engine")
    st.markdown("• 📄 Model Architecture")
    st.markdown("---")
    
    st.markdown("#### 🎯 Model Specs")
    st.markdown("**Backbone:** Custom CNN")
    st.markdown("**Params:** ~0.45M")
    st.markdown("**Accuracy:** `99.67%`")
    st.markdown("**Classes:** 8 Categories")

# Top Glass Navbar
st.markdown("""
    <div class="top-nav">
        <div style="font-size:1.2rem; font-weight:700; color:#f0f6fc;">Dashboard Overview</div>
        <div><span class="green-badge">● AI Engine Active</span></div>
    </div>
""", unsafe_allow_html=True)

# 4 Stat Cards Row (Exact Layout)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
        <div class="glass-card">
            <div class="stat-title">Total Classes</div>
            <div class="stat-val">8</div>
            <span class="green-badge">Categorized</span>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Model Accuracy</div>
            <div class="stat-val">99.67%</div>
            <span class="green-badge">Validation</span>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Parameters</div>
            <div class="stat-val">0.45M</div>
            <span class="green-badge">Lightweight</span>
        </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Inference Speed</div>
            <div class="stat-val">&lt; 0.2s</div>
            <span class="green-badge">Real-Time</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Grid Area
m1, m2 = st.columns([1, 1], gap="large")

with m1:
    st.markdown("""
        <div class="glass-card">
            <h4 style="margin:0 0 8px 0; color:#10b981;">📷 Specimen Ingestion</h4>
            <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">
                Upload a mango leaf photo to initiate diagnostic analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with m2:
    st.markdown("""
        <div class="glass-card">
            <h4 style="margin:0 0 8px 0; color:#10b981;">🔬 Diagnostic Results</h4>
            <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">
                Real-time classification output and confidence score.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Analyzing Specimen..."):
            predictions = model.predict(img_array)[0]
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)
            
        if predicted_class == 'Healthy':
            st.markdown(f"""
                <div class="res-green">
                    <h3 style="margin:0; color:#10b981;">Diagnosis: {predicted_class}</h3>
                    <p style="margin:4px 0 0 0; color:#f0f6fc;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="res-warn">
                    <h3 style="margin:0; color:#f59e0b;">Detected: {predicted_class}</h3>
                    <p style="margin:4px 0 0 0; color:#f0f6fc;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.markdown("**Confidence Level:**")
        st.progress(confidence / 100.0)
        
        st.markdown("---")
        st.markdown("##### 💡 Agronomic Recommendation:")
        st.info(DISEASE_INFO.get(predicted_class, "Standard maintenance recommended."))
    else:
        st.info("👈 Upload a leaf image from the left panel to execute real-time inference.")
