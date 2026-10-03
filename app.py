import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Mango Leaf Disease Detection", page_icon="🌿")
st.title("🌿 Mango Leaf Disease Diagnostic System")
st.write("Upload a mango leaf image to identify pathology conditions using Deep Learning.")

@st.cache_resource
def load_my_model():
    # Ensure this matches your trained model filename in the repository
    return tf.keras.models.load_model("cnn_model.h5")

model = load_my_model()

CLASS_NAMES = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Analyzing Image..."):
        predictions = model.predict(img_array)[0]
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
