import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained MobileNetV2 model
model = tf.keras.models.load_model("model/e-waste-model-MobileNetV2.keras")

# Define the class names
class_names = [
    'Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
    'PCB', 'Player', 'Printer', 'Television', 'Washing Machine'
]

# App UI
st.set_page_config(page_title="E-Waste Image Classifier", layout="centered")
st.title("♻️ E-Waste Image Classification")
st.markdown("Upload an image of an e-waste item to classify it into one of the 10 categories.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If user uploads an image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"#### 🔍 Confidence: `{confidence:.2f}%`")
