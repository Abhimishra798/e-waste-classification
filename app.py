import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the model
model_path = os.path.join("model", "e-waste-model-10class.keras")
model = tf.keras.models.load_model(model_path)

# Class labels (should match your dataset classes)
class_names = [
    'battery', 'keyboard', 'microwave', 'mobile', 'mouse',
    'pcb', 'player', 'printer', 'television', 'washing machine'
]

# Streamlit UI
st.title("E-Waste Image Classification")
st.write("Upload an image of an electronic device and let the model classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🔍 Predicted Class: `{predicted_class}`")
