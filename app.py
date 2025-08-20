import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("chest_xray_model_temp.h5", compile=False)

# Title
st.title("Edema Detection from Chest X-ray")

# Input: Image
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])
age = st.number_input("Age", min_value=0)
sex = st.selectbox("Sex", options=["Male", "Female"])
view = st.selectbox("View Position", options=["Frontal", "Lateral"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Process inputs
    age_tensor = tf.convert_to_tensor(float(age), dtype=tf.float32)
    sex_val = 0.0 if sex == "Male" else 1.0
    sex_tensor = tf.convert_to_tensor(sex_val, dtype=tf.float32)

    # Add batch dim
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    age_tensor = tf.expand_dims(age_tensor, axis=0)
    sex_tensor = tf.expand_dims(sex_tensor, axis=0)

    # Prediction
    pred = model.predict((img_tensor, age_tensor, sex_tensor))[0][0]
    result = "Edema Detected" if pred > 0.5 else "No Edema"

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### Prediction: **{result}** (Confidence: {pred:.2f})")
