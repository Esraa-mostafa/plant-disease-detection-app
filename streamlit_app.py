!pip install streamlit pyngrok
%%writefile app.py
import streamlit as st
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer

# Load model
model = TFSMLayer("/content/exported_model_tomato", call_endpoint="serving_default")

class_names = [
    "Tomato_healthy",
    "Tomato_leaf_curl",
    "Tomato_verticulium_wilt",
    "Tomato_leaf_blight",
    "Tomato_septoria_leaf_spot"
]

st.title("EfficientRefineNet Crop Disease Detector")
st.write("Upload a tomato leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model(img_array)['dense_6'].numpy()
    predicted_index = np.argmax(preds[0])
    predicted_label = class_names[predicted_index]
    confidence = preds[0][predicted_index]

    st.markdown(f"### ✅ Prediction: `{predicted_label}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
    
