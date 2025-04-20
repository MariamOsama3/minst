import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model with error handling
try:
    model = load_model('mnist_lstm_model.h5')
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

st.title("MNIST Digit Classifier")
st.write("Upload a 28x28 image of a handwritten digit")

uploaded_file = st.file_uploader("Choose image...", type=["png","jpg","jpeg"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert('L').resize((28,28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)
        
        st.image(img, caption="Your Image", width=200)
        
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        st.success(f"Prediction: {digit} (Confidence: {confidence:.2%})")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")