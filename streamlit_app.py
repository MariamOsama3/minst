import streamlit as st
import numpy as np

from PIL import Image
import tensorflow as tf

from keras.models import load_model  
from tensorflow.keras.models import load_model  # Keep this if using TF < 2.16
# Load the pre-trained LSTM model
model = load_model('mnist_lstm_model.h5')

st.title("MNIST Digit Classifier with LSTM")
st.write("Upload a 28x28 grayscale image of a digit (0-9)")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    img_array = np.array(image)
    
    # Normalize and reshape for LSTM input
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28)  # Reshape to (1, 28, 28)
    
    # Show image
    st.image(image, caption='Uploaded Image', width=200)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    st.write(f"Predicted Digit: {predicted_digit}")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")