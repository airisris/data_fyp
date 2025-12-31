import tensorflow
from tensorflow.keras import models
import streamlit as st
import numpy as np
from PIL import Image

model = tensorflow.keras.models.load_model('pill.keras')

st.title('Pill Classification App')

uploaded_file = st.file_uploader("Upload an image of a pill...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")