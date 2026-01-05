import tensorflow
from tensorflow.keras import models
import streamlit as st
import numpy as np
from PIL import Image
import os

model = tensorflow.keras.models.load_model('pill.keras')
class_names = ['Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc', 'Decolgen', 'Fish Oil', 'Kremil S', 
               'Medicol', 'Neozep']

st.title('Pill Classification App💊')

uploaded_file = st.file_uploader('Upload an image of a pill...', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    # file_name, ext = os.path.splitext(uploaded_file)
    # png_file = file_name + '.png'
    # image = im.save(png_file, "png")
 
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image')

    resized = image.resize((150, 150))
    input_arr = np.array(resized)
    rescaled = input_arr / 255
    final_img = np.expand_dims(rescaled, axis=0)
        
    prediction = model.predict(final_img)
    class_index = np.argmax(prediction[0]) # ai-generated
    confidence = prediction[0][class_index] * 100
    st.success(f"Predicted Pill: {class_names[class_index]} with {confidence:.2f}% confidence")