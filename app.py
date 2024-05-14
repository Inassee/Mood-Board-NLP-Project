import streamlit as st
from text_analysis import process_text
from image_retrieval import retrieve_images
from utils import log_error
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
from PIL import Image
from image_analysis import get_model

model = get_model()

img_rows, img_cols = 300, 300
model_path = 'image_sentiment_model.h5'

# Custom CSS to style the app
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Montserrat:wght@400;700&display=swap');

    body {
      background-color: #FF5757;
      color: #FFFFFF;
      font-family: 'Roboto', sans-serif;
    }
    .stApp {
      max-width: 800px;
      margin: auto;
      background-color: transparent;
    }
    .stTextInput, .stTextArea {
      background-color: #f0f0f0;
      border-radius: 10px;
      border: none;
      padding: 10px;
      color: black;
      opacity: 0.8;
      transition: all 0.3s ease-in-out;
    }
    .stTextInput:focus, .stTextArea:focus {
      box-shadow: 0 0 10px rgba(0,0,0,0.2);
    }
    .stImage {
      max-width: 100%;
      margin: auto;
      display: block;
      border-radius: 15px;
      box-shadow: 0 0 15px rgba(0,0,0,0.3);
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .stImage:hover {
      transform: scale(1.05);
      box-shadow: 0 0 25px rgba(0,0,0,0.4);
    }
    .logo {
      display: block;
      margin-left: auto;
      margin-right: auto;
      width: 200px;
      border-radius: 50%;
      box-shadow: 0 0 10px rgba(0,0,0,0.2);
    }
    .intro-text, .disclaimer-text {
      text-align: center;
      font-size: 1.1rem;
      color: #FFFFFF;
      margin-bottom: 2rem;
    }
    .intro-text-bold {
      text-align: center;
      font-size: 1.2rem;
      color: #FFFFFF;
      font-weight: bold;
      margin-bottom: 2rem;
    }
    .app-title {
      text-align: center;
      font-size: 1.5rem;
      margin-top: 1rem;
      margin-bottom: 1rem;
      font-family: 'Montserrat', sans-serif;
    }
    .main-title {
      text-align: center;
      font-size: 2rem;
      margin-top: 1rem;
      font-family: 'Montserrat', sans-serif;
    }
    .spinner {
      display: flex;
      justify-content: center;
      margin: 2rem 0;
    }
    .submit-button {
      display: block;
      width: 100%;
      padding: 10px;
      border-radius: 10px;
      background-color: #FFFFFF;
      color: #FF5757;
      font-size: 1.2rem;
      font-weight: bold;
      border: none;
      cursor: pointer;
      transition: all 0.3s ease-in-out;
    }
    .submit-button:hover {
      background-color: #FFDADA;
    }
    </style>
    """, unsafe_allow_html=True)

def analyze_image_sentiment(image_path):
    img_gray = load_img(image_path, target_size=(img_rows, img_cols), color_mode="grayscale")
    img_gray = img_to_array(img_gray)
    img_gray = np.expand_dims(img_gray, axis=0)
    img_gray /= 255.0
    prediction = model.predict(img_gray)
    return np.argmax(prediction)

def display_images(image_paths, emotion):
    if image_paths:
        st.subheader(f"Your Mood Board for {emotion.capitalize()} Emotion")
        cols = st.columns(len(image_paths))
        for index, image_path in enumerate(image_paths):
            with cols[index]:
                if emotion == 'positive':
                    img = load_img(image_path, target_size=(img_rows, img_cols))
                else:
                    img = load_img(image_path, target_size=(img_rows, img_cols), color_mode="grayscale")
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                st.image(img_bytes, use_column_width=True)
    else:
        st.write("No images to display yet.")

def main():
    st.image('logo.png', use_column_width=True, output_format="PNG", caption="NLP Text-to-Image Mood Board Generator")
    st.markdown("<div class='main-title'>NLP Text-to-Image Mood Board Generator</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intro-text-bold'>
        Welcome to our app that brings your words to life! Simply enter your text, and we'll create a mood board for you. Our tool analyzes the emotions and themes in your text, then selects images to match.
        <br> 🎨 Try our NLP Text-to-Image Mood Board Generator! 🎨
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area("Enter your text here:", height=150, placeholder="Type your mood or thoughts...")
    if user_input:
        with st.spinner('Analyzing your text...'):
            try:
                emotion = process_text(user_input)
                st.write(f"Detected emotion: {emotion}")
                images = retrieve_images(emotion, sample_size=100, display_size=5)
                display_images(images, emotion)
            except Exception as e:
                log_error(e)
                st.error(f"An error occurred: {str(e)}. Please try again - we're working on fixing it!")

    st.markdown("""
    <div class='disclaimer-text'>
        **Disclaimer:** The dataset used may have inconsistencies, and the retrieved images are sometimes unexpected or unusual.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
