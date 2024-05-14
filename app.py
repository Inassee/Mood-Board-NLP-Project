import streamlit as st
from text_analysis import process_text
from image_retrieval import retrieve_images
from utils import log_error

def display_images(images):
    if images:
        st.subheader("Your Mood Board")
        cols = st.columns(len(images))
        for index, image in enumerate(images):
            with cols[index]:
                st.image(image, width=200, use_column_width=True)
                st.text(f"Displayed: {image.filename}")  # Show the filename below the image
    else:
        st.write("No images to display yet.")


def main():
    st.title("NLP Text-to-Image Mood Board Generator")
    user_input = st.text_area("Enter your text here:", height=150, placeholder="Type your mood or thoughts...")
    if user_input:
        with st.spinner('Analyzing your text...'):
            emotion = process_text(user_input)
            images = retrieve_images(emotion)
            display_images(images)

if __name__ == "__main__":
    main()
