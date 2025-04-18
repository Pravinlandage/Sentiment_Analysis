import streamlit as st
from PIL import Image
import cv2
import numpy as np
import deepface.DeepFace as DeepFace
import textblob
from cv2 import dnn_superres
def analyze_sentiment_image(image):
    
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "Error in analysis"

def analyze_sentiment_text(text):
    blob = textblob.TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def detect_age_gender(image):
    try:
        result = DeepFace.analyze(image, actions=['age', 'gender'], enforce_detection=False)
        return result[0]['age'], result[0]['dominant_gender']
    except:
        return "Error in detection"

def main():
    st.set_page_config(page_title="AI Sentiment & Analysis", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Image Sentiment", "Text Sentiment", "Age & Gender Detection"])
    
    st.markdown("""
        <style>
        .stApp { background-color: #f5f5f5; }
        .css-1d391kg { padding: 20px; }
        </style>
    """, unsafe_allow_html=True)
    
    if page == "Image Sentiment":
        st.title("Image Sentiment Analysis")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            sentiment = analyze_sentiment_image(np.array(image))
            st.write(f"Sentiment: **{sentiment}**")
    
    elif page == "Text Sentiment":
        st.title("Text Sentiment Analysis")
        user_text = st.text_area("Enter text:")
        if st.button("Analyze"):
            sentiment = analyze_sentiment_text(user_text)
            st.write(f"Sentiment: **{sentiment}**")
    
    elif page == "Age & Gender Detection":
        st.title("Age & Gender Detection")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            age,gender = detect_age_gender(np.array(image))
            st.write(f"Age: **{age}**")
            st.write(f"Gender: **{gender}**")

if __name__ == "__main__":
    main()