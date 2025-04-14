import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Configuration
st.set_page_config(page_title="Emotion Detector", page_icon="😊")

# Load assets
@st.cache_resource
def load_model_and_maps():
    model = load_model('emotion_model_final.keras')
    emoji_map = {
        'angry': '😠', 'fear': '😨', 'happy': '😃',
        'neutral': '😐', 'sad': '😢', 'surprise': '😲'
    }
    return model, emoji_map

def predict_emotion(img_array, model):
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        elif img_array.shape[2] == 3:  # RGB image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:  # Already grayscale
        img_array = img_array
    
    img = cv2.resize(img_array, (48, 48))
    img = np.expand_dims(img/255.0, axis=(0, -1))
    pred = model.predict(img)
    return pred[0]

# UI Components
st.title("🎭 Emotion Detection App")
st.write("Upload a face image to detect emotions")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        model, emoji_map = load_model_and_maps()
        emotion_labels = list(emoji_map.keys())
        img_array = np.array(image)
        
        with st.spinner("Analyzing emotions..."):
            confidence_scores = predict_emotion(img_array, model)
        
        predicted_idx = np.argmax(confidence_scores)
        predicted_emotion = emotion_labels[predicted_idx]
        
        st.success(f"Predicted Emotion: {predicted_emotion.capitalize()} {emoji_map[predicted_emotion]}")
        
        st.subheader("Confidence Levels:")
        for i, (emotion, score) in enumerate(zip(emotion_labels, confidence_scores)):
            st.progress(float(score), text=f"{emotion.capitalize()}: {score*100:.2f}%")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")