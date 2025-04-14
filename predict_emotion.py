import cv2
import numpy as np
import sys
import os
from tensorflow.keras.models import load_model

EMOJI_MAP = {
    'angry': '😠',
    'fear': '😨',
    'happy': '😃',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

def predict_emotion(img_path):
    # Verify image exists
    if not os.path.exists(img_path):
        return f"Error: File not found at {img_path}"
    
    try:
        # Load model
        model = load_model('emotion_model_final.keras')
        
        # Read and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Error: Could not read image (might be corrupted or wrong format)"
            
        img = cv2.resize(img, (48, 48))
        img = np.expand_dims(img/255.0, axis=(0, -1))
        
        # Predict
        pred = model.predict(img)
        emotion_idx = np.argmax(pred)
        emotion = list(EMOJI_MAP.keys())[emotion_idx]
        return f"Predicted Emotion: {emotion} {EMOJI_MAP[emotion]}"
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_emotion.py <image_path>")
        print("Example: python predict_emotion.py happy_face.jpg")
        sys.exit(1)
    
    # Handle paths with spaces
    img_path = ' '.join(sys.argv[1:])
    result = predict_emotion(img_path)
    print(result)