import json
import os

# Assuming your training data is organized in folders named after the emotions
train_folder = 'D:/Emoji-Prediction/archive/train'

# Get the emotion classes (folder names) in alphabetical order
emotion_classes = sorted(os.listdir(train_folder))

# Create the class map (index to emotion name)
class_map = {i: emotion for i, emotion in enumerate(emotion_classes)}

# Save to a JSON file
with open('class_map.json', 'w') as f:
    json.dump(class_map, f, indent=4)

print("Class map saved to class_map.json")
print(class_map)