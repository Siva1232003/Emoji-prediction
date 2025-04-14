
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # Use all cores

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
                                    Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import tensorflow as tf

# Configure TensorFlow for maximum performance
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
tf.config.optimizer.set_jit(True)

# Constants
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 256  # Larger batch for better CPU utilization
EPOCHS = 50  # Reduced since we'll use LR scheduling

# Data Loading (Optimized)
def load_data(folder_path):
    images, labels = [], []
    classes = sorted(os.listdir(folder_path))
    for label_idx, label in enumerate(classes):
        for img_name in os.listdir(os.path.join(folder_path, label)):
            img = cv2.imread(os.path.join(folder_path, label, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, IMAGE_SIZE))
                labels.append(label_idx)
    return np.array(images).reshape(-1, *IMAGE_SIZE, 1) / 255.0, to_categorical(labels)

# Load data
print("Loading data...")
x_train, y_train = load_data('D:/Emoji-Prediction/archive/train')
x_test, y_test = load_data('D:/Emoji-Prediction/archive/test')

# Lightweight model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMAGE_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='softmax')
    ])
    return model

# Create model
model = create_model()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Optimized data pipeline
train_datagen = ImageDataGenerator(
    rotation_range=15,  # Reduced for faster augmentation
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Training
print("Training...")
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
model.load_weights('best_model.h5')
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Save and plot results (same as before)
model.save('emotion_model_final.keras')