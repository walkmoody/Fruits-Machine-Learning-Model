import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# ---------------------------
# SETTINGS
# ---------------------------
img_size = (128, 128)
batch_size = 32
data_dir = r'C:\Fruits\Fruit'  # path to your dataset

# ---------------------------
# LOAD DATA
# ---------------------------
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    label_mode='int',
    shuffle=True,
    seed=123
)

X, y = [], []
for images, labels in dataset:
    X.extend(images.numpy())
    y.extend(labels.numpy())

X = np.array(X) / 255.0
y = np.array(y)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# ---------------------------
# DATA AUGMENTATION
# ---------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ---------------------------
# MODEL
# ---------------------------
model = models.Sequential([
    tf.keras.Input(shape=(img_size[0], img_size[1], 3)),
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # output 3 probabilities
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# TRAIN
# ---------------------------
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# ---------------------------
# SAVE MODEL
# ---------------------------
model.save("fruit_model.h5")
print("Model saved as fruit_model.h5")

# ---------------------------
# OPTIONAL: PLOT TRAINING HISTORY
# ---------------------------
epochs = range(1, len(history.history['accuracy']) + 1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, history.history['accuracy'], 'b-', label='Train Acc')
plt.plot(epochs, history.history['val_accuracy'], 'r--', label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, history.history['loss'], 'b-', label='Train Loss')
plt.plot(epochs, history.history['val_loss'], 'r--', label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
