import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Set the image size (128x128 pixels) and how many pictures we look at at once (batch size)
img_size = (128, 128)
batch_size = 32

# Where the fruit pictures are stored
data_dir = r'C:\Fruits\Fruit'  
#change to your location path right click on the Fruit folder and copy the path, then paste it here between the quotes

# Load all the fruit pictures and assign labels (e.g. apple = 0, banana = 1)
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=img_size,  # Resize all images to same size
    label_mode='int',     # Labels are just numbers
    shuffle=True,         # Mix up the data
    seed=123              # Make results reproducible
)

X = []  # this will hold the image data
y = []  # this will hold the labels (apple or banana)
for images, labels in dataset:
    X.extend(images.numpy())   # convert images to numpy arrays
    y.extend(labels.numpy())   # convert labels to numpy arrays
X = np.array(X)  # final image array
y = np.array(y)  # final label array

# Pixel values originally go from 0 to 255, we shrink them to between 0 and 1 simplifies trainss faster
X = X / 255.0

# We split the data to train the model, check how it's doing (validation), and test it at the end
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# This adds some random changes to the pictures so the model learns better and doesn't just memorize
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),    # flip image sideways
    layers.RandomRotation(0.1),         # rotate image slightly
    layers.RandomZoom(0.1),             # zoom in a little
    layers.RandomContrast(0.1),         # change brightness/contrast
])

# Buidling hte model 
model = models.Sequential([
    tf.keras.Input(shape=(img_size[0], img_size[1], 3)),  # input is a color image
    data_augmentation,  # apply random changes to images before training
    layers.Conv2D(32, (3,3), activation='relu'),  # detects patterns in small 3x3 areas
    layers.MaxPooling2D(2, 2),                    # keeps only important info
    layers.Conv2D(64, (3,3), activation='relu'),  # deeper pattern recognition
    layers.MaxPooling2D(2, 2),                    # reduce size again
    layers.Flatten(),                             # flatten to 1D to prepare for decision
    layers.Dense(64, activation='relu'),          # hidden layer to learn combinations
    layers.Dense(1, activation='sigmoid')         # final yes/no decision (apple or banana) if another fruit is added, change to Dense(2, activation='softmax')
])

# This sets up how the model learns and how we measure success
model.compile(optimizer='adam',                
              loss='binary_crossentropy',      
              metrics=['accuracy'])            


# Now we let the model look at the training data many times try 10 epochs then try 100 see the difference
# Epoch = how many times the model sees the whole training data
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Ask the model to guess on new (test) pictures it has never seen before
y_pred_probs = model.predict(X_test)  # get probabilities
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # convert probabilities into 0 or 1 (apple or banana) if adding another fruit, change to np.argmax(y_pred_probs, axis=1) for multi-class classification


#Printing the results
print("\nClassification Report:")
# This gives precision, recall, f1-score, and support
print(classification_report(y_test, y_pred, target_names=['Apple', 'Banana']))

print("\nConfusion Matrix:")
# How many apples were mistaken as bananas and vice versa
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nF1 Score:", f1_score(y_test, y_pred, average='binary'))  # Combines accuracy and recall
# Show how the model improved during training (accuracy and loss)

epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], 'b-', label='Train Acc')      # training accuracy
plt.plot(epochs, history.history['val_accuracy'], 'r--', label='Val Acc')   # validation accuracy 
#validation is how well the model does on unseen data, not good for smaller sets so will see a difference between training and validation accuracy
#Validation accuracy will be likely alot higher then what it should be as its we don't have lots of data to validate on
#This is normal, results in overfitting, which is when the model learns the training data too well and doesn't generalize to new data
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# loss is how well the model is doing, lower is better
# Loss graph
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], 'b-', label='Train Loss')         # training loss
plt.plot(epochs, history.history['val_loss'], 'r--', label='Val Loss')      # validation loss
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()  
