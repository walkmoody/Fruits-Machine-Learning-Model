from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model(r"C:\Fruits\backend\fruit_model.h5")

# Define class names (adjust to your dataset)
CLASS_NAMES = ["Apple", "Banana", "Watermelon"]

@app.route('/')
def home():
    return "Fruit model API is running 🍎🍌🍊"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': round(confidence, 3)
    })

if __name__ == '__main__':
    app.run(debug=True)
