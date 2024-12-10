from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('rice_cnn_model.h5')  # Ensure the path to your model is correct

# Define the class labels for the rice types
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Route to serve the main page
@app.route('/')
def index():
    return render_template('rice_classification.html')  # Ensure you have this HTML template

# API to classify uploaded images
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file temporarily
    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    # Preprocess the image for model prediction
    image = load_img(file_path, target_size=(128, 128))  # Ensure this matches your model's expected input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image

    # Make a prediction using the model
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]  # Get the label with highest confidence

    # Confidence score
    confidence = float(np.max(prediction))

    # Clean up temporary file after prediction
    os.remove(file_path)

    # Return the prediction and confidence score as a JSON response
    return jsonify({'predicted_class': predicted_class, 'confidence': confidence})

# Ensure 'temp' directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')  # Create temp directory if it doesn't exist

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run Flask on port 5001
