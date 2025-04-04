from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import librosa
import numpy as np
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load Model
MODEL_PATH = "model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Serve HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Serve Static Files Manually (If needed)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = "uploads/" + file.filename
    file.save(file_path)

    # Extract features
    audio, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec = np.expand_dims(mel_spec, axis=(0, -1))  # Reshape for model
    
    # Get prediction
    prediction = model.predict(mel_spec)
    result = "Real" if prediction[0][0] > 0.5 else "Fake"

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
