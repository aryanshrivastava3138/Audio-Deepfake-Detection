from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import librosa
import numpy as np
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# Load Model
MODEL_PATH = "model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        # Extract features
        audio, sr = librosa.load(file_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or truncate to fixed size
        max_pad_len = 128
        if mel_spec_db.shape[1] < max_pad_len:
            pad_width = max_pad_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_pad_len]
        
        # Reshape for model
        mel_spec_db = np.expand_dims(mel_spec_db, axis=(0, -1))  # Shape: (1, height, width, 1)

        # Predict
        prediction = model.predict(mel_spec_db)
        result = "Real" if prediction[0][0] > 0.5 else "Fake"
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
