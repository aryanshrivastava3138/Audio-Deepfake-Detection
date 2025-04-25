# 🎙️ Audio Deepfake Detection

This project aims to detect deepfake audio using machine learning. The model is trained to classify audio files as **Real** or **Fake** based on extracted Mel-Spectrogram features.

## 🚀 Features
- Upload an audio file (`.wav` format).
- The model predicts if the audio is **Real** or **Fake**.
- Uses a Flask-based REST API for real-time inference.

## 🖼️ Screenshots

> Home Page:

![Home Page](images/1.png)

> Prediction Page:

![Click to select Audio File.. Page](images/2.png)

> Output Page:

![Predict Real Page](images/3.png)

![Predict Fack Page](images/4.png)



## 📂 Project Structure


Audio-Deepfake-Detection/ │── uploads/ # Folder for uploaded audio files │── model.keras # Trained deepfake detection model │── app.py # Flask API for inference │── templates/ │ ├── index.html # Frontend UI for file upload │── static/ │ ├── style.css # Styling for UI │── requirements.txt # Required dependencies │── README.md # Project Documentation

## ⚙️ Setup & Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/dasmrpmunna/Audio-Deepfake-Detection.git
   cd Audio-Deepfake-Detection


2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the Flask API
python app.py

5. Access the Web App Open your browser and go to:
http://127.0.0.1:5000


🛠 Technologies Used
Python

Flask

TensorFlow/Keras

Librosa (for audio processing)

HTML/CSS/JavaScript (Frontend)

🏗 Future Improvements
Support for more audio formats (.mp3, .ogg, etc.).

Improve model accuracy with a larger dataset.

Deploy on cloud platforms (AWS/GCP).

🤝 Contributing
Feel free to fork this repository and contribute to the project!

📜 License
This project is open-source and available under the MIT License.


---

### 📌 **Final Step**  
Save the file, and now your `README.md` is ready! You can view it in **VS Code**, **GitHub**, or **Markdown viewers**.

Want me to improve it further or customize it for your needs? 🚀🔥


## 📜 License
This project is licensed under the [MIT License](LICENSE).
