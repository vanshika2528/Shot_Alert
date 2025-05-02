import os
import librosa
import librosa.display
import scipy.signal
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path
import uuid  # To generate unique file names

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter

# Define paths
MODEL_PATH = Path("D:/STUDY/6th_sem/SGP-III/ShotAlert/backend/resnet18_model.onnx")
TEMP_DIR = Path("D:/STUDY/6th_sem/SGP-III/ShotAlert/backend/temp")

# Create temp directory if it doesn't exist
TEMP_DIR.mkdir(exist_ok=True)

# Load ONNX model
try:
    ort_session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    print("✅ ONNX model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")
    ort_session = None

# Convert audio file to Mel spectrogram
def generate_spectrogram(audio_path, spectrogram_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        f, t, Sxx = scipy.signal.spectrogram(y, sr)
        
        plt.figure(figsize=(2, 2), dpi=100)
        plt.axis('off')
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        raise

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Gun Sound Detection API. Use the /predict endpoint to make POST requests with an audio file."}), 200

@app.route("/predict", methods=["POST"])
def predict():
    print("🔄 Prediction is ongoing...")  # Log message to indicate prediction is in progress

    if ort_session is None:
        return jsonify({"error": "ONNX model not initialized"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["file"]
    if not audio_file.filename.endswith(('.wav', '.mp3')):
        return jsonify({"error": "Invalid file type. Only .wav and .mp3 files are supported."}), 400

    unique_id = str(uuid.uuid4())[:8]
    temp_audio_path = TEMP_DIR / f"temp_audio_{unique_id}.wav"
    temp_spectrogram_path = TEMP_DIR / f"temp_spectrogram_{unique_id}.png"

    try:
        audio_file.save(temp_audio_path)
        generate_spectrogram(temp_audio_path, temp_spectrogram_path)

        # Load spectrogram as numpy array
        img = Image.open(temp_spectrogram_path).convert('RGB')
        img = img.resize((224, 224))  # Resize to model input size
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
        img_array = np.transpose(img_array, (2, 0, 1))  # Convert to (C, H, W)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform ONNX inference
        inputs = {ort_session.get_inputs()[0].name: img_array}
        outputs = ort_session.run(None, inputs)
        pred_idx = np.argmax(outputs[0])
        confidence = np.max(outputs[0]) * 100

        # Map prediction index to label
        class_labels = {0: "No Gunshot", 1: "Gunshot"}
        prediction_label = class_labels.get(pred_idx, "Unknown")

        # Clean up temp files
        os.remove(temp_audio_path)
        os.remove(temp_spectrogram_path)

        print("✅ Prediction completed successfully.")  # Log success message

        return jsonify({
            "prediction": prediction_label,
            "confidence": f"{confidence:.2f}%"
        }), 200

    except Exception as e:
        print(f"❌ Prediction error: {e}")  # Log error message
        if temp_audio_path.exists():
            os.remove(temp_audio_path)
        if temp_spectrogram_path.exists():
            os.remove(temp_spectrogram_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)