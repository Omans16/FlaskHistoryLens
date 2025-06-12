import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import tflite_runtime.interpreter as tflite
from description import description
from location import location
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Ambil detail input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Daftar label
labels = [
    "Benteng Vredeburg", "Candi Borobudur", "Candi Prambanan", "Gedung Agung Istana Kepresidenan",
    "Masjid Gedhe Kauman", "Monumen Serangan 1 Maret", "Museum Gunungapi Merapi",
    "Situs Ratu Boko", "Taman Sari", "Tugu Yogyakarta"
]

# Fungsi klasifikasi gambar
def classify_image(img: Image.Image):
    try:
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        confidence = np.max(predictions)
        predicted_label = labels[np.argmax(predictions)]

        if confidence < 0.8:
            return {
                "label_output": f"Tidak dapat dikenali (Confidence: {confidence * 100:.2f}%)",
                "deskripsi": (
                    "Tolong arahkan ke objek yang jelas agar bisa diidentifikasikan. "
                    "Pastikan anda berada di salah satu tempat seperti:\n"
                    "- Benteng Vredeburg\n- Candi Borobudur\n- Candi Prambanan\n"
                    "- Gedung Agung Istana Kepresidenan Yogyakarta\n- Masjid Gedhe Kauman\n"
                    "- Monumen Serangan 1 Maret\n- Museum Gunungapi Merapi\n- Situs Ratu Boko\n"
                    "- Taman Sari\n- Tugu Yogyakarta"
                ),
                "lokasi": "-",
                "confidence": float(confidence)
            }
        else:
            lokasi_url = location.get(predicted_label, None)
            return {
                "label_output": f"{predicted_label} (Confidence: {confidence * 100:.2f}%)",
                "deskripsi": description.get(predicted_label, "Deskripsi belum tersedia."),
                "lokasi": lokasi_url if lokasi_url else None,
                "confidence": float(confidence)
            }

    except Exception as e:
        return {"label_output": "Error", "deskripsi": str(e), "lokasi": "-", "confidence": 0.0}


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")

    # Simpan gambar
    filename = f"{uuid.uuid4()}.jpg"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(save_path)

    result = classify_image(img)
    result["image_url"] = f"/static/uploads/{filename}"
    return jsonify(result)

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run server
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)