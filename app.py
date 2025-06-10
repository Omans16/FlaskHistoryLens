import gradio as gr
import numpy as np
from keras.preprocessing import image
from Model_Load import load_model_from_files
from description import description
from location import location
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# Load model dan label
model = load_model_from_files("model.json", "my_model.h5")

labels = [
    "Benteng Vredeburg", "Candi Borobudur", "Candi Prambanan", "Gedung Agung Istana Kepresidenan",
    "Masjid Gedhe Kauman", "Monumen Serangan 1 Maret", "Museum Gunungapi Merapi",
    "Situs Ratu Boko", "Taman Sari", "Tugu Yogyakarta"
]

# Fungsi preprocessing dan prediksi
def classify_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)[0]
    confidence = np.max(pred)
    predicted_label = labels[np.argmax(pred)]

    akurasi = float(confidence)
    if confidence < 0.8:
        label_output = "Tidak dapat dikenali (Confidence: {:.2f}%)".format(confidence * 100)
        deskripsi = (
            "Tolong arahkan ke objek yang jelas agar bisa diidentifikasikan. "
            "Pastikan anda berada di salah satu tempat seperti:\n"
            "- Benteng Vredeburg\n- Candi Borobudur\n- Candi Prambanan\n"
            "- Gedung Agung Istana Kepresidenan Yogyakarta\n- Masjid Gedhe Kauman\n"
            "- Monumen Serangan 1 Maret\n- Museum Gunungapi Merapi\n- Situs Ratu Boko\n"
            "- Taman Sari\n- Tugu Yogyakarta"
        )
        lokasi = "-"
    else:
        label_output = f"{predicted_label} (Confidence: {confidence * 100:.2f}%)"
        deskripsi = description.get(predicted_label, "Deskripsi belum tersedia.")
        lokasi = location.get(predicted_label, None)
        if lokasi:
            lokasi = f'<a href="{lokasi}" target="_blank">Lihat Lokasi di Google Maps</a>'
        else:
            lokasi = "Lokasi tidak ditemukan"

    return label_output, deskripsi, lokasi, akurasi


# FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # atau sesuaikan dengan asal frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    label_output, deskripsi, lokasi, akurasi = classify_image(img)
    return JSONResponse(content={
        "label_output": label_output,
        "deskripsi": deskripsi,
        "lokasi" : lokasi,
        "confidence": akurasi
    })


# Gradio antarmuka (opsional tetap ditampilkan)
gradio_app = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Gambar"),
    outputs=[
        gr.Textbox(label="Output Klasifikasi"),
        gr.Textbox(label="Deskripsi Lengkap", lines=20, max_lines=50),
        gr.HTML(label="Link Lokasi"),
    ],
    allow_flagging="never",
    title="Klasifikasi Gambar",
    description="Upload gambar, sistem akan mengklasifikasikan dan memberikan deskripsi mengenai gambar tersebut."
)

# Mount Gradio ke FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="")

# Jalankan app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

