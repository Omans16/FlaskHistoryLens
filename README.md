# HistoryLens - Capstone DBS Coding Camp
Proyek ini adalah aplikasi klasifikasi gambar berbasis deep learning untuk mengenali berbagai situs cagar budaya di Daerah Istimewa Yogyakarta (DIY). Dibuat menggunakan flask dan tflite runtime, serta dirancang agar mudah digunakan langsung melalui pythonanywhere. Project ini merupakan bagian dari Capstone DBS Coding Camp yang menggunakan TensorFlow untuk membangun dan melatih model machine learning.

## Persyaratan Sistem

Sistem ini direkomendasikan untuk dijalankan di:

- OS: Windows 10/11 64-bit
- Python 3.10

---

## Label Kategori
Model ini dapat mengenali 10 lokasi berikut:
- Benteng Vredeburg
- Candi Borobudur
- Candi Prambanan
- Gedung Agung Istana Kepresidenan
- Masjid Gedhe Kauman
- Monumen Serangan 1 Maret
- Museum Gunungapi Merapi
- Situs Ratu Boko
- Taman Sari
- Tugu Yogyakarta

## Arsitektur Model
- MobileNetV2 kustom dengan TensorFlow/Keras
- Input: Gambar RGB berukuran 224x224x3
- Output layer: Softmax (10 kelas)

## Instalasi

Ikuti langkah-langkah berikut untuk mengatur environment dan menjalankan proyek ini di Windows.

### 1. Install Python

- Install Python 3.10 terlebih dahulu

### 2. Buat Enviroment
- buat env terlebih dahulu jika dijalankan di local `python -m venv tf310-env`

- buat env terlebih dahulu `python3.10 -m venv tf310-env`

- akktifkan enc `tf310-env\Scripts\activate`

### 3. Instal Depedensi
- `pip install flask`
- `pip install tflite-runtime`
- `pip install flask-cors`
- `pip install numpy`
- `pip install flask-cors`

## Hosting Model
- pythonanywhere : https://capstoneml.pythonanywhere.com/api/predict karena menggunakan POST dan memberikan respon "Method Not Allowed The method is not allowed for the requested URL"  artinya API model sudah siap digunkan
- huggingface : https://huggingface.co/spaces/CapstoneML/HistoryLens hanya menampilkan load model publik


##  Fitur
- login dan register
- Upload gambar sesuai yang ada di point kategori
- Model akan memprediksi nama lokasi dari gambar tersebut
- Menampilkan gambar unggahan dan hasil klasifikasinya
- Menampilkan Deskripsi terkait gambar yang diupload
- Menampilkan link Google maps 
- Menyimpan History dari detekesi gambar
- Berjalan langsung di browser