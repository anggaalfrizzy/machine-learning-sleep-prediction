# SleepIQ – Prediksi Kualitas Tidur dengan KNN

Aplikasi web berbasis Flask untuk memprediksi kualitas tidur menggunakan algoritma **K-Nearest Neighbors (KNN)**.

## Tugas 9 – Kecerdasan Buatan
**Judul:** Implementasi Algoritma K-Nearest Neighbors (KNN) Untuk Prediksi Kualitas Tidur Berbasis Web Menggunakan Flask  
**Dosen:** Mohammad Bayu Anggara, S.Kom., M.Kom.

## Hasil Model
- K optimal: **13** (dipilih dengan Elbow Method + Cross Validation)
- Akurasi Test: **96%**
- CV Accuracy (5-fold): **95.17% ± 2.66%**

## Instalasi & Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Train model (generate model.pkl dan grafik)
python train_model.py

# 3. Jalankan aplikasi
python app.py

# Buka browser: http://localhost:5000
```

## Struktur Proyek
```
sleep_knn_app/
├── app.py                          # Flask backend
├── train_model.py                  # Script training KNN
├── model.pkl                       # Model hasil training
├── Sleep_health_and_lifestyle_dataset.csv
├── requirements.txt
├── Procfile                        # Untuk deployment Render/Railway
├── static/
│   ├── elbow_plot.png
│   ├── confusion_matrix.png
│   └── distribution.png
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    ├── dashboard.html
    └── about.html
```

## Deployment ke Render
1. Push ke GitHub
2. Buka [render.com](https://render.com) → New Web Service
3. Hubungkan repo GitHub
4. Build Command: `pip install -r requirements.txt && python train_model.py`
5. Start Command: `gunicorn app:app`
