# 📈 Analisis Sentimen Ulasan E-commerce

Proyek ini adalah **aplikasi web lengkap** yang dirancang untuk memberikan **insight mendalam** dari ulasan produk e-commerce. Dengan mengunggah file ulasan (CSV atau FastText), Anda dapat dengan mudah mengidentifikasi sentimen (positif, negatif, netral), mengekstrak kata kunci penting, dan memvisualisasikan data melalui **dasbor interaktif yang intuitif**.

![Demo Aplikasi (Contoh: Screenshot Dasbor atau Flow Unggah File)](https://via.placeholder.com/800x450?text=Screenshot+Dasbor+Aplikasi+Anda+Disini)
_Sertakan screenshot atau GIF dari dasbor atau alur unggah file Anda untuk visualisasi yang lebih baik._

---

## ✨ Fitur Unggulan

- **Unggah File Ulasan Fleksibel**: Mendukung unggahan file **CSV** dan **FastText (.ft.txt)** yang berisi ulasan produk Anda.
- **Analisis Sentimen Akurat**: Menggunakan **model pembelajaran mesin** canggih untuk mengklasifikasikan setiap ulasan sebagai **positif, negatif, atau netral**.
- **Ekstraksi Kata Kunci Cerdas**: Secara otomatis mengidentifikasi **kata kunci utama** dari setiap ulasan, membantu Anda memahami topik yang paling sering dibicarakan.
- **Dasbor Interaktif & Komprehensif**: Menampilkan ringkasan analisis sentimen, distribusi sentimen, tren dari waktu ke waktu, perbandingan produk, serta insight kata kunci dan kategori yang dapat difilter.
- **Penyimpanan Data Persisten**: Hasil analisis (ulasan dan ringkasan) disimpan dengan aman di database **PostgreSQL** untuk akses dan pelaporan di masa mendatang.
- **API Backend Robust**: Menyediakan **endpoint RESTful** untuk mengelola unggahan file, memicu analisis, dan mengambil data dasbor.
- **Antarmuka Pengguna Responsif**: Dibangun dengan **React.js** untuk pengalaman pengguna yang mulus dan intuitif di berbagai perangkat.

---

## 🚀 Teknologi yang Digunakan

Proyek ini dibangun dengan kombinasi teknologi modern untuk kinerja dan skalabilitas optimal:

### Frontend

- **React.js**: Pustaka JavaScript terkemuka untuk membangun antarmuka pengguna yang dinamis.
- **Tailwind CSS**: Kerangka kerja CSS utilitas-pertama untuk styling yang cepat dan responsif.
- **Recharts**: Pustaka grafik yang fleksibel untuk visualisasi data yang menawan.
- **Lucide React**: Koleksi ikon yang ringan dan dapat disesuaikan.

### Backend

- **Flask**: Microframework Python yang ringan dan kuat untuk membangun API web.
- **SQLAlchemy**: ORM Python yang fleksibel untuk interaksi database yang efisien.
- **Psycopg2-binary**: Adaptor PostgreSQL untuk Python.
- **python-dotenv**: Untuk pengelolaan variabel lingkungan yang aman.
- **Flask-CORS**: Untuk menangani Cross-Origin Resource Sharing.

### Pemrosesan Data / Machine Learning

- **Dataset**: Dataset untuk melatih model berasal dari [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Python**: Bahasa pemrograman utama untuk logika backend dan ML.
- **Pandas**: Pustaka fundamental untuk manipulasi dan analisis data.
- **Scikit-learn**: Digunakan untuk membangun dan mengelola model pembelajaran mesin (khususnya **TF-IDF** dan **Logistic Regression** untuk klasifikasi sentimen).
- **FastText**: Untuk pemrosesan teks dan pembuatan _embedding_ (jika digunakan dalam model).
- **Tqdm**: Untuk menampilkan progress bar yang informatif selama pemrosesan data.

### Database

- **PostgreSQL**: Sistem database relasional yang powerful dan andal untuk menyimpan semua data analisis.

---

## 📂 Struktur Proyek

ecommerce-review-sentiment/
│
├── frontend/ # Aplikasi React.js
│ ├── public/
│ ├── src/
│ │ ├── App.jsx # Komponen utama aplikasi frontend
│ │ └── index.js
│ ├── package.json
│ └── ...
│
├── backend/ # Aplikasi Flask API
│ ├── app.py # File utama aplikasi Flask
│ ├── config.py # Konfigurasi aplikasi Flask
│ ├── models.py # Definisi model SQLAlchemy (Analysis, Review)
│ ├── controllers/
│ │ └── sentiment_controller.py # Logika bisnis untuk endpoint API
│ ├── routes/
│ │ └── sentiment_routes.py # Definisi rute API
│ ├── .env.example # Contoh file variabel lingkungan
│ ├── requirements.txt # Dependensi Python untuk backend
│ └── uploads/ # Direktori untuk menyimpan file yang diunggah sementara
│
├── data-processing/ # Script pemrosesan data & ML
│ ├── scripts/
│ │ ├── dataLoader.py # Memuat dan membersihkan data
│ │ ├── data_preprocessing.py # Preprocessing teks
│ │ ├── sentiment_analyzer.py # Melatih dan menggunakan model sentimen
│ │ └── run_sentiment_prediction.py # Script utama untuk menjalankan pipeline prediksi
│ ├── models/ # Direktori untuk model ML yang dilatih (.pkl)
│ ├── data/ # Direktori opsional untuk data pelatihan/contoh
│ └── config.json # File konfigurasi untuk script ML
│
└── README.md

---

## 🛠️ Setup Lokal

Ikuti langkah-langkah mudah ini untuk menyiapkan dan menjalankan proyek di mesin lokal Anda.

### 1. Klon Repositori

```bash
git clone <URL_REPOSITORI_ANDA>
cd ecommerce-review-sentiment
```

### 2. Buat Virtual Environment

```bash
# Buat dan aktifkan virtual environment
python3 -m venv <nama virtual environment>
source <nama virtual environment>/bin/activate
```

### 3. Simpan dataset ke dalam folder /ML/data/raw

### 4. Install dependensi untuk ML dan latih model

```bash
cd ../ML
pip install -r requirements.txt

# Latih Model
python3 scripts/batch_processing.py
```

Ini akan menghasilkan file (.csv) pada folder ML/data/processed dan file model (.pkl) di direktori ML/models/.

### 5. Buat file .env

Buat file .env di direktori ecommerce_review_sentiment/backend/

```bash
# Jika belum berada di dalam direktori
cd ../backend

# Isi file .env
FLASK_APP=app.py
FLASK_ENV=development
FLASK_RUN_PORT=5000
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<database_name>
```

Penting: Ganti <user>, <password>, <host>, <port>, dan <database_name> dengan detail koneksi PostgreSQL Anda yang sebenarnya. Contoh: postgresql://postgres:mysecretpassword@localhost:5432/sentiment_db.

### 6. Install dependensi dan jalankan Backend

Pastikan Anda berada di direktori ecommerce_review_sentiment/backend/ dan virtual environment Anda aktif.

```bash
pip install -r requirements.txt

flask run
```

### 7. Install dependensi dan jalankan Frontend

Buka terminal baru (jangan tutup terminal backend Anda).

```bash
# Navigasi ke direktori frontend
cd frontend

# Instal dependensi Node.js
npm install

# Jalankan aplikasi React
npm start
```

Aplikasi frontend akan terbuka di browser Anda, biasanya di http://localhost:3000.

## 💡 Penggunaan

1. Buka aplikasi di browser Anda (http://localhost:3000).

2. Navigasi ke tab "Review Analyzer".

3. Unggah file CSV atau FastText (.ft.txt) yang berisi ulasan produk Anda. Pastikan file memiliki kolom teks yang jelas (misalnya, text).

4. Aplikasi akan memproses file dan menampilkan hasil analisis sentimen di dasbor.

5. Jelajahi tab "Dashboard", dan "Sentiment Insights" untuk melihat visualisasi dan insight dari data ulasan Anda.
