Analisis Sentimen Ulasan E-commerce
Proyek ini adalah aplikasi web lengkap yang dirancang untuk menganalisis sentimen ulasan produk e-commerce. Pengguna dapat mengunggah file ulasan (CSV atau FastText), dan aplikasi akan memproses ulasan tersebut untuk mengidentifikasi sentimen (positif, negatif, netral), mengekstrak kata kunci, dan menyajikan insight melalui dasbor interaktif.

Fitur
Unggah File Ulasan: Mendukung unggahan file CSV dan FastText (.ft.txt) yang berisi ulasan produk.

Analisis Sentimen: Menggunakan model pembelajaran mesin untuk mengklasifikasikan sentimen setiap ulasan sebagai positif, negatif, atau netral.

Ekstraksi Kata Kunci: Mengidentifikasi kata kunci utama dari setiap ulasan.

Dasbor Interaktif: Menyajikan ringkasan analisis sentimen, distribusi sentimen, tren, perbandingan produk, dan insight kata kunci/kategori.

Penyimpanan Data: Menyimpan hasil analisis (ulasan dan ringkasan) ke database PostgreSQL untuk persistensi.

API Backend: Menyediakan endpoint RESTful untuk mengunggah file, memicu analisis, dan mengambil data dasbor.

Antarmuka Pengguna Frontend: Dibangun dengan React.js untuk pengalaman pengguna yang responsif dan intuitif.

Teknologi yang Digunakan
Frontend:

React.js: Pustaka JavaScript untuk membangun antarmuka pengguna.

Tailwind CSS: Kerangka kerja CSS untuk styling yang cepat dan responsif.

Recharts: Pustaka grafik yang fleksibel untuk visualisasi data.

Lucide React: Koleksi ikon yang ringan dan dapat disesuaikan.

Backend:

Flask: Microframework Python untuk membangun API web.

SQLAlchemy: ORM Python untuk interaksi dengan database.

Psycopg2-binary: Adaptor PostgreSQL untuk Python.

python-dotenv: Untuk mengelola variabel lingkungan.

Flask-CORS: Untuk menangani Cross-Origin Resource Sharing.

Data Processing / Machine Learning:

Python: Bahasa pemrograman utama.

Pandas: Untuk manipulasi dan analisis data.

Scikit-learn: Untuk membangun dan mengelola model pembelajaran mesin (TF-IDF, Logistic Regression).

FastText: Untuk pemrosesan teks dan embedding (jika digunakan).

Tqdm: Untuk menampilkan progress bar saat memproses data.

Database:

PostgreSQL: Sistem database relasional yang kuat untuk menyimpan data analisis.

Struktur Proyek
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
├── .env.example # Contoh file .env untuk variabel lingkungan global
└── README.md

Setup Lokal
Ikuti langkah-langkah ini untuk menyiapkan dan menjalankan proyek di mesin lokal Anda.

1. Klon Repositori
   git clone <URL_REPOSITORI_ANDA>
   cd ecommerce-review-sentiment

2. Siapkan Database PostgreSQL
   Pastikan Anda memiliki server PostgreSQL yang berjalan.

Buat database baru, misalnya sentiment_db.

Buat pengguna dan password jika diperlukan, atau gunakan pengguna default (postgres).

3. Konfigurasi Variabel Lingkungan
   Buat file .env di direktori ecommerce-review-sentiment/backend/ berdasarkan backend/.env.example.

# backend/.env

FLASK_APP=app.py
FLASK_ENV=development
FLASK_RUN_PORT=5000
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<database_name>
SECRET_KEY=your_secret_key_here # Ganti dengan string acak yang kuat

Penting: Ganti <user>, <password>, <host>, <port>, dan <database_name> dengan detail koneksi PostgreSQL Anda yang sebenarnya. Contoh: postgresql://postgres:mysecretpassword@localhost:5432/sentiment_db.

4. Siapkan Lingkungan Python (Backend & Data Processing)
   Disarankan untuk menggunakan virtual environment.

# Navigasi ke direktori backend

cd backend

# Buat dan aktifkan virtual environment

python3 -m venv ecom_backend
source ecom_backend/bin/activate # Linux/macOS

# atau .\ecom_backend\Scripts\activate # Windows

# Instal dependensi backend

pip install -r requirements.txt

# Pindah ke direktori data-processing

cd ../data-processing

# Instal dependensi data-processing ke virtual environment yang sama

pip install -r requirements.txt

# Kembali ke direktori backend untuk menjalankan Flask

cd ../backend

5. Latih Model Sentimen (Opsional, tapi Direkomendasikan)
   Jika Anda belum melatih model sentimen, Anda perlu menjalankannya setidaknya sekali agar script prediksi dapat menemukan model yang dilatih. Pastikan Anda berada di virtual environment yang aktif.

cd ../data-processing/scripts
python dataLoader.py
python sentiment_analyzer.py

Ini akan menghasilkan file model (.pkl) di data-processing/models/.

6. Jalankan Backend Flask
   Pastikan Anda berada di direktori ecommerce-review-sentiment/backend/ dan virtual environment Anda aktif.

flask run

Saat pertama kali dijalankan, Flask akan membuat tabel database di PostgreSQL Anda.

Backend akan berjalan di http://localhost:5000.

7. Jalankan Frontend React
   Buka terminal baru (jangan tutup terminal backend).

# Navigasi ke direktori frontend

cd frontend

# Instal dependensi Node.js

npm install

# Jalankan aplikasi React

npm start

Aplikasi frontend akan terbuka di browser Anda, biasanya di http://localhost:3000.

Penggunaan
Buka aplikasi di browser Anda (http://localhost:3000).

Navigasi ke tab "Review Analyzer".

Unggah file CSV atau FastText (.ft.txt) yang berisi ulasan produk. Pastikan file memiliki kolom teks (misalnya, text, review, comment).

Aplikasi akan memproses file dan menampilkan hasil analisis sentimen di dasbor.

Jelajahi tab "Dashboard", "Product Comparison", dan "Sentiment Insights" untuk melihat visualisasi dan insight dari data ulasan Anda.

Pengujian API (Opsional)
Anda dapat menguji endpoint API backend menggunakan alat seperti Postman atau Thunder Client (ekstensi VS Code).

Endpoint Utama:

GET /: Health check.

POST /api/upload-reviews: Unggah file ulasan (gunakan form-data dengan key reviewsFile dan tipe File).

GET /api/dashboard-stats: Mengambil statistik ringkasan dasbor.

GET /api/sentiment-trends: Mengambil data tren sentimen.

GET /api/product-comparison: Mengambil data perbandingan produk.

GET /api/sentiment-insights: Mengambil insight sentimen (frasa positif/negatif, analisis kategori).

GET /api/latest-reviews: Mengambil daftar ulasan terbaru.

Kontribusi
Kontribusi dipersilakan! Silakan buka issue atau kirim pull request.

Lisensi
Proyek ini dilisensikan di bawah Lisensi MIT.
