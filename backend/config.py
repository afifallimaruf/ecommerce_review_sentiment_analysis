import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Config:
    # Konfigurasi aplikasi Flask
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    PORT = int(os.environ.get('FLASK_RUN_PORT', 5000))

    # Path ke script python run_sentiment_prediction.py
    PYTHON_SCRIPT_PATH = PROJECT_ROOT / 'ML' / 'scripts' / 'run_sentiment_prediction.py'

    # Direktori untuk menyimpan file yang diunggah sementara
    UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
    # Batas ukuran file 16MB
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # Pastikan direktori upload ada
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # Konfigurasi CORS
    CORS_HEADERS = 'Content-Type'
    # Izinkan semua origin untuk pengembangan
    CORS_RESOURCES = {r"/api/*": {"origins": "*"}}

    # Konfigurasi PostgreSQL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    