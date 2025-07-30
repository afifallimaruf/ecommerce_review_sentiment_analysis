# backend/app.py
import os
from flask import Flask, jsonify, g, current_app
from flask_cors import CORS
from dotenv import load_dotenv
import uuid # Untuk userId anonim

# Muat variabel lingkungan dari .env
load_dotenv()


from backend.config import Config
from backend.routes.sentiment_routes import sentiment_bp
from backend.models import db # Import instance db dari models.py

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        database_url = 'postgresql://afif:fifa@localhost:5432/sentiment_db_default'
        app.logger.warning(f"DATABASE_URL not found in .env. Using default: {database_url}")
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url

    app_id_from_env = os.environ.get('__app_id')
    app.config['APP_ID'] = app_id_from_env if app_id_from_env else 'default-app-id'


    # Inisialisasi SQLAlchemy
    db.init_app(app)

    # Inisialisasi CORS
    CORS(app)

    # Middleware untuk menangani user ID dan database session
    @app.before_request
    def before_request():
        # Dapatkan user ID - tanpa Firebase Auth, kita akan menggunakan UUID acak per sesi
        # atau Anda bisa menggunakan mekanisme session yang lebih canggih jika diperlukan.
        # Untuk demo ini, setiap permintaan akan memiliki 'user_id' yang unik jika tidak ada sesi.
        # Atau, untuk kesederhanaan, kita bisa menggunakan ID anonim tetap.
        # Menggunakan UUID per permintaan untuk membedakan data yang diunggah oleh "pengguna" yang berbeda.
        # Di aplikasi nyata, ini akan diganti dengan ID pengguna dari sistem otentikasi.
        g.user_id = str(uuid.uuid4())
        g.app_id = current_app.config['APP_ID'] # Menggunakan APP_ID dari Config
        app.logger.info(f"Request received for user: {g.user_id}, app: {g.app_id}")

    # Middleware untuk menutup sesi database setelah setiap permintaan
    @app.teardown_request
    def teardown_request(exception):
        db.session.remove()

    # Registrasi Blueprint
    app.register_blueprint(sentiment_bp)

    # Route health check sederhana
    @app.route('/')
    def health_check():
        return jsonify({"message": "E-commerce Review Sentiment Analysis Backend (Flask) is running with PostgreSQL!"}), 200

    # Penanganan error global
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f"Server Error: {error}")
        return jsonify({"error": "Internal server error"}), 500

    return app

if __name__ == '__main__':
    app = create_app()
    
    # Buat tabel database jika belum ada
    with app.app_context():
        db.create_all()
        app.logger.info("Database tables created (if not already existing).")

    app.run(host='0.0.0.0', port=app.config['PORT'])

