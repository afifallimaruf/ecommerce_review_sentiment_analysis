# backend/routes/sentiment_routes.py
from flask import Blueprint
from backend.controllers import sentiment_controller

# Buat Blueprint untuk rute sentimen
sentiment_bp = Blueprint('sentiment_bp', __name__, url_prefix='/api')

# Route POST untuk mengunggah dan menganalisis ulasan
sentiment_bp.route('/upload-reviews', methods=['POST'])(sentiment_controller.analyze_uploaded_reviews)

# Route GET untuk data dashboard
sentiment_bp.route('/dashboard-stats', methods=['GET'])(sentiment_controller.get_dashboard_stats)
sentiment_bp.route('/sentiment-insights', methods=['GET'])(sentiment_controller.get_sentiment_insights)
sentiment_bp.route('/latest-reviews', methods=['GET'])(sentiment_controller.get_latest_reviews) # Tambahkan route ini

