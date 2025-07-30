from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON, text
from datetime import datetime
import json

db = SQLAlchemy()

class Analysis(db.Model):
    """
    Model untuk menyimpan ringkasan hasil analisis sentimen dari setiap unggahan file
    """
    __tablename__ = 'analysis'
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now(), nullable=False)
    
    # Ringkasan hasil analisis
    total_reviews = db.Column(db.Integer, nullable=False)
    positive_count = db.Column(db.Integer, nullable=False)
    negative_count = db.Column(db.Integer, nullable=False)
    neutral_count = db.Column(db.Integer, nullable=False)
    avg_confidence = db.Column(db.Float, nullable=False)

    # Data dashboard lainnya (disimpan sebagai JSON)
    dashboard_stats = db.Column(JSON)
    sentiment_insights = db.Column(JSON)

    # Hubungan dengan Review (satu analisis memiliki banyak ulasan)
    reviews = db.relationship('Review', backref='analysis', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Analysis {self.id} - {self.file_name}>"

    def to_dict(self):
        return {
            'id': self.id,
            'fileName': self.file_name,
            'timestamp': self.timestamp.isoformat(),
            'totalReviews': self.total_reviews,
            'positiveCount': self.positive_count,
            'negativeCount': self.negative_count,
            'neutralCount': self.neutral_count,
            'avgConfidence': self.avg_confidence,
            'dashboard_stats': self.dashboard_stats,
            'sentiment_insights': self.sentiment_insights
        }

class Review(db.Model):
    """
    Model untuk menyimpan detail setiap ulasan yang dianalisis.
    """
    __tablename__ = 'reviews'
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=False)
    
    original_text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False) # 'positive', 'negative', 'neutral'
    confidence = db.Column(db.Float, nullable=False)
    keywords = db.Column(JSON) # List of strings, stored as JSON
    rating = db.Column(db.Integer, nullable=True) # Rating 1-5, bisa null jika tidak tersedia

    def __repr__(self):
        return f"<Review {self.id} - {self.sentiment}>"

    def to_dict(self):
        return {
            'id': self.id,
            'analysisId': self.analysis_id,
            'text': self.original_text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'keywords': self.keywords,
            'rating': self.rating
        }