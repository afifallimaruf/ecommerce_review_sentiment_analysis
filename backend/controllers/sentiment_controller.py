import json
import os
from typing import Any, Counter, Dict
import uuid
from flask import g, jsonify, request, current_app
from werkzeug.utils import secure_filename
from subprocess import Popen, PIPE
from pathlib import Path
from datetime import datetime
from sqlalchemy import desc, func

from backend.models import db, Analysis, Review


# Fungsi untuk menangani respons error
def _error_response(message, status_code=500):
    return jsonify({"error": message}), status_code

# Fungsi untuk mengizinkan ekstensi file yang diunggah
def allowed_file(filename):
    if '.' not in filename:
        return False
    
    file_extensions = filename.rsplit('.', 1)[1].lower()

    return file_extensions in {'csv', 'txt'}


def analyze_uploaded_reviews():
    """
    Menangani unggahan file ulasan dan memicu analisis sentimen
    """
    if 'reviewsFile' not in request.files:
        return _error_response('No file part in the request', 400)
    
    file = request.files['reviewsFile']
    print(f"File: {file.filename}")

    if file.filename == '':
        return _error_response('No selected file', 400)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Simpan file ke directori UPLOAD_FOLDER 
        file_path = Path(current_app.config['UPLOAD_FOLDER']) / filename
        try:
            file.save(file_path)
        except Exception as e:
            current_app.logger.error(f"Failed to save uploaded file: {e}")
            return _error_response(f"Failed to save uploaded file: {e}")
        
        try:
            current_app.logger.info(f"Executing Python script: {current_app.config['PYTHON_SCRIPT_PATH']} with file: {file_path}")
            
            # Panggil script python run_sentiment_prediction.py sebagai subprocess
            process = Popen(
                ['python3', str(current_app.config['PYTHON_SCRIPT_PATH']), str(file_path)],
                stdout=PIPE,
                stderr=PIPE,
                text=True
            )

            stdout, stderr = process.communicate()

            print(f"Return code: {process.returncode}")
            print(f"Stdout: '{stdout}'")
            print(f"Stderr: '{stderr}'")

            # Hapus file setelah processing
            try:
                os.remove(file_path)
            except Exception as e:
                current_app.logger.warning(f"Could not remove uploaded file: {e}")


            if process.returncode != 0:
                current_app.logger.error(f"Python script exited with code {process.returncode}")
                current_app.logger.error(f"Python Error (stderr): {stderr}")
                try:
                    error_json = json.loads(stderr or stdout)
                    error_message = error_json.get('error', 'Sentiment analysis failed due to Python script error.')
                except Exception as e:
                    error_message = f"Sentiment analysis failed. Details: {stderr or stdout}"
                return _error_response(error_message)
            
            try:
                results = json.loads(stdout)
                if results.get('error'):
                    return _error_response(results['error'])
                
                # Simpan hasil ke postgreSQL

                summary_data = results.get('summary', {})
                reviews_data = results.get('reviews', [])

                # Buat entry analysis baru
                new_analysis = Analysis(
                    file_name=filename,
                    timestamp=datetime.now(),
                    total_reviews=summary_data.get('totalReviews', 0),
                    positive_count=summary_data.get('positiveCount', 0),
                    negative_count=summary_data.get('negativeCount', 0),
                    neutral_count=summary_data.get('neutralCount', 0),
                    avg_confidence=summary_data.get('avgConfidence', 0.0),
                    # Data dashboard statis untuk demo, di aplikasi nyata bisa dihitung atau disimpan lebih fleksibel
                    dashboard_stats={
                        "totalReviews": summary_data.get('totalReviews', 0),
                        "positiveReviews": summary_data.get('positiveCount', 0),
                        "negativeReviews": summary_data.get('negativeCount', 0),
                        "neutralReviews": summary_data.get('neutralCount', 0),
                        "averageRating": summary_data.get('avgConfidence', 0.0)
                    },
                    
                )
                db.session.add(new_analysis)
                db.session.flush() # Dapatkan ID analisis sebelum menambahkan ulasan

                # Tambahkan ulasan terkait
                for review_item in reviews_data:
                    print(f"Keywords reviews: {review_item.get('keywords')}")
                    new_review = Review(
                        analysis_id=new_analysis.id,
                        original_text=review_item.get('text', ''),
                        sentiment=review_item.get('sentiment', 'neutral'),
                        confidence=review_item.get('confidence', 0.0),
                        keywords=review_item.get('keywords', []),
                        rating=review_item.get('rating', None)
                    )
                    db.session.add(new_review)
                
                db.session.commit()
                current_app.logger.info(f"Analysis results saved to PostgreSQL, analysis ID: {new_analysis.id}")

                return jsonify({"analysisId": new_analysis.id, **results})

            except json.JSONDecodeError as e:
                current_app.logger.error(f"Failed to parse Python output: {stdout}")
                current_app.logger.error(f"Parse Error: {e}")
                db.session.rollback() # Rollback transaksi jika ada error
                return _error_response('Failed to parse sentiment analysis results from Python script.')
            except Exception as e:
                current_app.logger.error(f"Error saving results to PostgreSQL: {e}")
                db.session.rollback() # Rollback transaksi jika ada error
                return _error_response(f"Error saving results to PostgreSQL: {e}")

        except Exception as e:
            current_app.logger.error(f"Error during sentiment analysis process: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return _error_response(f"An unexpected error occurred during analysis: {e}")
    else:
        return _error_response('Invalid file type. Only CSV or FastText (.ft.txt) files are allowed.', 400)

# --- Fungsi untuk mengambil data dari PostgreSQL ---

def get_latest_analysis_data(data_field: str):
    """
    Mengambil data dari analisis terbaru pengguna dari PostgreSQL.
    """

    # Ambil analisis terbaru untuk pengguna ini
    latest_analysis = Analysis.query.order_by(desc(Analysis.timestamp)).first()
    if latest_analysis:
        # Mengembalikan field data_field (misal 'dashboard_stats')
        # Pastikan data JSON diakses dengan benar
        if data_field == 'dashboard_stats':
            return latest_analysis.dashboard_stats
        elif data_field == 'sentiment_trends':
            return latest_analysis.sentiment_trends
        elif data_field == 'product_comparison':
            return latest_analysis.product_comparison
        elif data_field == 'sentiment_insights':
            if latest_analysis.sentiment_insights:
                return latest_analysis.sentiment_insights
            else:
                return _calculate_dynamic_insights(latest_analysis.reviews)
        elif data_field == 'reviews': # Untuk mengambil daftar ulasan
            return [review.to_dict() for review in latest_analysis.reviews]
    return None # Mengembalikan None jika tidak ada data

def get_dashboard_stats():
    data = get_latest_analysis_data('dashboard_stats')
    if not data:
        # Fallback ke data statis jika tidak ada analisis yang diunggah
        data = {
            "totalReviews": 12543,
            "positiveReviews": 8123,
            "negativeReviews": 3124,
            "neutralReviews": 1296,
            "averageRating": 4.2
        }
    return jsonify(data)

def _calculate_dynamic_insights(reviews: list) -> Dict[str, Any]:
    positive_keywords_counter = Counter()
    negative_keywords_counter = Counter()

    category_keywords_map = {
        "Product Quality": ["quality", "good", "great", "bad", "damaged", "durable", "material", "defective", "disappointing"],
        "Delivery Speed": ["shipping", "send", "arrived", "fast", "slow", "late"],
        "Customer Service": ["service", "customer", "customer support", "help", "friendly", "responsive"],
        "Value for Money": ["price", "cheap", "expensive", "economical", "value", "money"]
    }

    # Inisialisasi untuk kategori
    category_counts = {
        cat: {"positive": 0, "negative": 0, "total": 0}
        for cat in category_keywords_map.keys()
    }

    for review in reviews:
        # Pastikan keywords adalah list
        keywords = review.keywords if isinstance(review.keywords, list) else []
        # Pastikan sentiment lowrcase
        sentiment = review.sentiment.lower()

        if sentiment == 'positive':
            positive_keywords_counter.update(keywords)
        elif sentiment == 'negative':
            negative_keywords_counter.update(keywords)

        # Analisis kategori
        # Gunakan set untuk melacak kategori yang sudah dihitung untuk ulasan ini
        # agar satu ulasan tidak dihitung berkali-kali untuk kategori yang sama
        matched_categories_for_review = set()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for category, trigger_keywords in category_keywords_map.items():
                if keyword_lower in trigger_keywords and category not in matched_categories_for_review:
                    category_counts[category]["total"] += 1
                    if sentiment == 'positive':
                        category_counts[category]["positive"] += 1
                    elif sentiment == 'negative':
                        category_counts[category]["negative"] += 1
                    matched_categories_for_review.add(category)
        
    # Format Most positive/negative
    most_positive_phrases = [{'phrase': k, 'count': v} for k, v in positive_keywords_counter.most_common(5)]
    most_negative_phrases = [{'phrase': k, 'count': v} for k, v in negative_keywords_counter.most_common(5)]

    # Format category analysis
    category_analysis_results = []
    for category, counts in category_counts.items():
        if counts["total"] > 0:
            positive_percent = round((counts["positive"] / counts["total"]) * 100)
            negative_percent = round((counts["negative"] / counts["total"]) * 100)
        else:
            positive_percent = 0
            negative_percent = 0
            
        category_analysis_results.append({
            "category": category,
            "positive": positive_percent,
            "negative": negative_percent
        })

    return {
            "mostPositivePhrases": most_positive_phrases,
            "mostNegativePhrases": most_negative_phrases,
            "categoryAnalysis": category_analysis_results
    }

def get_sentiment_insights():
    data = get_latest_analysis_data('sentiment_insights')
    print(f"Data from get_sentiment_insights: {data}")
    if not data:
        data = {
            "mostPositivePhrases": [
                { "phrase": "amazing quality", "count": 234 },
                { "phrase": "fast delivery", "count": 189 },
                { "phrase": "excellent service", "count": 156 },
                { "phrase": "highly recommend", "count": 145 },
                { "phrase": "love this product", "count": 123 }
            ],
            "mostNegativePhrases": [
                { "phrase": "poor quality", "count": 87 },
                { "phrase": "broke quickly", "count": 76 },
                { "phrase": "waste of money", "count": 65 },
                { "phrase": "terrible experience", "count": 54 },
                { "phrase": "not recommended", "count": 43 }
            ],
            "categoryAnalysis": [
                { "category": "Product Quality", "positive": 78, "negative": 22 },
                { "category": "Delivery Speed", "positive": 85, "negative": 15 },
                { "category": "Customer Service", "positive": 72, "negative": 28 },
                { "category": "Value for Money", "positive": 65, "negative": 35 }
            ]
        }
    return jsonify(data)

# Fungsi untuk mengambil semua ulasan dari analisis terakhir
def get_latest_reviews():
    print("Hi from get latest reviews function")

    latest_analysis = Analysis.query.order_by(desc(Analysis.timestamp)).first()
    print(f"Latest analysis: {latest_analysis}")
    
    if latest_analysis:
        reviews_data = [review.to_dict() for review in latest_analysis.reviews]
        return jsonify(reviews_data)
    return jsonify([])

