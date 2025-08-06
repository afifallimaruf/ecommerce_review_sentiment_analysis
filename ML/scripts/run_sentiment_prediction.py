import sys
import json
import os
import pandas as pd
from pathlib import Path

from data_loader import DataLoader
from data_preprocessing import TextPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from config.config import LoadConfig
from log.logging import SetupLogging

logger_init = SetupLogging()
logger = logger_init.set_logger()

config_init = LoadConfig()
config = config_init.load_config()

def extract_keywords_from_text(text_tokens: str, sentiment_analyzer_obj: SentimentAnalyzer, top_k: int = 5) -> list:
    """
    Ekstrak keywords berdasarkan TF-IDF

    args:
        text_tokens (str): Text yang sudah di tokenize
        sentiment_analyzer_obj (SentimentAnalyzer): Instance SentimentAnalyzer yang sudah memuat vectorizer
        top_k (int): Jumlah keywords yang diambil
    
    Returns:
        list: Keywords yang paling relevan
    """

    vectorizer = sentiment_analyzer_obj.vectorizer
    logger.info(f"Type of vectorizer in extract_keywords_from_text: {type(vectorizer)}")
    logger.info(f"Vectorizer is None: {vectorizer is None}")

    logger.info(f"Text tokens: {text_tokens}")

    # Validasi vectorizer
    if vectorizer is None:
        logger.error("Vectorizer not loaded. Cannot extract keywords.")
        raise ValueError("Vectorizer must be loaded before keyword extraction")
            
    logger.info(f"vocabulary size: {len(vectorizer.vocabulary_)}")

    try:
        if not text_tokens or not text_tokens.strip():
            logger.warning("Empty text_tokens provided")
            return []
        
        # Transform text ke TF-IDF menggunakan vectorizer yang sudah 
        tfidf_matrix = vectorizer.transform([text_tokens])

        # Dapatkan feature names (vocabulary) dari vectorizer yang sudah dilatih
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            # Fallback menggunakan vocabulary_
            vocab = vectorizer.vocabulary_
            feature_names = [''] * len(vocab)
            for word, idx in vocab.items():
                feature_names[idx] = word
        
        non_zero_indices = tfidf_matrix.nonzero()[1]
        non_zero_values = tfidf_matrix.data

        # Buat dictionary word: score hanya untuk kata yang ada di text
        word_scores = {}
        for idx, score in zip(non_zero_indices, non_zero_values):
            word = feature_names[idx]
            word_scores[word] = score

        if not word_scores:
            logger.warning("No words found with TF-IDF scores > 0")
            return fallback_keyword_extraction(text_tokens, top_k)

        # Urutkan berdasarkan score
        top_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        keywords = [word for word, score in top_keywords]

        logger.debug(f"Extracted keywords with TF-IDF: {keywords}")

        return keywords
    
    except Exception as e:
        logger.warning(f"Error extracting ketwords with TF-IDF: {e}")
        return fallback_keyword_extraction(text_tokens, top_k)
    
def fallback_keyword_extraction(text_tokens: str, top_k: int = 5) -> list:
    """
        Fallback method untuk ekstraksi keywords jika TF-IDF gagal
        
        Args:
            text_tokens (str): Text yang sudah di-tokenize
            top_k (int): Jumlah keywords yang diambil
            
        Returns:
            list: Keywords menggunakan metode sederhana
    """
    try:
        # Stopwords sederhana
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
            'could', 'should', 'can', 'cant', 'dont', 'wont', 'im', 'youre', 
            'its', 'thats', 'get', 'got', 'go', 'went'
        }
            
        tokens = text_tokens.split()
            
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if (len(token) >= 3 and 
                token.lower() not in stopwords and 
                token.isalpha()):
                filtered_tokens.append(token)
            
        # Return top_k tokens atau semua jika kurang dari top_k
        return filtered_tokens[:top_k] if len(filtered_tokens) >= top_k else filtered_tokens
            
    except Exception as e:
        logger.error(f"Error in fallback keyword extraction: {e}")
        return []


def run_prediction_pipeline(input_csv_path: str):
    """
    Menjalankan pipeline prediksi sentimen pada file CSV
    """
    
    try:
        # Inisialisasi komponen-komponen yang diperlukan
        data_loader = DataLoader()
        text_preprocessor = TextPreprocessor()
        sentiment_analyzer = SentimentAnalyzer()

        # Load file raw csv yang diunggah
        try:
            if input_csv_path.endswith('.ft.txt'):
            #     # Untuk file FastText, gunakakan load_dataset dengan absolut path
                raw_df = data_loader.load_dataset(input_csv_path, use_absolute_path=True)
            else:
                # Untuk file csv biasa
                raw_df = pd.read_csv(input_csv_path, encoding='utf-8')

            if 'text' not in raw_df.columns:
                raise ValueError("Input file must contain a 'text' column.")

             # Buat kolom ID sementara untuk pelacakan
            raw_df['original_id'] = raw_df.index 
            logger.info(f"Raw DataFrame loaded. Shape: {raw_df.shape}")
            logger.info(f"Raw DataFrame head:\n{raw_df.head()}")

        except Exception as e:
            logger.error(json.dumps({"error": f"Failed to load or parse input CSV/FastText file: {e}"}))
            sys.exit(1)
        
        if raw_df.empty:
            logger.error(json.dumps({"error": "No data found in the uploaded file."}))
            sys.exit(1)
        
        # Terapkan basic cleaning
        raw_df['cleaned_text'] = raw_df['text'].apply(data_loader.basic_clean_text)
        # Log baris ke-80 setelah basic cleaning untuk debugging
        if 80 in raw_df.index:
            logger.info(f"Raw df (row 80) after basic cleaning: {raw_df.loc[80].to_dict()}")
        else:
            logger.info("Row 80 not found in raw_df after basic cleaning (might be filtered).")

        logger.info(f"After basic cleaning. First 3 cleaned texts: {raw_df['cleaned_text'].head(3).tolist()}")
        logger.info(f"Raw df: {raw_df.iloc[80]}")

        # Terapkan preprocessing lanjutan (dari TextPreprocessor)
        processed_samples = text_preprocessor.preprocess_batch(raw_df, prediction=True)
        logger.info(f"Number of processed samples after TextPreprocessor: {len(processed_samples)}")

        if not processed_samples:
            logger.error(json.dumps({"error": "No valid reviews after preprocessing. Check text length or content"}))
            sys.exit(1)
        
        processed_df = pd.DataFrame(processed_samples)

         # Log baris ke-80 dari processed_df jika ada
        if 80 in processed_df['original_id'].values:
            logger.info(f"Processed DataFrame (original_id 80) head:\n{processed_df[processed_df['original_id'] == 80].iloc[0].to_dict()}")
        else:
            logger.info("original_id 80 not found in processed_df (might be filtered).")

        logger.info(f"Processed DataFrame created. Shape: {processed_df.shape}")
        logger.info(f"Processed DataFrame head:\n{processed_df.iloc[80]}")

        # Load model yang sudah dilatih
        if not sentiment_analyzer.load_trained_model('logistic'):
            logger.error(json.dumps({"error": "Failed to load pre-trained sentiment model. Ensure models are trained and saved."}))
            sys.exit(1)
        
         # Validasi bahwa vectorizer sudah dimuat
        if sentiment_analyzer.vectorizer is None:
            logger.error(json.dumps({"error": "TF-IDF vectorizer not loaded. Cannot proceed with prediction."}))
            sys.exit(1)
        
        logger.info(f"Using TF-IDF vectorizer with vocabulary size: {len(sentiment_analyzer.vectorizer.vocabulary_)}")

        
        # Siapkan fitur untuk digunakan model
        X_predict, _ = sentiment_analyzer.prepare_features(processed_df, predictions=True)
        logger.info(f"Shape of X_predict: {X_predict.shape if X_predict is not None else 'None'}")
        if X_predict is not None and X_predict.shape[0] == 0:
            logger.warning("X_predict is empty after feature preparation. Cannot proceed with prediction.")
            # Consider sys.exit(1) here if this is an unrecoverable error
            print(json.dumps({"summary": {}, "reviews": [], "error": "No features generated for prediction."}), flush=True)
            sys.exit(1)

        # Prediksi
        predictions_labels, probabilities = sentiment_analyzer.predict(X_predict, 'logistic')
        logger.info(f"Predictions labels (first 5): {predictions_labels[:5] if predictions_labels is not None else 'None'}")
        logger.info(f"Probabilities (first 5 rows): {probabilities[:5].tolist() if probabilities is not None else 'None'}")

        if predictions_labels is None or len(predictions_labels) == 0:
            logger.error(json.dumps({"error": "Sentiment prediction returned no labels. Check model or data."}))
            sys.exit(1)
        
        # Pastikan panjang predictions_labels sesuai dengan processed_df
        if len(predictions_labels) != len(processed_df):
            logger.error("Mismatch in lengths of predictions and processed_df. This indicates a processing error.")
            sys.exit(1)

        # Tambahkan hasil prediksi dan probabilitas ke processed_df
        processed_df['sentiment_label'] = predictions_labels
        processed_df['probabilities'] = probabilities.tolist()

        # Gabungkan processed_df dengan raw_df berdasarkan 'original_id'
        # Hanya butuh kolom 'text' dan 'original_id' dari raw_df
        final_df = pd.merge(raw_df[['text', 'original_id']], processed_df, on='original_id', how='inner')
        logger.info(f"Final DataFrame after merge. Shape: {final_df.shape}")
        logger.info(f"Final DataFrame head:\n{final_df.head()}")

        # Format hasil untuk frontend
        results_for_frontend = []

        sentiment_map = {1: "negative", 2: "positive"}

        for original_id, row in final_df.iterrows():
            original_id = row['original_id']
            original_text = row['text']
            processed_tokens_str = row['tokens'] # Ini adalah string token yang sudah diproses
            sentiment = sentiment_map.get(row['sentiment_label'], 'neutral')

            logger.info(f"--- Debugging Row Index: {original_id} ---")
            logger.info(f"Rows: {row}")
            logger.info(f"Original Text from raw_df (via original_id): {original_text}")
            logger.info(f"Cleaned Text from row_data: {row['cleaned_text']}")
            logger.info(f"Tokens from row_data: {row['tokens']}")

            # Hitung confidence
            # Perhatikan bahwa probabilities adalah list dari list (atau array)
            if row['sentiment_label'] == 1: # Negative
                confidence = row['probabilities'][0]
            elif row['sentiment_label'] == 2: # Positive
                confidence = row['probabilities'][1]
            else:
                confidence = max(row['probabilities']) # Ambil probabilitas tertinggi jika netral

            # --- ADD THESE DEBUG LINES ---
            logger.info(f"Review ID: {original_id + 1}")
            logger.info(f"Original Text for ID {original_id+1}: '{raw_df['text'].iloc[original_id]}'")
            logger.info(f"PROCESSED TOKENS (row['tokens']) for ID {original_id+1}: '{row['tokens']}'")
            logger.info(f"Type of row['tokens'] for ID {original_id+1}: {type(row['tokens'])}")
            # --- END DEBUG LINES ---

            keywords = extract_keywords_from_text(processed_tokens_str, sentiment_analyzer, top_k=5)

            processed_tokens_list = processed_tokens_str.split()
            
            validated_keywords = []
            seen_keywords_lower = set()

            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Cek apakah keyword ada di text asli
                if any(part in processed_tokens_list for part in keyword_lower.split()):
                    if keyword_lower not in seen_keywords_lower:
                        validated_keywords.append(keyword)
                        seen_keywords_lower.add(keyword_lower)
                else:
                    logger.debug(f"Keyword '{keyword}' from TF-IDF not found in processed tokens for ID {original_id+1}. Skipping.")

            # Jika tidak ada keywords yang valid, gunakan fallback
            if not validated_keywords:
                logger.warning(f"No valid keywords found for text: {raw_df['text'].iloc[original_id][:50]}...")
                validated_keywords = fallback_keyword_extraction(processed_tokens_str, top_k=5)
                
            # urutkan final keywords
            final_keywords = sorted(list(set(validated_keywords)))
    
            # Debug logging untuk tracking
            logger.debug(f"Text: {original_text[:50]}...")
            logger.debug(f"Tokens: {processed_tokens_str[:50]}...")
            logger.debug(f"Extracted keywords: {keywords}")
            logger.debug(f"Final keywords: {final_keywords}")
            logger.debug("-" * 50)

            results_for_frontend.append({
                "id": original_id + 1,
                "text": original_text,
                "confidence": float(confidence),
                "keywords": final_keywords,
                "sentiment": sentiment,
                "rating": 0
            })

        # Hitung statistik ringkasan
        total_reviews = len(results_for_frontend)
        positive_count = sum(1 for r in results_for_frontend if r['sentiment'] == 'positive')
        negative_count = sum(1 for r in results_for_frontend if r['sentiment'] == 'negative')
        neutral_count = total_reviews - positive_count - negative_count
        avg_confidence = sum(r['confidence'] for r in results_for_frontend) / total_reviews if total_reviews > 0 else 0

        summary = {
            "totalReviews": total_reviews,
            "positiveCount": positive_count,
            "negativeCount": negative_count,
            "neutralCount": neutral_count,
            "avgConfidence": float(avg_confidence)
        }

        result_json = json.dumps({"summary": summary, "reviews": results_for_frontend}, indent=2)
        print(result_json, flush=True)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in prediction pipeline: {error_msg}")
        # Print error to stdout as JSON so Flask can parse it
        error_result = json.dumps({"error": error_msg}, indent=2)
        print(error_result, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file_path = sys.argv[1]
        run_prediction_pipeline(input_file_path)
    else:
        logger.error(json.dumps({"error": "No input file path provided."}))
        sys.exit(1)