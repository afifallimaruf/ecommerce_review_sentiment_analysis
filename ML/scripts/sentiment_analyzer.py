"""
Script untuk melakukan sentiment analysis meggunakan algoritma ML
dengan dataset yang sudah diproses dalam format JSON.

Support:
- Traditional ML (SVM, Naive Bayes, Random Forest)
- Ensemble methods

Author: Afif
Date: 2025

"""

from contextlib import contextmanager
import gc
import json
import os
import pickle
import numpy as np
import pandas as pd
import psutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.config import LoadConfig
from log.logging import SetupLogging

from data_loader import DataLoader

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
warnings.filterwarnings('ignore')


logger_init = SetupLogging()
logger = logger_init.set_logger()


@contextmanager
def memory_monitor():
    """Context manager untuk monitoring memori"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    yield
    final_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB")


class SentimentAnalyzer:
    """ Class Sentiment Analyzer """

    def __init__(self):
        """
        Inisialisasi class Sentiment Analyzer

        Args:
            config_path (str): Alamat ke config.json
        """

        self.root_path = Path(__file__).resolve().parent.parent
        self.config_loader = LoadConfig()
        self.config = self.config_loader.load_config()
        self.data_processed_path = Path(self.config['data_paths']['processed']['training'])
        self.output_dir = Path(self.config['data_paths']['output'])
        self.models_path = self.root_path / Path(self.config['ml_model']['models_path'])

        # Models
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.scaler = None
        self.feature_names = None

        # Hasil
        self.results = {}
        self.predictions = {}

        # Konfigurasi model - menggunakan parameter dari config
        self.model_configs = {
            'logistic': {
                'classifier': LogisticRegression(random_state=self.config['processing']['random_state'], class_weight='balanced'),
                'param_grid': self.config['model_params']['logistic']
            },
            'svm': {
                'classifier': LinearSVC(random_state=self.config['processing']['random_state'], class_weight='balanced'),
                'param_grid': self.config['model_params']['svm']
            },
            'naive_bayes': {
                'classifier': MultinomialNB(),
                'param_grid': self.config['model_params']['naive_bayes']
            },
            'random_forest': {
                'classifier': RandomForestClassifier(random_state=self.config['processing']['random_state']),
                'param_grid': self.config['model_params']['random_forest']
            }
        }

        logger.info("Sentiment Analyzer initialized")

    def load_processed_data(self, filename: str = 'train_preprocessed.csv.gz') -> Tuple[pd.DataFrame, bool]:
        """
        Load data yang sudah diproses untuk digunakan oleh model
        
        Returns (Tuple[pd.DataFrame, bool]): DataFrame dan status load (berhasil/tidak)
        """
        logger.info("Loading processed data...")

        try:
            # Alamat data yang sudah diproses
            data_file = self.data_processed_path / filename

            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}")
                return pd.DataFrame(), False
            
            logger.info(f"Loading data from: {data_file}")

            # Load data dengan kompresi gzip jika file berekstensi .gz
            if data_file.suffix == '.gz':
                df = pd.read_csv(data_file, compression='gzip', encoding='utf-8')
            else:
                df = pd.read_csv(data_file, encoding='utf-8')

            logger.info(f"Data loaded successfully: {len(df)} samples")

            # Validasi kolom yang diperlukan
            required_columns = ['labels','tokens']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame(), False
            
            # Info distribusi label
            label_counts = df['labels'].value_counts()
            logger.info(f"Label distribution: {label_counts.to_dict()}")

            return df, True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame(), False

    def load_vectorizer(self) -> bool:
        """
        Load TI-IDF vectorizer yang sudah ada

        Returns:
            bool: Status berhasil tidaknya
        """
        try:
            vectorizer_path = self.models_path / 'tfidf_vectorizer.pkl'

            if not vectorizer_path.exists():
                alternative_paths = [
                    self.models_path / 'vectorizer.pkl',
                    self.models_path / 'count_vectorizer.pkl'
                ]

                for alt_path in alternative_paths:
                    if alt_path.exists():
                        vectorizer_path = alt_path
                        logger.info(f"Founf vectorizer at alternative path: {vectorizer_path}")
                        break

            
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                # Validasi vectorizer yang dimuat
                if not hasattr(self.vectorizer, 'vocabulary_'):
                    logger.error("Loaded vectorizer doesn't have vocabulary_ attribute")
                    return False
                
                if not hasattr(self.vectorizer, 'transform'):
                    logger.error("Loaded vectorizer doesn't have transform method")
                    return False
                
                vocab_size = len(self.vectorizer.vocabulary_)
                if vocab_size == 0:
                    logger.error("LOaded vectorizer has empty vocabulary")
                    return False
                
                logger.info(f"Pre-trained vectorizer loaded successfully from: {vectorizer_path}")
                logger.info(f"Vectorizer type: {type(self.vectorizer).__name__}")
                logger.info(f"Vocabulary size: {vocab_size}")
            
                # Debug: tampilkan beberapa contoh vocabulary
                sample_vocab = list(self.vectorizer.vocabulary_.keys())[:10]
                logger.debug(f"Sample vocabulary: {sample_vocab}")

                return True
            else:
                logger.warning(f"Pre-trained vectorizer not found at: {vectorizer_path}")
                logger.warning("Available files in models directory:")
                if self.models_path.exists():
                    for file in self.models_path.iterdir():
                        logger.warning(f"  - {file.name}")
                return False
            
        except Exception as e:
            logger.error(f"Error loading vectorizer: {e}")
            self.vectorizer = None
            return False
        
    
    def create_vectorizer(self, texts: List[str]) -> bool:
        """
        Buat TF-IDF vectorizer baru

        Args:
            texts (List[str]): List teks untuk training

        Returns:
            bool: Status berhasil tidaknya
        """

        try:
            logger.info("Creating new TF-IDF vectorizer...")

            self.vectorizer = TfidfVectorizer(
                max_features=self.config['processing']['max_features'],
                min_df=self.config['processing']['min_df'],
                max_df=self.config['processing']['max_df'],
                ngram_range=tuple(self.config['processing']['ngram_range']),
                lowercase=False,
                dtype=np.float32)

            self.vectorizer.fit(texts)

            # Save vectorizer
            vectorizer_path = self.models_path / 'tfidf_vectorizer.pkl'
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)

            logger.info(f"Vectorizer created an saved: {vectorizer_path}")
            logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

            return True

        except Exception as e:
            logger.info(f"Error when creating vectorizer: {e}")
            return False

    def prepare_features(self, df: pd.DataFrame, predictions: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features untuk training

        Args:
            df (pd.DataFrame): Input dataframe


        Returns:
            Tuple[np.ndarray, np.ndarray]: Features dan labels
        """ 

        try:
            logger.info("Preparing features...")

            # TF-IDF features
            X_tfidf = self.vectorizer.transform(df['tokens'])            

            # Pastikan tidak ada nilai negatif
            if hasattr(X_tfidf, 'min'):
                min_val = X_tfidf.min()
                if min_val < 0:
                    logger.warning(f"Found negative values: {min_val}")

                    X_text_dense = X_tfidf.toarray()
                    X_text_dense = np.abs(X_text_dense)
                    X_tfidf = X_text_dense

            # Feature tambahan jika tersedia
            if self.config['feature_engineering']['use_feature_engineering']:
                feature_columns = [
                    'word_count', 'char_count', 'exclamation_count',
                    'question_count', 'uppercase_ratio'
                ]

                available_features = [col for col in feature_columns if col in df.columns]
                if available_features:
                    logger.info(f"Using additional features: {available_features}")
                    additional_features = df[available_features].values

                    # Normalisasi
                    if self.scaler is None:
                        self.scaler = MinMaxScaler()
                        additional_features = self.scaler.fit_transform(additional_features)
                    else:
                        additional_features = self.scaler.transform(additional_features)

                    # Gabung TF-IDF dengan features tambahan
                    X = hstack([X_tfidf, additional_features])
                else:
                    logger.info("No additional features available, using TF-IDF only")
                    X = X_tfidf

            else:
                X = X_tfidf

            if predictions:
                return X, None

            y = df['labels']

            y = df['labels'].values
            
            return X, y
        
        except Exception as e:
            logger.info(f"Error when preparing features: {e}")
            raise
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train single model

        Args:
            model_name (str): Nama model
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels


        Returns:
            Dict: Hasil trining
        """

        try:
            logger.info(f"Training {model_name}")

            config = self.model_configs[model_name]

            if self.config['grid_search']:
                grid_search = GridSearchCV(
                    config['classifier'],
                    config['param_grid'],
                    cv=self.config['processing']['cross_validation_folds'],
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                logger.info(f"Best parameters for {model_name}: {best_params}")

            else:
                # Training model tanpa grid search
                best_model = config['classifier']
                best_model.fit(X_train, y_train)
                best_params = {}
            
            # Prediksi
            y_pred_train = best_model.predict(X_train)
            y_pred_val = best_model.predict(X_val)

            # Evaluasi
            train_accuracy = accuracy_score(y_train, y_pred_train)
            val_accuracy = accuracy_score(y_val, y_pred_val)

            # Cross validation score
            cv_scores = cross_val_score(
                best_model, X_train, y_train,
                cv=self.config['processing']['cross_validation_folds'],
                scoring='accuracy'
            )

            # Classification report
            class_report = classification_report(
                y_val, y_pred_val,
                output_dict=True,
                zero_division=0
            )

            # Confusion matrix
            conf_matrix = confusion_matrix(y_val, y_pred_val)

            # Probability predictions untuk ROC AUC (jika binary classification)
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_val)
                if y_pred_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_val, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
            else:
                roc_auc = None

            # Store model
            self.models[model_name] = best_model

            # Results
            results = {
                'model_name': model_name,
                'model': best_model,
                'best_params': best_params,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'roc_auc': roc_auc,
                'predictions': y_pred_val
            }

            logger.info(f"{model_name} - Train Accuracy: {train_accuracy:.4f}")
            logger.info(f"{model_name} - Val Accuracy: {val_accuracy:.4f}")
            logger.info(f"{model_name} - CV Score: {cv_scores.mean():.4f}(+/- {cv_scores.std() * 2:.4f})")

            return results
        
        except Exception as e:
            logger.error(f"Error when training {model_name}: {e}")
            raise

    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train semua model

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels


        Returns:
           Dict: Hasil training semua model             
        """

        results = {}

        for model_name in self.config['ml_model']['models_to_train']:
            if model_name in self.model_configs:
                try:
                    with memory_monitor():
                        result = self.train_single_model(model_name, X_train, y_train, X_val, y_val)
                        results[model_name] = result

                        # Cleanup
                        gc.collect()
                except Exception as e:
                    logger.error(f"failed to train {model_name}: {e}")
            else:
                logger.warning(f"Model {model_name} not found in configurations")
            
        return results

    
    def select_best_model(self, results: Dict) -> str:
        """
        Pilih model terbaik berdasarkan validation accuracy

        Args:
            results (Dict): Hasil taining semua model

        Returns:
            str: Nama model terbaik
        """

        best_model_name = None
        best_score = 0

        for model_name, result in results.items():
            val_accuracy = result['val_accuracy']
            if val_accuracy > best_score:
                best_score = val_accuracy
                best_model_name = model_name

        if best_model_name:
            self.best_model = self.models[best_model_name]
            logger.info(f"Best model: {best_model_name} with validation accuracy: {best_score:.4f}")

        return best_model_name
    
    def save_models(self, results: Dict) -> None:
        """
        Save semua model yang sudah dilatih

        Args:
            results (Dict): Hasil training
        """

        try:
            for model_name, result in results.items():
                model_path = self.models_path /f"{model_name}_model.pkl"

                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)

                logger.info(f"Model {model_name} saved: {model_path}")

            # Save scaler jika ada
            if self.scaler:
                scaler_path = self.models_path / 'scaler.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                
                logger.info(f"Scaler saved: {scaler_path}")

            # Save results
            results_path = self.models_path / 'training_results.json'
            serializable_results = {}

            for model_name, result in results.items():
                serializable_results[model_name] = {
                    'model_name': result['model_name'],
                    'best_params': result['best_params'],
                    'train_accuracy': float(result['train_accuracy']),
                    'val_accuracy': float(result['val_accuracy']),
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std']),
                    'roc_auc': float(result['roc_auc']) if result['roc_auc'] else None,
                    'classification_report': result['classification_report']
                }

                with open(results_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)

                logger.info(f"Results saved: {results_path}")

        except Exception as e:
            logger.error(f"Error when saving models: {e}")
            raise

    
    def create_visualizations(self, results: Dict, y_val: np.ndarray) -> None:
        """
        Buat visualisasi hasil training

        Args:
            results (Dict): Hasil training
            y_val (np.ndarray): Validation labels
        """

        try:
            # Perbandingan performance
            model_names = list(results.keys())
            val_accuracies = [results[name]['val_accuracy'] for name in model_names]
            cv_means = [results[name]['cv_mean'] for name in model_names]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Perbaningan validation accuracy
            ax1.bar(model_names, val_accuracies, color='skyblue', alpha=0.7)
            ax1.set_title("Model Validation Accuracy Comparison")
            ax1.set_ylabel("Accuracy")
            ax1.set_ylim([0, 1])
            ax1.tick_params(axis='x', rotation=45)

            for i, v in enumerate(val_accuracies):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

            # Cross validation scores
            ax2.bar(model_names, cv_means, color='lightgreen', alpha=0.7)
            ax2.set_title("Cross Validation Score Comparison")
            ax2.set_ylabel("CV Score")
            ax2.set_ylim([0, 1])
            ax2.tick_params(axis='x', rotation=45)

            for i, v in enumerate(cv_means):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Confusion matrix
            n_models = len(results)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, (model_name, result) in enumerate(results.items()):
                if i < 4:
                    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d',
                                cmap='Blues', ax=axes[1])
                    axes[i].set_title(f'{model_name} - Confusion Matrix')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
            
            # Sembunyikan subplots yang tidak digunakan
            for i in range(n_models, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            fig.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Visualizations saved to data/output directory")

        except Exception as e:
            logger.error(f"Error when creating visualizations: {e}")

        
    def predict(self, texts: np.ndarray, model_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediksi sentiment untuk teks baru

        Args:
            texts (List[str]): List teks untuk prediksi
            model_name (str, optional): Nama model

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prediksi dan probabilities
        """      

        try:
            if model_name:
                if model_name not in self.models:
                    raise ValueError(f"Model {model_name} not found")
                model = self.models[model_name]
            else:
                if self.best_model is None:
                    raise ValueError("No best model selected")
                model = self.best_model
            
            # # Transform texts
            # X_tfidf = self.vectorizer.transform(texts)

            # # Additional features jika digunakan
            # if self.config['feature_engineering']['use_feature_engineering'] and self.scaler:
            #     logger.warning("Additional features not implemented for predict prediction")
            #     X = X_tfidf
            # else:
            #     X = X_tfidf

            # Predict
            predictions = model.predict(texts)

            # Dapatkan probabilites jika tersedia
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(texts)
            else:
                probabilities = None
            
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def run_analysis(self, data_file: Optional[str] = None) -> Dict:
        """
        Jalankan full sentiment analysis pipeline

        Args:
            data_file (str, optional): Path ke data file
        
        Returns:
            Dict: Hasil analisis
        """

        try:
            logger.info("Starting sentiment analysis pipeline...")

            # Load data
            df, success = self.load_processed_data()
            if not success:
                raise ValueError("Failed to load data")
            
            # Load atau buat vectorizer
            if not self.load_vectorizer():
                if not self.create_vectorizer(df['tokens'].tolist()):
                    raise ValueError("Failed to create vectorizer")
            # Preapare features
            X, y = self.prepare_features(df)

            # Train-test split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config['processing']['test_size'],
                random_state=self.config['processing']['random_state'],
                stratify=y
            )
            
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Validation set: {X_val.shape[0]} samples")

            # Train models
            with memory_monitor():
                results = self.train_all_models(X_train, y_train, X_val, y_val)

                # Pilih best model
                best_model_name = self.select_best_model(results)

                # Save models
                if self.config['ml_model']['save_models']:
                    self.save_models(results)

                # Buat visualisasi
                self.create_visualizations(results, y_val)

                # Simpan results
                self.results = results

            logger.info("Sentiment analysis pipeline completed successfully!")

            return {
                'results': results,
                'best_model': best_model_name,
                'data_shape': df.shape,
                'feature_shape': X.shape
            }

        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            raise

    
    def load_trained_model(self, model_name: str) -> bool:
        """
        Load model yang sudah dilatih

        Args:
            model_name (str): Nama model

        Returns:
            bool: Status berhasil atau tidaknya
        """

        try:
            model_path = self.models_path / f'{model_name}_model.pkl'

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # logger.info(f"Model: {model}")

            self.models[model_name] = model
            self.best_model = model

            # Load vectorizer dan scaler
            vectorizer_success = self.load_vectorizer()
            if not vectorizer_success:
                logger.error("Failed to load vectorizer. This will affect keyword extraction.")
                return False

            logger.info(f"Vectorizer loaded with vocabulary size: {len(self.vectorizer.vocabulary_)}")

            scaler_path = self.models_path / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

            # Validasi final
            if self.vectorizer is None:
                logger.error("Vectorizer is None after loading. Check vectorizer file.")
            
            logger.info(f"Model {model_name} load successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
        
    
def main():
    """
    Fungsi utama untuk menjalankan sentiment analysis
    """

    try:
        # Inisalisasi
        analyzer = SentimentAnalyzer()

        # Jalankan analysis
        results = analyzer.run_analysis()

        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS RESULTS")
        print("="*50)

        for model_name, result in results['results'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Validation Accuracy: {result['val_accuracy']:.4f}")
            print(f"  Cross Validation: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
            if result['roc_auc']:
                print(f"  ROC AUC: {result['roc_auc']:.4f}")
        
        print(f"\nBest Model: {results['best_model']}")
        print(f"Data Shape: {results['data_shape']}")
        print(f"Feature Shape: {results['feature_shape']}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    main()