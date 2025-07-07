"""
Data preprocessing lanjutan

Script ini adalah data preprocessing lanjutan untuk data text
seperti text normalization, tokenization, dan feature engineering


Author: Afif
Date: 2025

"""


import csv
import gc
import os
import re
import string
import json
import pickle
import pandas as pd
import numpy as np
import gzip
import nltk
import psutil

from contextlib import contextmanager
from typing import Iterator, List, Dict, Tuple, Optional, Set
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from log.logging import SetupLogging



# Konfigurasi sistem logging.
logger_init = SetupLogging()
logger = logger_init.set_logger()

@contextmanager
def memory_monitor():
    """Context manager untuk monitoring memori"""
    process = psutil.Process(os.getpid())
    inital_memory = process.memory_info().rss / 1024 / 1024 
    yield
    final_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {inital_memory:.1f}MB -> {final_memory:.1f}MB")

# Download data NLTK yang diperlukan
for package in ['punkt_tab', 'stopwords', 'wordnet', 'vader_lexicon', 'omw-1.4']:
    try:
        nltk.download(package, quiet=True)
    except Exception as e:
        logger.error(f"Error when dowloading NLTK: {e}")
        raise


class TextPreprocessor:
    """
    text preprocessor lanjutan untuk sentiment analysis
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Inisialisasi preprocessor

        Args:
            config(Dict, optional): konfigurasi yang bersifat opsional
        """

        self.config = {
            'min_text_length': 3,
            'max_text_length': 200,
            'remove_stopwords': True,
            'apply_lemmatization': True,
            'min_word_freq': 2,
            'chunk_size': 10000,
            'save_frequency': 5,
            'max_features': 10000,
        }

        if config:
            self.config.update(config)


        # Inisialisasi NLTK
        self._lemmatizer = None
        self._sentiment_analyzer = None

        # Load stopwords
        try:
            self.stopwords = frozenset(stopwords.words('english'))
        except:
            self.stopwords = frozenset()

        # Tambahkan stopwords tambahan untuk e-commerce
        aditional_stopwords = {'product', 'item', 'buy', 'bought', 'purchase', 'amazon'}
        self.stopwords = self.stopwords | aditional_stopwords

        # Memetakan kata yang disingkat
        self.constractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "it's": "it is", "that's": "that is"
        }

        # Compile regex patterns untuk performa lebih baik
        self._contraction_pattern = re.compile('|'.join(re.escape(key) for key in self.constractions.keys()))
        self._punctuation_pattern = re.compile(r'[^\w\s]')
        self._digits_pattern = re.compile(r'\d+')
        self._spaces_pattern = re.compile(r'\s+')
        
        # Tidak menyimpan seluruh vocabulary dalam memori
        self.vocabulary = None
        self.output_dir = Path("data/processed/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Counter untuk tracking progress
        self.total_processed = 0
        self.chunk_counter = 0
    @property
    def lemmatizer(self):
        """ loading lemmatizer """
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer
    
    @property
    def sentiment_analyzer(self):
        """ loading sentiment analyzer """
        if self._sentiment_analyzer in None:
            self._sentiment_analyzer = SentimentIntensityAnalyzer()
        return self._sentiment_analyzer

    def _is_gzip_file(self, file_path: str) -> bool:
        """
        Deteksi apakah file dikompres dengan gzip berdasarkan magic number

        Args:
            file_path (str): Alamat file

        Returns:
            bool: False/True
        """

        try:
            with open(file_path, 'rb') as f:
                # Baca 2 byte pertama untuk mengecek magic number gzip
                magic = f.read(2)
                return magic == b'\x1f\x8b'
        except:
            return False
        
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validasi input dataframe

        Args:
            df (pd.DataFrame): Input dataframe

        
        Returns:
            bool: True jika valid
        """

        required_columns = ['labels', 'cleaned_text']

        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Expected: {required_columns}, Got: {list(df.columns)}")
            return False
        
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        # Cek nilai null
        null_counts = df[['labels', 'text', 'cleaned_text']].isnull().sum()

        if null_counts['labels'] > 0:
            logger.warning(f"Found {null_counts['labels']} null text values")
        if null_counts['cleaned_text'] > 0:
            logger.warning(f"Found {null_counts['cleaned_text']} null cleaned_text values")

        return True


    def load_data_generator(self, file_path: str):
        """
        Load data yang sudah pemrosesan dasar di file data_loader

        Args:
            file_path (str): Path/alamat ke file csv yang sudah di proses di awal
            
        Returns:
            pd.DataFrame: data yang sudah di load
        """

        chunk_size = self.config['chunk_size']

        try:
            # Deteksi apakah file dikompres gzip
            if self._is_gzip_file(file_path):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    chunk = []
                    for row in reader:
                        chunk.append(row)
                        if len(chunk) >= chunk_size:
                            df = pd.DataFrame(chunk)
                            yield df
                            chunk = []
                            del df
                            gc.collect()
                    if chunk:
                        yield pd.DataFrame(chunk)
            else:
                dtypes = {
                    'labels': 'int8',
                    'cleaned_text': 'string'
                }

                # Hanya load kolom yang diperlukan
                usecols = ['labels', 'cleaned_text']

                for chunk in pd.read_csv(
                    file_path,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    engine='c',
                    chunksize=chunk_size,
                    low_memory=False,
                    dtype=dtypes
                ):
                    yield chunk
        except Exception as e:
            logger.info(f"Error when load data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Membersihkan dan normalisasi teks

        Args:
            text (str): Input teks

        
        Returns:
            str: Teks yang sudah di bersihkan dan di normalisasi
        """

        # Ubah teks ke dalam bentuk lowercase
        text = text.lower()

        # Ubah kata yang singkat (don't) dari dictionary contractions menjadi bentuk panjangnya(do not)
        text = self._contraction_pattern.sub(lambda m: self.constractions[m.group()], text)
        

        # Hapus tanda baca dan angka
        text = self._punctuation_pattern.sub(' ', text)
        text = self._digits_pattern.sub(' ', text)

        # Hapus ekstra spasi
        text = self._spaces_pattern.sub(' ', text).strip()

        return text
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenisasi dan proses teks

        Args:
            text (str): Input teks


        Returns:
            List[str]: 
        """

        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        processed_tokens = [
            self.lemmatizer.lemmatize(token) if self.config['apply_lemmatization'] else token
            for token in tokens
            if len(token) >= 2 and (not self.config['remove_stopwords'] or token not in self.stopwords)
        ]

        return processed_tokens
    
    def extract_features(self, text: str, tokens:List[str]) -> Dict:
        """
        Ekstrak fitur dasar dari teks

        Args:
            text (str): Input teks
            tokens (List[str]): Daftar token

        """

        text_len = len(text)

        # Hitung karakter khusus sekali saja
        exclamation_count = text.count('!')
        question_count = text.count('?')
        uppercase_count = sum(1 for c in text if c.isupper())

        features = {
            'word_count': len(tokens),
            'char_count': text_len,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'uppercase_ratio': uppercase_count / text_len if text_len > 0 else 0
        }

        # Simplifikasi sentiment analysis - skip untuk menghemat CPU
        features.update({
                'vader_compound': 0, 'vader_positive': 0,
                'vader_negative': 0, 'vader_neutral': 0
        })

        return features
    
    def preprocess_single_text(self, text: str, label: int) -> Optional[Dict]:
        """
        Preprocess satu sample teks

        Args:
            text (str): Input teks
            label (int): Input label


        Returns:
            Optional[Dict]: Output dalam bentuk dictionary
        """

        if not isinstance(text, str) or not text.strip():
            return None

        # Bersihkan teks
        cleaned_text = self.clean_text(text)

        # Cek length
        words = cleaned_text.split()
        word_count = len(words)
        if len(words) < self.config['min_text_length'] or word_count > self.config['max_text_length']:
            return None
        
        # Tokenisasi dan prosesing teks
        tokens = self.tokenize_and_process(cleaned_text)
        if not tokens:
            return None
        
        # Ekstrak features
        features = self.extract_features(cleaned_text, tokens)


        return {
            'cleaned_text': cleaned_text,
            'tokens': ' '.join(tokens),
            'label': label,
            'features': features
        }
    
    def preprocessing_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """
        Preprocess seluruh dataset

        Args:
            df (pd.DataFrame): Input dataset


        Returns:
            List[Dict]: Output dalam bentuk list
        """

        logger.info("Starting dataset preprocessing...")

        # Validasi input
        if not self.validate_dataframe(df):
            raise ValueError("Invalid input dataframe")
        
        # Hapus data duplikat
        df = df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
        # Hapus kolom text
        df = df.drop('text', axis=1)

        # Proses sampel
        processed_samples = []
        for idx, row in df.iterrows():
            try:
                processed_sample = self.preprocess_single_text(row['cleaned_text'], row['labels'])
                if processed_sample:
                    processed_samples.append(processed_sample)

                if (idx + 1) % 2000 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} samples in chunk {self.chunk_counter + 1}")

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue

        logger.info(f"Chunk: {self.chunk_counter + 1} completed: {len(processed_samples)} samples processed")
        
        # Cleanup
        del df
        gc.collect()

        return processed_samples

    def save_processed_chunk(self, processed_samples: List[Dict], chunk_idx: int) -> None:
        """
        Save data yang sudah di proses

        Args:
            processed_generator (Iterator[Dict]): Generator yang menghasilkan processed samples

        """

        if not processed_samples:
            logger.warning(f"No processed samples to save for chunk {chunk_idx}")

        # Save ke file terpisah per chunk
        file_path = self.output_dir / f'train_preprocessed_chunk_{chunk_idx:04d}.json'

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'data': processed_samples,
                'count': len(processed_samples),
                'chunk_idx': chunk_idx,
                'config': self.config
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Chunk {chunk_idx} saved: {file_path} ({len(processed_samples)} samples)")

        # Update counter
        self.total_processed += len(processed_samples)

    def merge_chunks(self) -> None:
        """
        Merge semua chunk files menjadi satu file final
        """

        logger.info("Merging all chunks...")

        # Cari semua chunk files
        chunk_files = sorted(self.output_dir.glob('train_preprocessed_chunk_*.json'))

        if not chunk_files:
            logger.warning("No chunk files found to merge")
            return
        
        # Merge chunks
        all_samples = []
        all_texts = []

        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    samples = chunk_data['data']

                    all_samples.extend(samples)
                    all_texts.extend([sample['tokens'] for sample in samples])

                # Hapus chunk file setelah diproses
                chunk_file.unlink()
            except Exception as e:
                logger.info(f"Error preprocessing chunk file {chunk_file}: {e}")
                continue
            
        # Save final merged file
        final_file = self.output_dir / 'train_preprocessed.json'
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump({
                'data': all_samples,
                'count': len(all_samples),
                'config': self.config
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final merged file saved: {final_file} ({len(all_samples)}) samples")
        # Save TF-IDF vectorizer
        if all_texts:
            vectorizer = TfidfVectorizer(
                max_features=self.config['max_features'],
                min_df=self.config['min_word_freq']
            )

            vectorizer.fit(all_texts)

            vectorizer_path = self.output_dir / 'tfidf_vectorizer.pkl'
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            logger.info(f"TF-IDF vectorizer saved: {vectorizer_path}")
        
        # Cleanuo
        del all_samples
        del all_texts
        gc.collect()


def main():
    """ Fungsi main untuk menjalankan preprocessing """
    config = {
        'min_text_length': 3,
        'max_text_length': 200,
        'remove_stopwords': True,
        'apply_lemmatization': True,
        'chunk_size': 10000,
        'save_frequency': 5,
        'min_word_freq': 2,
        'max_features': 10000
    }

    try:
        
        # Inisialisasi preprocessor
        preprocessor = TextPreprocessor(config)

        # Load file
        file_path = 'data/processed/training/train_processed.csv'

        # Proses data dalam chunks
        all_processed_samples = []

        logger.info("Loading and processing data...")
        with memory_monitor():
            for chunk_idx, chunk in enumerate(preprocessor.load_data_generator(file_path)):
                logger.info(f"Preprocessing chunk {chunk_idx + 1}...")

                # Set chunk counter
                preprocessor.chunk_counter = chunk_idx
                
                # Preprocess chunk
                processed_samples = preprocessor.preprocessing_dataset(chunk)
                
                # Save chunk langsung
                preprocessor.save_processed_chunk(processed_samples, chunk_idx)

                # Cleanup chunk
                del processed_samples
                gc.collect()
        
                # Monitor memori setiap 10 chunk
                if (chunk_idx + 1) % 10 == 0:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Memory usage after chunk {chunk_idx + 1}: {memory_mb:.1f}MB")

            
        # Merge semua chunks
        preprocessor.merge_chunks()


        logger.info("PREPROCESSING COMPLETED!")
        logger.info(f"Total samples processed: {preprocessor.total_processed}")


    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        # Final cleanup
        gc.collect()


if __name__ == "__main__":
    main()
