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
from typing import Generator, Iterator, List, Dict, Tuple, Optional, Set
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
            'chunk_size': 5000,
            'max_features': 10000,
            'batch_size': 1000,
            'memory_threshold': 1000
        }

        if config:
            self.config.update(config)


        # Inisialisasi NLTK
        self._lemmatizer = None
        self._stopwords = None
        self._contractions = None
        self._patterns = None

        
        # output directory
        self.output_dir = Path("data/processed/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Counter untuk tracking progress
        self.total_processed = 0
        self.chunk_counter = 0

        # Pre-compile patterns untuk performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns"""
        self._patterns = {
            'contraction': re.compile(r"won't|can't|n't|'re|'ve|'ll|'d|'m|it's|that's", re.IGNORECASE),
            'punctuation': re.compile(r'[^\w\s]'),
            'digits': re.compile(r'\d+'),
            'spaces': re.compile(r'\s+'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        }

    @property
    def lemmatizer(self):
        """ loading lemmatizer """
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer
    
    @property
    def stopwords(self):
        """Loading stopwords"""
        if self._stopwords is None:
            try:
                base_stopwords = set(stopwords.words('english'))
            except:
                base_stopwords = set()
            
            # Tambahkan stopwords khusus
            additional_stopwords = {
                'product', 'item', 'buy', 'bought', 'purchase', 'amazon',
                'good', 'bad', 'ok', 'okay', 'yes', 'no', 'well', 'get'
            }

            self._stopwords = frozenset(base_stopwords | additional_stopwords)
        return self._stopwords

    @property
    def contractions(self):
        """Loading contractions"""

        if self._contractions is None:
            self._contractions = {
                "won't": "will not", "can't": "cannot", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
                "'m": " am", "it's": "it is", "that's": "that is"
            }
        return self._contractions
    
    def _check_memory_usage(self) -> float:
        """
        Check penggunaan memory

        Returns float: Ukuran memory bertipe float
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _cleanup_memory(self):
        """garbage collection"""
        gc.collect()

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


    def load_data_generator(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """
        Load data yang sudah pemrosesan dasar di file data_loader

        Args:
            file_path (str): Path/alamat ke file csv yang sudah di proses di awal
            
        Returns:
            pd.DataFrame: data yang sudah di load
        """

        chunk_size = self.config['chunk_size']

        dtypes = {
            'labels': 'int8',
            'cleaned_text': 'string'
        }

        # Hanya load kolom yang diperlukan
        usecols = ['labels', 'cleaned_text']

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
                            self._cleanup_memory()
                    if chunk:
                        yield pd.DataFrame(chunk)
            else:
                # Handle regular CSV files
                for chunk in pd.read_csv(
                    file_path,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    engine='c',
                    chunksize=chunk_size,
                    low_memory=False,
                    dtype=dtypes,
                    usecols=usecols
                ):
                    chunk = chunk.dropna(subset=['cleaned_text'])
                    if not chunk.empty:
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

        if not isinstance(text, str):
            return ""
        
        # Ubah teks ke dalam bentuk lowercase
        text = text.lower()

        # Hapus URLs dan emails
        text = self._patterns['url'].sub('', text)
        text = self._patterns['email'].sub('', text)
        
        # Ubah kata yang singkat (don't) dari dictionary contractions menjadi bentuk panjangnya(do not)
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        

        # Hapus tanda baca dan angka
        text = self._patterns['punctuation'].sub(' ', text)
        text = self._patterns['digits'].sub(' ', text)

        # Hapus ekstra spasi
        text = self._patterns['spaces'].sub(' ', text).strip()

        return text
    
    def tokenize_and_process(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenisasi dan proses teks

        Args:
            texts (List[str]): List teks


        Returns:
            List[List[str]]: List texts yang sudah ditokenisasi 
        """

        results = []

        for text in texts:
            try:
                # Simple split lebih cepat untuk text yang sudah dibersihkan
                tokens = text.split()
                
                # Filter dan lemmatize
                processed_tokens = []
                for token in tokens:
                    # Skip jika panjang token kurang dari 2
                    if len(token) < 2:
                        continue

                    # Skip stopwords
                    if self.config['remove_stopwords'] and token in self.stopwords:
                        continue
                    
                    # Apply lemmatization
                    if self.config['apply_lemmatization']:
                        token = self.lemmatizer.lemmatize(token)

                    processed_tokens.append(token)
                results.append(processed_tokens)
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                results.append([])

        return results
    
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

        return {
            'word_count': len(tokens),
            'char_count': text_len,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'uppercase_ratio': uppercase_count / text_len if text_len > 0 else 0
        }
    
    def preprocess_batch(self, batch_df: pd.DataFrame) -> List[Dict]:
        """
        Proses batch data

        Args:
            batch_df (pd.DataFrame): Batch dataframe


        Returns:
            List[Dict]: Output samples yang sudah di proses
        """

        # Filter berdasarkan panjang text
        batch_df = batch_df[
            (batch_df['cleaned_text'].str.len() >= self.config['min_text_length']) &
            (batch_df['cleaned_text'].str.len() <= self.config['max_text_length'])
        ]

        if batch_df.empty:
            return []

        # Bersihkan teks dalam batch
        cleaned_texts = [self.clean_text(text) for text in batch_df['cleaned_text']]
        
        # Tokenisasi dalam batch
        all_tokens = self.tokenize_and_process(cleaned_texts)
        
        # Results
        results = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            tokens = all_tokens[idx]

            # Skip jika tidak ada token
            if not tokens:
                continue

            # Ekstrak minimal features
            features = self.extract_features(cleaned_texts[idx], tokens)

            result = {
                'cleaned_text': cleaned_texts[idx],
                'tokens': ' '.join(tokens),
                'labels': row['labels'],
                'word_count': features['word_count'],
                'char_count': features['char_count'],
                'exclamation_count': features['exclamation_count'],
                'question_count': features['question_count'],
                'uppercase_ratio': features['uppercase_ratio'],
            }

            results.append(result)
        return results
    
    def process_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Proses chunk dengan batch processing

        Args:
            df (pd.DataFrame): Input chunk dataset


        Returns:
            pd.DataFrame
        """
        
        # Hapus data duplikat
        chunk_df = chunk_df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
        
        
        # Proses sampel
        batch_size = self.config['batch_size']
        all_processed_samples = []

        for start_idx in range(0, len(chunk_df), batch_size):
            end_idx = min(start_idx + batch_size, len(chunk_df))
            batch = chunk_df.iloc[start_idx:end_idx]
         
            try:
                processed_samples = self.preprocess_batch(batch)

                if processed_samples:
                    all_processed_samples.extend(processed_samples)

            except Exception as e:
                logger.warning(f"Error processing sample {start_idx}-{end_idx}: {e}")
                continue
            # Log progress
            if (end_idx) % 2000 == 0:
                logger.info(f"Processed {end_idx}/{len(chunk_df)} samples in chunk {self.chunk_counter + 1}")
                
            # Periodic garbage collection
            if len(all_processed_samples) > 10000:
                gc.collect()
        logger.info(f"Chunk {self.chunk_counter + 1} completed: {len(all_processed_samples)} samples processed")
        # Convert ke dataframe
        if all_processed_samples:
            processed_df = pd.DataFrame(all_processed_samples)        
    
            processed_df['labels'] = processed_df['labels'].astype('int8')
            processed_df['word_count'] = processed_df['word_count'].astype('int16')
            processed_df['char_count'] = processed_df['char_count'].astype('int16')
            processed_df['exclamation_count'] = processed_df['exclamation_count'].astype('int8')
            processed_df['question_count'] = processed_df['question_count'].astype('int8')
            processed_df['uppercase_ratio'] = processed_df['uppercase_ratio'].astype('float32')
        else:
            processed_df = pd.DataFrame()
        
        # Cleanup
        del chunk_df
        self._cleanup_memory()

        return processed_df
    
    def save_chunk_compressed(self, processed_df: pd.DataFrame, chunk_idx: int) -> None:
        """
        Save data yang sudah di proses

        Args:
            processed_df (pd.DataFrame): dataframe
            chunk_idx (int): Chunk index
        """

        if processed_df.empty:
            logger.warning(f"No processed samples to save for chunk {chunk_idx}")
            return

        # Save dengan kompresi gzip
        file_path = self.output_dir / f'chunk_{chunk_idx:04d}.csv.gz'

        processed_df.to_csv(
            file_path,
            index=False,
            encoding='utf-8',
            compression='gzip'
        )

        logger.info(f"Chunk {chunk_idx} saved: {file_path} ({len(processed_df)} samples)")

        # Update counter
        self.total_processed += len(processed_df)

    def merge_chunks(self) -> None:
        """
        Merge semua chunk files menjadi satu file final
        """

        logger.info("Merging all chunks...")

        # Cari semua chunk files
        chunk_files = sorted(self.output_dir.glob('chunk_*.csv*'))

        if not chunk_files:
            logger.warning("No chunk files found to merge")
            return
        
        # Merge chunks
        all_texts = []
        chunk_dataframes = []
        for chunk_file in chunk_files:
            try:
                if chunk_file.suffix == '.gz':
                    chunk_df = pd.read_csv(chunk_file, encoding='utf-8', compression='gzip')
                else:
                    chunk_df = pd.read_csv(chunk_file, encoding='utf-8')
                
                chunk_dataframes.append(chunk_df)
                all_texts.extend(chunk_df['tokens'].tolist())

                # Hapus chunk file setelah diproses
                chunk_file.unlink()
            except Exception as e:
                logger.info(f"Error preprocessing chunk file {chunk_file}: {e}")
                continue
        
        if chunk_dataframes:
            final_df = pd.concat(chunk_dataframes, ignore_index=True)

            # Save final file
            final_file = self.output_dir / 'train_preprocessed.csv.gz'
            final_df.to_csv(
                final_file,
                index=False,
                encoding='utf-8',
                compression='gzip'
            )
        
        logger.info(f"Final file saved: {final_file} ({len(final_df)} samples)")
        
        
        # Cleanup
        del chunk_dataframes
        del all_texts
        del final_df
        self._cleanup_memory()


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

        logger.info("Loading and processing data...")
        with memory_monitor():
            for chunk_idx, chunk in enumerate(preprocessor.load_data_generator(file_path)):
                logger.info(f"Preprocessing chunk {chunk_idx + 1}...")

                # Set chunk counter
                preprocessor.chunk_counter = chunk_idx
                
                # Preprocess chunk
                processed_df = preprocessor.process_chunk(chunk)
                
                # Save chunk langsung
                preprocessor.save_chunk_compressed(processed_df, chunk_idx)

                # Cleanup
                del processed_df
                gc.collect()
        
                # Monitor memori setiap 10 chunk
                if (chunk_idx + 1) % 10 == 0:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Memory usage after chunk {chunk_idx + 1}: {memory_mb:.1f}MB")

            
        # Merge semua chunks
        preprocessor.merge_chunks()


        logger.info("PREPROCESSING COMPLETED!")


    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        # Final cleanup
        gc.collect()


if __name__ == "__main__":
    main()
