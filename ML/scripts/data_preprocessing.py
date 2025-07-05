"""
Data preprocessing lanjutan

Script ini adalah data preprocessing lanjutan untuk data text
seperti text normalization, tokenization, dan feature engineering


Author: Afif
Date: 2025

"""


import os
import re
import string
import pickle
import pandas as pd
import numpy as np
import nltk

from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from log.logging import SetupLogging


# Download data NLTK yang dibutuhkan
# Daftar yang akan diunduh
nltk_downloads = [
    'punkt_tab',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'vader_lexicon',
    'omw-1.4'
]

# Konfigurasi sistem logging.
logger_init = SetupLogging()
logger = logger_init.set_logger()


# Perulangan untuk mengunduh setiap paket NLTK
for download in nltk_downloads:
    try:
        nltk.download(download, quiet=True)
    except Exception as e:
        logger.error("Error when download paket NLTK: {e}")
        raise

class AdvanceTextPreprocessor:
    """
    text preprocessor lanjutan untuk sentiment analysis
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Inisialisasi preprocessor

        Args:
            config(Dict, optional): konfigurasi yang bersifat opsional
        """

        # Default configuration
        self.config = {
            'min_text_length': 3,
            'max_text_length': 500,
            'remove_stopwords': True,
            'apply_stemming': False,
            'apply_lemmatization': True,
            'extract_ngrams': True,
            'ngram_range': (1, 2),
            'min_word_freq': 2,
            'max_features': 10000,
            'apply_text_augmentation': False,
            'augmentation_factor': 0.1,
            'remove_rare_words': True,
            'expand_contractions': True,
            'normalize_numbers': True,
            'remove_duplicates': True
        }

        # Jika parameter config di isi, ini akan menimpa pengaturan default
        if config:
            self.config.update(config)
        
        # Inisialisasi NLP 
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Load stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()
            logger.warning("Could not load stopwords, using empty set")
        
        # Menentukan stopwords tambahan untuk domain ulasan pada e-commerce
        self.custom_stopwords = {
            'product', 'item', 'buy', 'bought', 'purchase', 'purchased',
            'order', 'ordered', 'amazon', 'seller', 'delivery', 'shipping'
        }

        # Menggabungkan stopwords standar NLTK dengan stopwords custom
        self.all_stopwords = self.stopwords.union(self.custom_stopwords)

        # Daftar singkatan atau kotraksi dan bentuk panjangnya
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "let's": "let us", "that's": "that is",
            "who's": "who is", "what's": "what is", "here's": "here is",
            "there's": "there is", "where's": "where is", "how's": "how is",
            "it's": "it is", "he's": "he is", "she's": "she is"
        }

        # Output directory
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Advance text preprocessor initialized")

        def load_data(self, file_path: str) -> pd.DataFrame:
            """
            Load data yang sudah pemrosesan dasar di file data_loader

            Args:
                file_path (str): Path/alamat ke file csv yang sudah di proses di awal
            
            Returns:
                pd.DataFrame: data yang sudah di load
            """

            try:
                df = pd.read_csv(file_path)
                logger.info("Load data has been successfuly")
                return df
            except Exception as e:
                logger.info(f"Error when load data: {e}")
                raise
        
        def expand_contractions(self, text: str) -> str:
            """
            Fungsi untuk mengubah bentuk singkat dari kata menjadi bentuk panjangnya
            misal: kata "don't" menjadi kata "do not"
            
            Args:
                text (str): Input teks
            
            Returns:
                str: Teks yang sudah diperluas
            """

            # Konversi teks ke lowercase
            expanded_text = text.lower()

            # Mengganti setiap kontraksi menjadi bentuk yang sudah dperluas
            for contraction, expansion in self.contraction.items():
                expanded_text = expanded_text.replace(contraction, expansion)

            return expanded_text
        

        def normalize_text(self, text: str) -> str:
            """
            Normalisasi teks

            Args:
                text (str): Input teks

            returns:
                str: teks yang sudah di normalisasi
            """

            # Ubah bentuk kata yang singkat(don't) menjadi (do not) jika konfigurasi 'expand_contractions' True.
            if self.config['expand_contractions']:
                text = self.expand_contractions(text)
            
            # Normalisasi angka dengan menggantinya denga placeholder 'NUMBER' jika normalize_numbers' True.
            if self.config['normalize_numbers']:
                text = re.sub(r'\d+', ' NUMBERS ', text)
            
            # Menghapus tanda baca yang muncul dua kali atau lebih
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{2,}', '.', text)

            # Menghilangkan spasi berlebihan
            text = re.sub(r'\s+', ' ', text)

            # Menghilangkan spasi di awal dan akhir teks
            text = text.strip()

            return text
        

        def advance_clean_text(self, text: str) -> str:
            """
            Text cleaning lanjutan dengan berbagai teknik

            Args:
                text (str): Input teks

            Returns:
                str: Teks yang sudah di bersihkan
            """

            # Konversi seluruh teks ke lowercase
            text = text.lower()

            # Menghapus kode HTML ('&amp', '&lt;')
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)

            # Menghapus URL.
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

            # Menghapus alamat email
            text = re.sub(r'\S+@\S+', ' ', text)

            # Menghapus semua tanda baca kecuali spasi
            text = re.sub(r'[\w\s]', ' ', text)

            # Menghapus karakter tunggal (kata tunggal) kecuali 'a' dan 'i'
            text = re.sub(r'\b[b-hj-z]\b', ' ', text)

            # Menghapus angka jika konfigurasi 'normalize_numbers' adalah False.
            if not self.config['normalize_numbers']:
                text = re.sub(r'\d+', ' ', text)
            
            # Menormalkan spasi lagi (mengganti beberapa spasi dengan satu spasi) dan menghilangkan spasi di awal/akhir.
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        
        def tokenize_and_process(self, text: str) -> List[str]:
            """
            Tokenize text dan menerapkan stemming/lemmatization

            Args:
                text (str): Input teks.

            
            Returns:
                List[str]: Tokens yang sudah diproses
            """

            # Melakukan tokenisasi teks menjadi kata-kata menggunakan NLTK
            try:
                tokens = word_tokenize(text)
            except:
                # Jika NLTK word_tokenize gagal.
                tokens = text.split()

            processed_tokens = []

            for token in tokens:
                # Lewati token kosong atau token dengan panjang kurang dari 2
                if not token or len(token) < 2:
                    continue

                # Lewati token yang berisi angka dan jika konfigurasi 'normalize_numbers' adalah False.
                if token.isdigit() and not self.config['normalize_numbers']:
                    continue

                # Hapus stopwords jika 'remove_stopwords' True dan token ada di daftar stopwords
                if self.config['remove_stopwords'] and token.lower() in self.all_stopwords:
                    continue

                # Terapkan stemming jika 'apply_stemming' True.
                if self.config['apply_stemming']:
                    token = self.stemmer.stem(token)
                # Terapkan lemmatization jika 'apply_lemmatization' True
                elif self.config['apply_lemmatization']:
                    token = self.lemmatizer.lemmatize(token)

                processed_tokens.append(token.lower())

            return processed_tokens
        
        