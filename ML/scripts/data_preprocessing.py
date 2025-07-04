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
import logging

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        