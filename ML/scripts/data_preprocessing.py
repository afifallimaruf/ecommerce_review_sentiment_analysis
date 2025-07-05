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


# Konfigurasi sistem logging.
logger_init = SetupLogging()
logger = logger_init.set_logger()


