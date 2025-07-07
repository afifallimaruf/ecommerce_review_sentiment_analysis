"""
Script untuk melakukan sentiment analysis meggunakan algoritma ML
dengan dataset yang sudah diproses dalam format JSON.

Support:
- Traditional ML (SVM, Naive Bayes, Random Forest)
- Ensemble methods

Author: Afif
Date: 2025

"""

import json
import pickle
import numpy as np
import pandas as pd
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from config.config import LoadConfig
from log.logging import SetupLogging

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
warnings.filterwarnings('ignore')

