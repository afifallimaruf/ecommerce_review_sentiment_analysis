"""

Data loader untuk Amazon Reviews Dataset


"""


import pandas as pd
import json
import logging
import warnings
import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class untuk load dan preprocessing Reviews dataset"""
    def __init__(self, config_path: str = 'config.json'):
        self.config = self._load_config(config_path)
        self.base_path = Path(__file__).parent.parent.parent 
        self.raw_data_path = Path(self.config['data_paths']['raw'])
        self.df = None

        self.stats = {}


    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi dari JSON file"""
        try:
            with open(config_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {
                "data_paths": {
                    "raw": "data/raw/",
                }
            }
    
    def _ft_to_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset bertipe FastText dan ubah menjadi .csv

        Args:
            file_path: Alamat file yang digunakan
        
            Returns:
                pandas.DataFrame: Dataset yang sudah di ubah dari
                file berkestensi ft.txt menjadi .csv
        """
        
        chunk_size = 10000
        data = []
        current_chunk = []
        invalid_lines = 0
        # Baca file menggunakan fungsi open()
        with open(file_path, 'r', encoding='utf-8') as file:
            for _, line in enumerate(file):
                # Bersihkan baris dan pisahkan label dengan teks
                parts = line.strip().split(' ', 1)

                # Pastikan baris memiliki label dan teks
                if len(parts) == 2:
                    label, text = parts
                    # Ekstrak number dari __label__X
                    label_match = re.search(f'__label__(\d+)', label)
                    if not label_match:
                        invalid_lines += 1
                        continue

                    label = int(label_match.group(1))
                    # validasi label (harus 1 atau 2)
                    if label not in [1, 2]:
                        invalid_lines += 1
                        continue
                    
                    current_chunk.append([label, text])
                
                # Jika jumlah baris pada current_chunk sudah mencapai ukuran chunk_size(10000), simpan ke variabel data
                if len(current_chunk) >= chunk_size:
                    # Konversi data ke DataFrame
                    data.append(pd.DataFrame(current_chunk, columns=['labels', 'text']))
                    # Reset current_chunk
                    current_chunk = []
            # Simpan sisa baris yang tidak mencapai chunk_size(10000)
            if current_chunk:
                data.append(pd.DataFrame(current_chunk, columns=['labels', 'text']))
        
        # Gabungkan semua chunk
        self.df = pd.concat(data, ignore_index=True) if data else pd.DataFrame(columns=['label', 'text'])

        return self.df


    def load_dataset(self, file_name: str) -> pd.DataFrame:
        """
        Load dataset Amazon reviews
        Suports CSV, JSON

        Args:
            file_name: Nama file
        
            Returns:
                pandas.DataFrame: Dataset yang sudah di load
        """
        filepath = self.raw_data_path / file_name

        try:
            logger.info(f"Loading dataset from: {filepath}")
            # Jika dataset memiliki ekstensi .csv
            if file_name.endswith(".csv"):
                self.df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            # Jika dataset memiliki ekstensi .json
            elif file_name.endswith('.json'):
                self.df = pd.read_json(filepath, lines=True)
            # Jika dataset memiliki ekstensi .parquet
            elif file_name.endswith('.parquet'):
                self.df = pd.read_parquet(filepath)
            # Jika dataset memiliki ekstensi FastText
            elif file_name.endswith('.ft.txt'):
                self.df = self._ft_to_csv(filepath)
            else:
                # gunakan csv sebagai default
                self.df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error when loading dataset: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Membersihkan text dari karakter yang tidak diinginkan

        Args:
            text (str): Raw text

        
        Returns:
            str: Text yang sudah dibersihkan
        """

        # Ubah text ke bentuk lowercase
        text = text.lower()

        # Hapus HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Hapus URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Hapus alamat email
        text = re.sub(r'\S+@\S+', '', text)

        # Pertahankan alphanumeric, spasi, tanda baca dasar
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\-\'\"]', '', text)

        # Hapus ekstra spasi
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def analyze_dataset(self) -> Dict:
        """
        Analisis statistik pada dataset

        Returns:
            Dict: Statistik dataset
        """

        logger.info("Analyzing dataset statistics...")

        if self.df is None:
            raise ValueError("The dataset has not been loaded. Call load_dataset() first")

        # Distribusi label
        positive_dist = [len(label) for label in self.df['labels'] if label == 2 ]
        negative_dist = [len(label) for label in self.df['labels'] if label == 1]

        # Analisis text length
        word_counts = [text.split() for text in self.df['text']]

        all_words = []
        for text in self.df['text']:
            all_words.extend(text.split())
        
        vocabulary = Counter(all_words)
        unique_words = len(vocabulary)
        total_words = len(all_words)


        stats = {
            'total_samples': len(self.df['labels']),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'missing_value': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'label_distribution': {
                'positive': positive_dist,
                'negative': negative_dist,
                'postive_percentage': (positive_dist / len(self.df['labels'])) * 100,
                'negative_persentage': (negative_dist / len(self.df['lables'])) * 100, 
            },
            'text_length': {
                'avg_words': np.mean(word_counts),
                'median_words': np.median(word_counts),
                'min_words': np.min(word_counts),
                'max_words': np.max(word_counts),
                'std_words': np.std(word_counts),
            },
            'vocabulary': {
                'unique_words': unique_words,
                'total_words': total_words,
                'avg_word_frequency': total_words / unique_words
            }
        }

        self.stats = stats

        logger.info("Dataset analysis completed!")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Total columns: {stats['total_columns']}")
        logger.info(f"Columns: {stats['columns']}")
        logger.info(f"Missing value: {stats['missing_value']}")
        logger.info(f"Duplicate data: {stats['duplicates']}")
        logger.info(f"Positive: {stats['label_distribution']['positive']} ({stats['label_distribution']['positive_percentage']:.1}%)")
        logger.info(f"Negative: {stats['label_distribution']['negative']} ({stats['label_distribution']['negative_percentage']:.1}%)")
        logger.info(f"Average word per text: {stats['text_length']['avg_words']:.1f}")
        logger.info(f"Vocabulary size: {stats['vocabulary']['unique_words']}")

        return stats
    
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """
        Ambil data untuk preview

        Args:
            n: Jumlah baris data

        Returns:
            pandas.DataFrame: menanpilkan data
        """

        if self.df is None:
            raise ValueError("The dataset has not been loaded.")
    
        return self.df.head(n)
    

def main():
    """
    fungsi utama untuk menjalankan fungsi load_dataset,
    validate_dataset dan get_sample_data 
    """

    # Inisialisasi loader
    loader = DataLoader()

    # Nama file
    train_dataset_filename = "train.ft.txt"
    test_dataset_filename = "test.ft.txt"

    try:
        # Load raw data
        train_data = loader.load_dataset(train_dataset_filename)
        test_data = loader.load_dataset(test_dataset_filename)

        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
    
        # Validasi dataset
        df_val = loader.validate_dataset()
        logger.info(df_val)
        print(df_val)

        # ambil 5 data dari dataset
        data_priview = loader.get_sample_data()
        print(data_priview)
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    logger.info("Data loading started...")
    main()