"""

Data loader untuk Amazon Reviews Dataset


"""


import os
import pandas as pd
import json
import logging
import warnings
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Dict, Optional
from config.config import LoadConfig
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class untuk load dan preprocessing Reviews dataset"""
    def __init__(self, config_path: str = 'config.json'):
        self.config = LoadConfig(config_path=config_path)
        self.base_path = Path(__file__).parent.parent.parent 
        self.raw_data_path = Path(self.config['data_paths']['raw'])
        self.output_dir = Path(self.config['data_paths']['output'])
        self.processed_data_path = Path(self.config['data_paths']['processed'])
        
        self.df = None

        self.stats = {}

    
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

        return text

    def analyze_dataset(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analisis statistik pada dataset

        Args: DataFrame opsional untuk di analisis (gunakan self.df jika tidak tersedia)

        Returns:
            Dict: Statistik dataset
        """

        if df is not None:
            self.df = df

        logger.info("Analyzing dataset statistics...")

        total_samples = len(self.df)
        
        # Distribusi label
        label_counts = self.df['labels'].value_counts()
        positive_count = label_counts.get(2, 0)
        negative_count = label_counts.get(1, 0)

        chunk_size = 10000

        # Analisis text length
        word_counts_list = []
        vocabulary = Counter()

        for i in range(0, len(self.df), chunk_size):
            chunk = self.df['text'].iloc[i:i+chunk_size]

            # proses word counts untuk chunk ini
            chunk_word_counts = chunk.str.split().str.len()
            word_counts_list.extend(chunk_word_counts.to_list())

            # Perbarui vocabulary
            for text in chunk:
                vocabulary.update(str(text).split())

        # Ubah kedalam  pandas series untuk statistik
        word_counts = pd.Series(word_counts_list)

        unique_words = len(vocabulary)
        total_words = sum(vocabulary.values())

        avg_word_prequency = total_words / unique_words if unique_words > 0 else 0

        stats = {
            'total_samples': len(self.df['labels']),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'missing_value': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'label_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'positive_percentage': (positive_count / total_samples) * 100 if total_samples > 0 else 0,
                'negative_percentage': (negative_count / total_samples) * 100 if total_samples > 0 else 0, 
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
                'most_common_overall': vocabulary.most_common(20),
                'avg_word_frequency': avg_word_prequency
            }
        }

        self.stats = stats

        logger.info("Dataset analysis completed!")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Total columns: {stats['total_columns']}")
        logger.info(f"Columns: {stats['columns']}")
        logger.info(f"Missing value: {stats['missing_value']}")
        logger.info(f"Duplicate data: {stats['duplicates']}")
        logger.info(f"Positive: {stats['label_distribution']['positive']} ({stats['label_distribution']['positive_percentage']:.1f}%)")
        logger.info(f"Negative: {stats['label_distribution']['negative']} ({stats['label_distribution']['negative_percentage']:.1f}%)")
        logger.info(f"Average word per text: {stats['text_length']['avg_words']:.1f}")
        logger.info(f"Vocabulary size: {stats['vocabulary']['unique_words']}")

        return stats
    
    def create_visualizations(self) -> None:
        """
        Membuat visualisasi dataset
        """

        logger.info("Creating dataset visualizations")

        if self.df is None:
            raise ValueError("The dataset has not been loaded. Call load_dataset() first.")

        # Set style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Amazon Dataset Analysis', fontsize=14, fontweight='bold')

        # Distribusi label
        labels = ['Negative', 'Positive']
        sizes = [self.stats['label_distribution']['negative'],
                 self.stats['label_distribution']['positive']
                ]
        colors = ['#ff6b6b', '#4ecdc4']

        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Sentiment Distribution')

        # Distribusi Text Length 
        word_counts = self.df['text'].str.count(r'\S+').fillna(0).astype(int)
        axes[0, 1].hist(word_counts, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Text Length Distribution (Words)')
        axes[0, 1].set_xlabel('Number of Words')
        axes[0, 1].set_ylabel('Frequency')
        mean_words = np.mean(word_counts)
        axes[0, 1].axvline(np.mean(mean_words), color='red', linestyle='--', label=f'Mean: {mean_words:.1f}')
        axes[0, 1].legend()

        # Kata yang paling umum (overall)
        common_words = self.stats['vocabulary']['most_common_overall'][:15]
        words, counts = zip(*common_words)

        axes[1, 0].barh(range(len(words)), counts, color='lightcoral')
        axes[1, 0].set_yticks(range(len(words)))
        axes[1, 0].set_yticklabels(words)
        axes[1, 0].set_title('Most Common Words (Overall)')
        axes[1, 0].set_xlabel('Frequency')

        # Perbandingan sentimen (Positive vs Negative)
        pos_count = self.stats['label_distribution']['positive']
        neg_count = self.stats['label_distribution']['negative']

        sentiment_data = ['Positive', 'Negative']
        sentiment_counts = [pos_count, neg_count]

        bars = axes[1, 1].bar(sentiment_data, sentiment_counts, color=['#4ecdc4', '#ff6b6b'])
        axes[1, 1].set_title('Review Count by Sentiment')
        axes[1, 1].set_ylabel('Number of Reviews')

        # Tambahkan label nilai pada bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / 'dataset_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualizations saved to: {plot_path}")

    def save_processed_data(self, df: Optional[pd.DataFrame], context: str = 'train', apply_cleaning: bool = True) -> None:
        """
        Menyimpan data yang sudah diproses ke file csv
        
        Args:
            df: file DataFrame
            context_file: konteks apakah file untuk training atau testing (train/test)
        
        """

        self.df = df

        if df is None:
            raise ValueError("The dataset has not been loaded.")
        
        if context not in ['train', 'test']:
            raise ValueError("The context is wrong. must(train/test)")
        
        # Buat folder jika belum ada
        train_folder = self.processed_data_path / 'training'
        train_folder.mkdir(exist_ok=True)

        test_folder = self.processed_data_path / 'testing'
        test_folder.mkdir(exist_ok=True)

        # Alamat untuk file yang akan di save
        if context == 'train':
            save_path = train_folder / 'train_processed.csv'
        else:
            save_path = test_folder / 'test_processed.csv'

        try:
            # Buat salinan untuk menghindari perubahan pada file asli
            df_to_save = self.df.copy()
            # Jika konteks file untuk training
            if apply_cleaning and context == 'train':
                logger.info("Applying text cleaning")
                # bersihkan teks pada kolom text sebelum disimpan
                df_to_save['cleaned_text'] = df_to_save['text'].apply(self._clean_text)
            
            # Save file dengan menggunakan kompresi data untuk file yang besar
            df_to_save.to_csv(save_path, index=False, compression='gzip' if len(df_to_save) > 100000 else None)
            logger.info("The dataset has been saved successfully")
        except Exception as e:
            logger.error(f"Error when save dataset: {e}")
            raise
    
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
    
        # Analisis dataset
        stats = loader.analyze_dataset(train_data)


        # Buat visualisasi dataset
        loader.create_visualizations()

        loader.save_processed_data(train_data)
        loader.save_processed_data(test_data, context='test', apply_cleaning=False)

        # ambil 5 data dari dataset
        data_priview = loader.get_sample_data()
        print(data_priview)
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    logger.info("Data loading started...")
    main()
