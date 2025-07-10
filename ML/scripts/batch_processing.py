"""
Seluruh kode dalam file ini untuk memproses data loader, data preprocessing
dan sentiment analyzer secara batch

Author: Afif
Date: 2025

"""

import subprocess
import sys
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
from log.logging import SetupLogging
from config.config import SetupLogging

# Inisialisasi logger dan set logger
logger_init = SetupLogging()
logger = logger_init.set_logger()


class BatchProcessor:
    """
    Class unutk mengorkestrasi dan menjalankan pemrosesan
    data secara batch
    """

    def __init__(self, config_path: str = "config.json"):
        self.config_init = SetupLogging()
        self.config = self.config_init.set_logger()
        self.script_path = Path(__file__).parent

        # Definisikan skrip
        self.scripts = {
            'data_loader': self.scripts / "data_loader.py",
            'data_preprocessor': self.scripts / "data_preprocessing.py",
            'sentiment_analyzer': self.scripts / "sentiment_analyzer.py"
        }

        # Validasi dependencies saat inisialisasi
        self._validate_dependencies()