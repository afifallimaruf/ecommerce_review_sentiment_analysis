"""
class untuk load config.json

Author: Afif
Date:2025

"""

import json
from typing import Dict
from log.logging import SetupLogging
from pathlib import Path


logger_init = SetupLogging()
logger = logger_init.set_logger()

class LoadConfig:
    """ Class untuk load config.json """
    def __init__(self, config_name="config.json"):
        root_path = Path(__file__).resolve().parent.parent.parent
        self.config_path = root_path / config_name

    def load_config(self) -> Dict:
        """Load konfigurasi dari JSON file"""
        try:
            with open(self.config_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error when load config file: {e}")