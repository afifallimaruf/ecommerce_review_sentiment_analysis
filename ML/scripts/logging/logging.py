"""

Class untuk setup loggging

Author: Afif
Date: 2025

"""


import logging



class SetupLogging:

    def set_logger(self) -> logging.Logger:
        """
        Set logger

        Returns:
            logging.Logger: instances dari class Logger
        """
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        return logger