# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import logging
from colorama import Fore, Style, init

# Initialize Colorama to enable ANSI codes on Windows as well
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors based on log level.
    """
    # Define formats for each log level
    FORMATS = {
        logging.DEBUG:    Fore.CYAN + "%(asctime)s - DEBUG - %(message)s" + Style.RESET_ALL,
        logging.INFO:     Fore.GREEN + "%(asctime)s - INFO - %(message)s" + Style.RESET_ALL,
        logging.WARNING:  Fore.YELLOW + "%(asctime)s - WARNING - %(message)s" + Style.RESET_ALL,
        logging.ERROR:    Fore.RED + "%(asctime)s - ERROR - %(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.MAGENTA + "%(asctime)s - CRITICAL - %(message)s" + Style.RESET_ALL,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class ColorLogger:
    """
    Wrapper for Python's built-in logger that applies colored formatting.
    Usage:
        logger = ColorLogger("my_logger").get_logger()
        logger.info("This is an info message.")
    """
    def __init__(self, name: str, level: int = logging.DEBUG) -> None:
        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent adding multiple handlers if already configured
        if not self.logger.handlers:
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(ColoredFormatter())
            self.logger.addHandler(ch)

    def get_logger(self) -> logging.Logger:
        return self.logger

logger = ColorLogger(__name__, level=logging.DEBUG).get_logger()