"""
Logging configuration - outputs to both terminal and logs/app.log.
"""
import os
import logging
import sys

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, 'logs')


def setup_logger(name: str = None) -> logging.Logger:
    """
    Create and return a configured logger that outputs to both
    stdout (terminal) and logs/app.log.
    """
    # Ensure logs directory exists
    os.makedirs(_LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name or __name__)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(_LOG_DIR, 'app.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
