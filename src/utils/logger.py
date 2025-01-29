"""Logging configuration for the application."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.config.settings import LOG_LEVEL, LOG_FILE


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(LOG_LEVEL)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(LOG_LEVEL)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
