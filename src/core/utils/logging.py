import logging
from ..config import config


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level))

    # Add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
