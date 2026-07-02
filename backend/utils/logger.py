"""
Logging Configuration
=====================
Structured logging for the Semantix backend.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a configured logger.

    Args:
        name: logger name (typically __name__)

    Returns:
        configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
