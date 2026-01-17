"""Centralized logging configuration."""

import logging
import sys
from pathlib import Path

from ecommerce_image_validator.config import settings


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the module)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    
    Examples
    --------
    >>> from validator.logger import setup_logger
    >>> logger = setup_logger(__name__)
    >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger