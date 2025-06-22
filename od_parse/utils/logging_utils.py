"""
Logging utility functions for the od-parse library.
"""

import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
    
    Returns:
        Logger object
    """
    if level is None:
        level = logging.INFO
    
    logger = logging.getLogger(name)
    
    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Logging level to set
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific loggers to appropriate levels
    logging.getLogger('od_parse').setLevel(level)

    # Suppress verbose logs from third-party libraries unless in debug mode
    if level > logging.DEBUG:
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
