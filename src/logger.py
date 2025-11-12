# src/logger.py
"""
Logging configuration for Space Debris Decay Prediction Project
"""
import logging
import os
import sys
from datetime import datetime

def setup_logging(log_dir="logs", level=logging.INFO):
    """
    Setup logging configuration for the project
    
    Args:
        log_dir (str): Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"space_debris_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Create a default logger instance
logger = setup_logging()

def get_logger(name):
    """
    Get a logger for a specific module
    
    Args:
        name (str): Module name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)