# src/__init__.py
"""
Space Debris Decay Prediction Project - Deployment Safe Version
"""

# Only import what actually exists and is safe for deployment
try:
    from .data_loader import load_data, create_target_variable, estimate_missing_decay_dates
    from .utils import calculate_statistics, create_sample_space_data, predict_decay_demo
    from .logger import get_logger
    from .exception import log_exception
except ImportError as e:
    # Graceful fallback for deployment
    print(f"Import warning in __init__.py: {e}")

# Define version
__version__ = "1.0.0"
__author__ = "Space Debris Team"

# List only available modules
__all__ = [
    'load_data',
    'create_target_variable', 
    'estimate_missing_decay_dates',
    'calculate_statistics',
    'create_sample_space_data',
    'predict_decay_demo',
    'get_logger',
    'log_exception'
]