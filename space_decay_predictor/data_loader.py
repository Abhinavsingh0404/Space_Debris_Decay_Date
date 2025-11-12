# space_decay_predictor/data_loader.py
"""
Data loading functions for Space Debris Decay Prediction
"""
import sys
import os

# Add the parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import (
    load_data as src_load_data,
    create_target_variable as src_create_target_variable, 
    estimate_missing_decay_dates as src_estimate_missing_decay_dates
)

# Re-export the functions with the same names
load_data = src_load_data
create_target_variable = src_create_target_variable
estimate_missing_decay_dates = src_estimate_missing_decay_dates

__all__ = ['load_data', 'create_target_variable', 'estimate_missing_decay_dates']