# space_decay_predictor/preprocessing.py
"""
Preprocessing functions for Space Debris Decay Prediction
"""
import sys
import os

# Add the parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import (
    handle_missing_values as src_handle_missing_values,
    engineer_features as src_engineer_features,
    prepare_features as src_prepare_features, 
    scale_features as src_scale_features
)

# Re-export the functions with the same names
handle_missing_values = src_handle_missing_values
engineer_features = src_engineer_features
prepare_features = src_prepare_features
scale_features = src_scale_features

__all__ = [
    'handle_missing_values', 
    'engineer_features', 
    'prepare_features', 
    'scale_features'
]