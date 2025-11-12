# space_decay_predictor/__init__.py 
"""
Space Debris Decay Prediction Package

A machine learning pipeline to predict space debris decay dates using orbital parameters.
"""

from .data_loader import load_data, create_target_variable, estimate_missing_decay_dates
from .preprocessing import handle_missing_values, engineer_features, prepare_features, scale_features
from .model_training import compare_models, train_final_model, evaluate_model
from .model_trainer import ModelTrainer, optimize_model, create_ensemble_predictor
from .pipeline import run_full_pipeline, predict_new_data
from .utils import (
    create_directories,
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    save_model,
    load_model,
    get_timestamp,
    calculate_statistics,
    remove_outliers_iqr,
    format_time,
    memory_usage,
    validate_dataframe
)
from .config import DATA_PATH, RANDOM_STATE, NUMERIC_FEATURES

__version__ = "0.1.0"
__author__ = "Abhi"
__email__ = "abhinavsingh121299@gmail.com"

__all__ = [
    'load_data',
    'create_target_variable', 
    'estimate_missing_decay_dates',
    'handle_missing_values',
    'engineer_features',
    'prepare_features',
    'scale_features',
    'compare_models',
    'train_final_model',
    'evaluate_model',
    'ModelTrainer',
    'optimize_model',
    'create_ensemble_predictor',
    'run_full_pipeline',
    'predict_new_data',
    # Utils functions
    'create_directories',
    'save_json',
    'load_json',
    'save_yaml', 
    'load_yaml',
    'save_model',
    'load_model',
    'get_timestamp',
    'calculate_statistics',
    'remove_outliers_iqr',
    'format_time',
    'memory_usage',
    'validate_dataframe',
    # Config constants
    'DATA_PATH',
    'RANDOM_STATE',
    'NUMERIC_FEATURES'
]