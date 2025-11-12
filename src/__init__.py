# src/__init__.py
"""
Space Debris Decay Prediction Project
"""
from .data_loader import load_data, create_target_variable, estimate_missing_decay_dates
from .preprocessing import handle_missing_values, engineer_features, prepare_features, scale_features
from .model_training import compare_models, train_final_model, evaluate_model, plot_model_comparison, save_plot
from .pipeline import run_full_pipeline, predict_new_data
from .logger import setup_logging, get_logger
from .exception import (
    SpaceDebrisException, DataLoadingError, DataValidationError, 
    PreprocessingError, FeatureEngineeringError, ModelTrainingError, 
    ModelEvaluationError, ModelSavingError, ConfigurationError, 
    PipelineExecutionError, log_exception
)