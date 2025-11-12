# space_decay_predictor/pipeline.py
"""
Pipeline functions for Space Debris Decay Prediction
"""
import sys
import os

# Add the parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import (
    run_full_pipeline as src_run_full_pipeline,
    predict_new_data as src_predict_new_data,
    save_pipeline as src_save_pipeline,
    plot_predictions as src_plot_predictions
)

# Re-export the functions with the same names
run_full_pipeline = src_run_full_pipeline
predict_new_data = src_predict_new_data
save_pipeline = src_save_pipeline
plot_predictions = src_plot_predictions

__all__ = [
    'run_full_pipeline',
    'predict_new_data',
    'save_pipeline', 
    'plot_predictions'
]