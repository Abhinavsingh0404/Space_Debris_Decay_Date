# space_decay_predictor/model_training.py
"""
Model training functions for Space Debris Decay Prediction
"""
import sys
import os

# Add the parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_training import (
    compare_models as src_compare_models,
    train_final_model as src_train_final_model,
    evaluate_model as src_evaluate_model,
    plot_model_comparison as src_plot_model_comparison,
    save_plot as src_save_plot
)

# Re-export the functions with the same names
compare_models = src_compare_models
train_final_model = src_train_final_model
evaluate_model = src_evaluate_model
plot_model_comparison = src_plot_model_comparison
save_plot = src_save_plot

__all__ = [
    'compare_models',
    'train_final_model', 
    'evaluate_model',
    'plot_model_comparison',
    'save_plot'
]