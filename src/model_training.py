# src/model_training.py
"""
Model training, evaluation, and comparison functions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import xgboost as xgb
import time
import joblib

from src.config import RANDOM_STATE, TEST_SIZE, PCA_VARIANCE_THRESHOLD, PLOT_CONFIG

def save_plot(filename, folder='output/plots'):
    """Save plot to specified folder."""
    import os
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{filename}", bbox_inches='tight', dpi=300)

def compare_models(X_train, X_test, y_train, y_test):
    """
    Compare multiple regression models
    """
    print("Comparing multiple models...")
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Training_Time': training_time,
            'model': model
        }
        
        print(f"     {name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    return results

def train_final_model(X_train, y_train):
    """
    Train the final Random Forest model
    """
    print("Training final model...")
    
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    print("✅ Final model trained")
    return model

def evaluate_model(model, X_test, y_test, model_name="Final Model"):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"   MAE:  {mae:.2f} days")
    print(f"   RMSE: {rmse:.2f} days")
    print(f"   R²:   {r2:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'predictions': y_pred}

def plot_model_comparison(results, save_path=None):
    """
    Create comparison plots for model performance
    """
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.sort_values('RMSE')
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # RMSE comparison
    ax1.bar(results_df.index, results_df['RMSE'], color='skyblue')
    ax1.set_title('Model Comparison - RMSE')
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.tick_params(axis='x', rotation=45)
    
    # R² comparison
    ax2.bar(results_df.index, results_df['R2'], color='lightgreen')
    ax2.set_title('Model Comparison - R² Score')
    ax2.set_ylabel('R² (higher is better)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(save_path)
    
    plt.show()
    return results_df