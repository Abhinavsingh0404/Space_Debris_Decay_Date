# src/utils.py
"""
Utility functions for Space Debris Decay Prediction Project
"""
import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional

from .logger import get_logger
from .exception import log_exception

logger = get_logger('utils')

def create_directories(directories: List[str]) -> None:
    """
    Create multiple directories if they don't exist
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Directory created/verified: {directory}")
        except Exception as e:
            log_exception(logger, e, f"Error creating directory {directory}")
            raise

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Data saved to JSON: {filepath}")
    except Exception as e:
        log_exception(logger, e, f"Error saving JSON to {filepath}")
        raise

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with loaded data
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.debug(f"Data loaded from JSON: {filepath}")
        return data
    except Exception as e:
        log_exception(logger, e, f"Error loading JSON from {filepath}")
        raise

def save_yaml(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to YAML file
    
    Args:
        data: Dictionary to save
        filepath: Path to save YAML file
    """
    try:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.debug(f"Data saved to YAML: {filepath}")
    except Exception as e:
        log_exception(logger, e, f"Error saving YAML to {filepath}")
        raise

def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load data from YAML file
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary with loaded data
    """
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        logger.debug(f"Data loaded from YAML: {filepath}")
        return data
    except Exception as e:
        log_exception(logger, e, f"Error loading YAML from {filepath}")
        raise

def save_model(model: Any, filepath: str) -> None:
    """
    Save machine learning model
    
    Args:
        model: Trained model object
        filepath: Path to save model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved: {filepath}")
    except Exception as e:
        log_exception(logger, e, f"Error saving model to {filepath}")
        raise

def load_model(filepath: str) -> Any:
    """
    Load machine learning model
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded: {filepath}")
        return model
    except Exception as e:
        log_exception(logger, e, f"Error loading model from {filepath}")
        raise

def get_timestamp() -> str:
    """
    Get current timestamp in formatted string
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def calculate_statistics(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate basic statistics for specified columns
    
    Args:
        df: DataFrame
        columns: List of column names to analyze
        
    Returns:
        Dictionary with statistics for each column
    """
    stats = {}
    for col in columns:
        if col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'missing': int(df[col].isna().sum()),
                'missing_percentage': float(df[col].isna().sum() / len(df) * 100)
            }
    return stats

def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR method
    
    Args:
        df: DataFrame
        columns: List of column names to process
        factor: IQR factor (default 1.5)
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    outliers_count = 0
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Count outliers
            col_outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            outliers_count += col_outliers
            
            # Remove outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    logger.info(f"Removed {outliers_count} outliers using IQR method")
    return df_clean

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Calculate memory usage of DataFrame
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    def format_bytes(size):
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    return {
        'total_memory': format_bytes(memory_bytes),
        'shape': df.shape,
        'columns': len(df.columns),
        'rows': len(df)
    }

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            logger.warning("DataFrame is empty")
            return False
        
        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                return False
        
        # Check for excessive missing values
        missing_percentage = (df.isna().sum() / len(df)) * 100
        high_missing = missing_percentage[missing_percentage > 50]
        if not high_missing.empty:
            logger.warning(f"Columns with >50% missing values: {list(high_missing.index)}")
        
        logger.info("DataFrame validation passed")
        return True
        
    except Exception as e:
        log_exception(logger, e, "Error during DataFrame validation")
        return False

# Test the utility functions
if __name__ == "__main__":
    logger.info("Testing utils module...")
    
    # Test directory creation
    test_dirs = ['test_dir1', 'test_dir2/subdir']
    create_directories(test_dirs)
    
    # Test JSON operations
    test_data = {'test': 'data', 'number': 42}
    save_json(test_data, 'test_dir1/test.json')
    loaded_data = load_json('test_dir1/test.json')
    assert test_data == loaded_data
    
    # Test timestamp
    timestamp = get_timestamp()
    print(f"Current timestamp: {timestamp}")
    
    # Test statistics
    test_df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10, 20, 30, 40, 50]
    })
    stats = calculate_statistics(test_df, ['col1', 'col2'])
    print(f"Statistics: {stats}")
    
    # Clean up
    import shutil
    for dir_path in ['test_dir1', 'test_dir2']:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    logger.info("Utils module test completed successfully!")