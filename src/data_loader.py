# src/data_loader.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import logger and exceptions - use relative imports since we're inside src/
try:
    from .logger import get_logger
    from .exception import DataLoadingError, DataValidationError, log_exception
except ImportError:
    # Fallback for when running this file directly
    from logger import get_logger
    from exception import DataLoadingError, DataValidationError, log_exception

# Get module-specific logger
logger = get_logger('data_loader')

def load_data(file_path='data/space_decay.csv'):
    """
    Load the space debris dataset
    """
    logger.info("Loading dataset...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        error_msg = f"Data file not found: {file_path}"
        log_exception(logger, e, error_msg)
        raise DataLoadingError(error_msg) from e
    except Exception as e:
        error_msg = f"Error loading dataset from {file_path}"
        log_exception(logger, e, error_msg)
        raise DataLoadingError(error_msg) from e

def create_target_variable(df):
    """
    Create DECAY_DAYS target variable from date columns
    """
    logger.info("Creating target variable...")
    
    try:
        # Ensure datetime columns
        df['LAUNCH_DATE'] = pd.to_datetime(df['LAUNCH_DATE'], errors='coerce')
        df['DECAY_DATE'] = pd.to_datetime(df['DECAY_DATE'], errors='coerce')
        
        valid_launch = df['LAUNCH_DATE'].notna().sum()
        missing_launch = df['LAUNCH_DATE'].isna().sum()
        valid_decay = df['DECAY_DATE'].notna().sum()
        missing_decay = df['DECAY_DATE'].isna().sum()
        
        logger.info(f"Launch dates: {valid_launch} valid, {missing_launch} missing")
        logger.info(f"Decay dates: {valid_decay} valid, {missing_decay} missing")
        
        # Calculate decay days only for rows with both dates
        mask = df['LAUNCH_DATE'].notna() & df['DECAY_DATE'].notna()
        df.loc[mask, 'DECAY_DAYS'] = (df.loc[mask, 'DECAY_DATE'] - df.loc[mask, 'LAUNCH_DATE']).dt.days
        
        # For rows missing decay dates but having launch dates, we'll estimate later
        launch_only_mask = df['LAUNCH_DATE'].notna() & df['DECAY_DATE'].isna()
        df.loc[launch_only_mask, 'DECAY_DAYS'] = np.nan
        
        valid_samples = df['DECAY_DAYS'].notna().sum()
        total_samples = len(df)
        
        logger.info(f"Target variable created. Valid samples: {valid_samples}/{total_samples}")
        
        if valid_samples == 0:
            logger.warning("No valid decay dates found. We'll need to estimate them.")
        
        return df
        
    except Exception as e:
        error_msg = "Error creating target variable"
        log_exception(logger, e, error_msg)
        raise DataValidationError(error_msg) from e

def estimate_missing_decay_dates(df):
    """
    Estimate decay dates for objects where they're missing
    Using orbital parameters to make educated guesses
    """
    logger.info("Estimating missing decay dates...")
    
    try:
        # Count before estimation
        missing_before = df['DECAY_DAYS'].isna().sum()
        logger.info(f"Missing decay dates before estimation: {missing_before}")
        
        # Create DECAY_ESTIMATED column if it doesn't exist
        if 'DECAY_ESTIMATED' not in df.columns:
            df['DECAY_ESTIMATED'] = False
        
        # Simple estimation based on orbital parameters
        estimated_count = 0
        for idx, row in df[df['DECAY_DAYS'].isna()].iterrows():
            if pd.notna(row['LAUNCH_DATE']):
                # Use orbital period for estimation if available
                if pd.notna(row.get('PERIOD')):
                    if row['PERIOD'] < 100:  # LEO objects
                        estimated_days = 365 * 5  # 5 years for LEO
                    elif row['PERIOD'] < 600:  # MEO objects
                        estimated_days = 365 * 10  # 10 years for MEO
                    else:  # GEO objects
                        estimated_days = 365 * 50  # 50 years for GEO
                # Use semi-major axis if period is not available
                elif pd.notna(row.get('SEMIMAJOR_AXIS')):
                    altitude = row['SEMIMAJOR_AXIS'] - 6371  # Earth radius in km
                    if altitude < 2000:  # LEO
                        estimated_days = 365 * 5
                    elif altitude < 10000:  # MEO
                        estimated_days = 365 * 15
                    else:  # GEO
                        estimated_days = 365 * 50
                else:
                    # Default estimation
                    estimated_days = 365 * 10  # Default 10 years
                
                df.loc[idx, 'DECAY_DAYS'] = estimated_days
                df.loc[idx, 'DECAY_DATE'] = row['LAUNCH_DATE'] + pd.Timedelta(days=estimated_days)
                df.loc[idx, 'DECAY_ESTIMATED'] = True
                estimated_count += 1
        
        # Handle cases where launch date is also missing
        missing_launch_mask = df['LAUNCH_DATE'].isna() & df['DECAY_DAYS'].isna()
        if missing_launch_mask.any():
            missing_launch_count = missing_launch_mask.sum()
            logger.info(f"Handling {missing_launch_count} samples with missing launch dates...")
            # Use median decay days from estimated values
            median_decay_days = df[df['DECAY_ESTIMATED']]['DECAY_DAYS'].median()
            df.loc[missing_launch_mask, 'DECAY_DAYS'] = median_decay_days
            df.loc[missing_launch_mask, 'DECAY_ESTIMATED'] = True
            estimated_count += missing_launch_count
            logger.info(f"Assigned median decay days ({median_decay_days}) to samples with missing launch dates")
        
        missing_after = df['DECAY_DAYS'].isna().sum()
        
        logger.info(f"Estimated {estimated_count} decay dates. Remaining missing: {missing_after}")
        
        if missing_after > 0:
            logger.warning(f"{missing_after} samples still have missing decay dates. Filling with median...")
            median_val = df['DECAY_DAYS'].median()
            df['DECAY_DAYS'].fillna(median_val, inplace=True)
            logger.info(f"Filled remaining missing values with median: {median_val}")
        
        return df
        
    except Exception as e:
        error_msg = "Error estimating missing decay dates"
        log_exception(logger, e, error_msg)
        raise DataValidationError(error_msg) from e

# Test the functions if this file is run directly
if __name__ == "__main__":
    logger.info("Testing data_loader module directly")
    try:
        df = load_data()
        if df is not None:
            df = create_target_variable(df)
            
            # If no valid decay dates, estimate them
            if df['DECAY_DAYS'].isna().all() or df['DECAY_DAYS'].notna().sum() == 0:
                logger.info("No valid decay dates found. Estimating...")
                df = estimate_missing_decay_dates(df)
            
            logger.info(f"Final dataset: {df['DECAY_DAYS'].notna().sum()} samples with target variable")
            logger.info("Data loader test completed successfully!")
        else:
            logger.error("Failed to load data for testing")
            
    except Exception as e:
        logger.error(f"Data loader test failed: {e}")