# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def handle_missing_values(df, numeric_features):
    """
    Handle missing values in the dataset
    """
    print("Handling missing values...")
    
    # Fill numeric missing values with median
    for col in numeric_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   Filled {col} with median: {median_val:.4f}")
    
    return df

def engineer_features(df):
    """
    Create new features for the model
    """
    print("Engineering features...")
    
    # Space Era Classification
    def get_space_era(year):
        if pd.isna(year):
            return "Unknown"
        if year < 1970:
            return "Early Space Age"
        elif year < 1990:
            return "Cold War Space"
        elif year < 2010:
            return "Modern Space"
        else:
            return "New Space Era"
    
    # Extract launch year and create space era
    if 'LAUNCH_DATE' in df.columns:
        df['LAUNCH_YEAR'] = pd.to_numeric(pd.to_datetime(df['LAUNCH_DATE']).dt.year, errors='coerce')
        df['SPACE_ERA'] = df['LAUNCH_YEAR'].apply(get_space_era)
    
    # Orbit Type Classification
    def classify_orbit_type(period):
        if pd.isna(period):
            return 'UNKNOWN'
        if period <= 100:
            return 'LEO'
        elif period <= 600:
            return 'MEO'
        elif period >= 1400:
            return 'GEO'
        else:
            return 'OTHER'
    
    # Orbit Shape Classification
    def classify_orbit_shape(ecc):
        if pd.isna(ecc):
            return 'UNKNOWN'
        if ecc < 0.01:
            return 'CIRCULAR'
        elif ecc < 0.1:
            return 'NEAR_CIRCULAR'
        elif ecc < 0.5:
            return 'ELLIPTICAL'
        else:
            return 'HIGHLY_ELLIPTICAL'
    
    if 'PERIOD' in df.columns:
        df['ORBIT_TYPE'] = df['PERIOD'].apply(classify_orbit_type)
    
    if 'ECCENTRICITY' in df.columns:
        df['ORBIT_SHAPE'] = df['ECCENTRICITY'].apply(classify_orbit_shape)
    
    print("Feature engineering completed")
    return df

def prepare_features(df, features_list):
    """
    Prepare final feature set for modeling
    """
    print("Preparing features for modeling...")
    
    # Select only available features
    available_features = [f for f in features_list if f in df.columns]
    
    # Create feature matrix and target
    X = df[available_features].copy()
    
    if 'DECAY_DAYS' in df.columns:
        y = df['DECAY_DAYS']
    else:
        y = None
        print("WARNING: DECAY_DAYS column not found")
    
    print(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
    return X, y, available_features

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    """
    print("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled")
    return X_train_scaled, X_test_scaled, scaler