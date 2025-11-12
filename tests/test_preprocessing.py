# tests/test_preprocessing.py
"""
Unit tests for preprocessing module
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import handle_missing_values, engineer_features, prepare_features, scale_features

class TestPreprocessing:
    """Test cases for preprocessing functions"""
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        print("Testing handle_missing_values...")
        
        # Create sample data with missing values
        sample_data = pd.DataFrame({
            'SEMIMAJOR_AXIS': [7000, np.nan, 7200, 7500],
            'PERIOD': [90, 95, np.nan, 100],
            'APOAPSIS': [7100, 7300, 7600, np.nan],
            'INCLINATION': [51.6, 97.4, 28.5, 45.0]
        })
        
        numeric_features = ['SEMIMAJOR_AXIS', 'PERIOD', 'APOAPSIS', 'INCLINATION']
        result = handle_missing_values(sample_data, numeric_features)
        
        # Check that missing values are filled
        assert result['SEMIMAJOR_AXIS'].isna().sum() == 0
        assert result['PERIOD'].isna().sum() == 0
        assert result['APOAPSIS'].isna().sum() == 0
        
        print("PASS: handle_missing_values fills missing values correctly")
    
    def test_engineer_features(self):
        """Test feature engineering"""
        print("Testing engineer_features...")
        
        sample_data = pd.DataFrame({
            'LAUNCH_DATE': ['2020-01-01', '1985-06-15', '2015-03-20'],
            'PERIOD': [90, 300, 1500],
            'ECCENTRICITY': [0.001, 0.05, 0.6]
        })
        
        result = engineer_features(sample_data)
        
        # Check that new features are created
        assert 'SPACE_ERA' in result.columns
        assert 'ORBIT_TYPE' in result.columns
        assert 'ORBIT_SHAPE' in result.columns
        
        print("PASS: engineer_features creates new features correctly")
    
    def test_prepare_features(self):
        """Test feature preparation"""
        print("Testing prepare_features...")
        
        sample_data = pd.DataFrame({
            'SEMIMAJOR_AXIS': [7000, 7200, 7500],
            'PERIOD': [90, 95, 100],
            'APOAPSIS': [7100, 7300, 7600],
            'DECAY_DAYS': [1000, 1500, 2000]
        })
        
        features_list = ['SEMIMAJOR_AXIS', 'PERIOD', 'APOAPSIS']
        X, y, feature_names = prepare_features(sample_data, features_list)
        
        assert X.shape[1] == len(features_list)
        assert len(y) == len(sample_data)
        assert feature_names == features_list
        
        print("PASS: prepare_features returns correct features and target")
    
    def test_scale_features(self):
        """Test feature scaling"""
        print("Testing scale_features...")
        
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        X_test = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [60, 70]
        })
        
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert scaler is not None
        
        print("PASS: scale_features scales features correctly")

def run_all_preprocessing_tests():
    """Run all preprocessing tests"""
    print("Running Preprocessing Tests...")
    print("=" * 50)
    
    test_class = TestPreprocessing()
    
    tests = [
        'test_handle_missing_values',
        'test_engineer_features',
        'test_prepare_features',
        'test_scale_features'
    ]
    
    passed = 0
    failed = 0
    
    for test_name in tests:
        try:
            test_method = getattr(test_class, test_name)
            test_method()
            passed += 1
            print(f"PASS: {test_name}")
        except Exception as e:
            failed += 1
            print(f"FAIL: {test_name} - {e}")
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("SUCCESS: All preprocessing tests passed!")
    else:
        print(f"WARNING: {failed} tests failed")

if __name__ == "__main__":
    run_all_preprocessing_tests()