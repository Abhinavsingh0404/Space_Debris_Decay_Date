# tests/test_data_loader.py
"""
Unit tests for data_loader module
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Add the parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, create_target_variable, estimate_missing_decay_dates

# Use your actual data file path
DATA_FILE_PATH = 'data/space_decay.csv'

class TestDataLoader:
    """Test cases for data loading functions"""
    
    def test_load_data_success(self):
        """Test that load_data returns a DataFrame when file exists"""
        # Test with the actual file path
        df = load_data(DATA_FILE_PATH)
        
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            print(f"PASS: load_data() returns valid DataFrame with {len(df)} rows")
        else:
            pytest.skip(f"Data file not found at {DATA_FILE_PATH}")
    
    def test_load_data_with_default_path(self):
        """Test load_data with default path"""
        df = load_data()  # Uses default path from config
        # Even if file doesn't exist at default path, function should handle it gracefully
        assert df is None or isinstance(df, pd.DataFrame)
        print("PASS: load_data() handles default path correctly")
    
    def test_create_target_variable_structure(self):
        """Test that create_target_variable adds correct columns"""
        # Create sample data that matches your actual data structure
        sample_data = pd.DataFrame({
            'OBJECT_ID': [1, 2, 3],
            'NORAD_CAT_ID': [25544, 43013, 43014],
            'LAUNCH_DATE': ['2020-01-01', '2021-06-15', None],
            'DECAY_DATE': ['2022-01-01', None, None],
            'SEMIMAJOR_AXIS': [7000, 7200, 7500],
            'PERIOD': [90, 95, 100],
            'APOAPSIS': [7100, 7300, 7600],
            'PERIAPSIS': [6900, 7100, 7400],
            'INCLINATION': [51.6, 97.4, 28.5],
            'ECCENTRICITY': [0.001, 0.002, 0.0015]
        })
        
        result = create_target_variable(sample_data)
        
        # Check that DECAY_DAYS column was added
        assert 'DECAY_DAYS' in result.columns
        assert 'LAUNCH_DATE' in result.columns
        assert 'DECAY_DATE' in result.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result['LAUNCH_DATE'])
        assert pd.api.types.is_datetime64_any_dtype(result['DECAY_DATE'])
        
        print("PASS: create_target_variable() adds correct columns with proper data types")
    
    def test_create_target_variable_calculation(self):
        """Test that decay days are calculated correctly"""
        sample_data = pd.DataFrame({
            'LAUNCH_DATE': ['2020-01-01', '2021-01-01'],
            'DECAY_DATE': ['2020-02-01', '2021-03-01']  # 31 days and 59 days difference
        })
        
        result = create_target_variable(sample_data)
        
        # Check calculations
        expected_days = [31, 59]  # Based on the dates above
        calculated_days = result['DECAY_DAYS'].dropna().tolist()
        
        assert len(calculated_days) == 2
        assert calculated_days == expected_days
        
        print("PASS: create_target_variable() calculates decay days correctly")
    
    def test_estimate_missing_decay_dates(self):
        """Test that missing decay dates are estimated"""
        sample_data = pd.DataFrame({
            'OBJECT_ID': [1, 2, 3, 4],
            'LAUNCH_DATE': ['2020-01-01', '2021-01-01', None, '2022-01-01'],
            'DECAY_DATE': [None, None, None, '2023-01-01'],
            'PERIOD': [90, 600, 100, 1500],  # LEO, MEO, LEO, GEO
            'SEMIMAJOR_AXIS': [7000, 15000, 7200, 42000],
            'APOAPSIS': [7100, 15500, 7300, 42100],
            'PERIAPSIS': [6900, 14500, 7100, 41900]
        })
        
        # First create the target variable to get DECAY_DAYS column
        sample_data = create_target_variable(sample_data)
        result = estimate_missing_decay_dates(sample_data)
        
        # Check that all missing decay days are filled
        assert result['DECAY_DAYS'].isna().sum() == 0
        
        # Check that DECAY_ESTIMATED column was added/updated
        assert 'DECAY_ESTIMATED' in result.columns
        
        # Check that estimated flags are set correctly
        estimated_count = result['DECAY_ESTIMATED'].sum()
        assert estimated_count == 3  # First three rows should be estimated
        
        print("PASS: estimate_missing_decay_dates() fills all missing values")
    
    def test_estimate_missing_decay_dates_logic(self):
        """Test the estimation logic based on orbital parameters"""
        sample_data = pd.DataFrame({
            'LAUNCH_DATE': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'DECAY_DATE': [None, None, None],
            'PERIOD': [90, 300, 1500],  # LEO, MEO, GEO
            'SEMIMAJOR_AXIS': [7000, 10000, 42000],
            'APOAPSIS': [7100, 10500, 42100],
            'PERIAPSIS': [6900, 9500, 41900]
        })
        
        # First create the target variable to get DECAY_DAYS column
        sample_data = create_target_variable(sample_data)
        result = estimate_missing_decay_dates(sample_data)
        
        # Check that different orbit types get different estimates
        decay_days = result['DECAY_DAYS'].tolist()
        
        # Should have different values for different orbit types
        assert len(set(decay_days)) > 1
        
        print("PASS: estimate_missing_decay_dates() uses orbital parameters for estimation")

def test_data_quality_checks():
    """Test data quality aspects"""
    # Create test data with potential issues
    sample_data = pd.DataFrame({
        'LAUNCH_DATE': ['2020-01-01', 'invalid_date', None],
        'DECAY_DATE': ['2019-01-01', None, None],  # One date before launch (invalid)
        'SEMIMAJOR_AXIS': [7000, None, 7200],  # One missing value
        'PERIOD': [90, 95, -100]  # One negative value (invalid)
    })
    
    result = create_target_variable(sample_data)
    
    # Test that function handles invalid dates gracefully
    assert result is not None
    assert 'DECAY_DAYS' in result.columns
    
    print("PASS: Data loader handles invalid data gracefully")

def test_with_actual_data():
    """Test with the actual space decay data file"""
    if os.path.exists(DATA_FILE_PATH):
        df = load_data(DATA_FILE_PATH)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        
        # Test the full pipeline on actual data
        df_with_target = create_target_variable(df)
        assert 'DECAY_DAYS' in df_with_target.columns
        
        # If there are missing decay dates, test estimation
        if df_with_target['DECAY_DAYS'].isna().any():
            df_estimated = estimate_missing_decay_dates(df_with_target)
            assert df_estimated['DECAY_DAYS'].isna().sum() == 0
            print("PASS: Full data pipeline works with actual data")
        else:
            print("PASS: Actual data has complete decay dates")
    else:
        pytest.skip(f"Actual data file not found at {DATA_FILE_PATH}")

def run_all_tests():
    """Run all tests and print summary"""
    print("Running Data Loader Tests...")
    print("=" * 50)
    
    test_class = TestDataLoader()
    
    # Run each test method
    test_methods = [
        'test_load_data_success',
        'test_load_data_with_default_path',
        'test_create_target_variable_structure', 
        'test_create_target_variable_calculation',
        'test_estimate_missing_decay_dates',
        'test_estimate_missing_decay_dates_logic'
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            method()
            passed += 1
            print(f"PASS: {method_name}")
        except Exception as e:
            if "skip" in str(e).lower():
                skipped += 1
                print(f"SKIP: {method_name} - {e}")
            else:
                failed += 1
                print(f"FAIL: {method_name} - {e}")
    
    # Run additional tests
    try:
        test_data_quality_checks()
        passed += 1
        print("PASS: test_data_quality_checks")
    except Exception as e:
        failed += 1
        print(f"FAIL: test_data_quality_checks - {e}")
    
    try:
        test_with_actual_data()
        passed += 1
        print("PASS: test_with_actual_data")
    except Exception as e:
        if "skip" in str(e).lower():
            skipped += 1
            print(f"SKIP: test_with_actual_data - {e}")
        else:
            failed += 1
            print(f"FAIL: test_with_actual_data - {e}")
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("SUCCESS: All tests passed!")
    else:
        print(f"WARNING: {failed} tests failed")

if __name__ == "__main__":
    # Run tests when script is executed directly
    run_all_tests()