# tests/test_models.py
"""
Unit tests for model training and evaluation
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_training import compare_models, train_final_model, evaluate_model, plot_model_comparison

class TestModelTraining:
    """Test cases for model training functions"""
    
    def setUp(self):
        """Create sample data for testing"""
        # Generate sample data for testing
        np.random.seed(42)
        n_samples = 100
        
        self.X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        
        # Create target with some relationship to features
        self.y_train = (
            2 * self.X_train['feature1'] + 
            1.5 * self.X_train['feature2'] + 
            0.5 * self.X_train['feature3'] + 
            np.random.normal(0, 0.5, n_samples)
        )
        
        self.X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20),
            'feature3': np.random.normal(0, 1, 20)
        })
        
        self.y_test = (
            2 * self.X_test['feature1'] + 
            1.5 * self.X_test['feature2'] + 
            0.5 * self.X_test['feature3'] + 
            np.random.normal(0, 0.5, 20)
        )
    
    def test_compare_models(self):
        """Test model comparison function"""
        print("Testing compare_models...")
        self.setUp()
        
        results = compare_models(self.X_train, self.X_test, self.y_train, self.y_test)
        
        # Check that results are returned for all models
        expected_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Linear Regression', 'Ridge Regression']
        for model_name in expected_models:
            assert model_name in results, f"{model_name} not in results"
        
        # Check that each model has required metrics
        for model_name, model_results in results.items():
            assert 'MAE' in model_results
            assert 'RMSE' in model_results
            assert 'R2' in model_results
            assert 'Training_Time' in model_results
            assert 'model' in model_results
        
        print("PASS: compare_models returns results for all models with correct metrics")
    
    def test_train_final_model(self):
        """Test final model training"""
        print("Testing train_final_model...")
        self.setUp()
        
        model = train_final_model(self.X_train, self.y_train)
        
        # Check that model is trained
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')
        
        # Test prediction
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        
        print("PASS: train_final_model returns trained model")
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        print("Testing evaluate_model...")
        self.setUp()
        
        # Train a simple model for testing
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        metrics = evaluate_model(model, self.X_test, self.y_test, "Test Model")
        
        # Check that all metrics are returned
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        assert 'predictions' in metrics
        
        # Check metric values are reasonable
        assert metrics['MAE'] >= 0
        assert metrics['RMSE'] >= 0
        assert -1 <= metrics['R2'] <= 1  # R² should be between -1 and 1
        
        print("PASS: evaluate_model returns correct metrics")
    
    def test_evaluate_model_with_positive_r2(self):
        """Test that model evaluation returns positive R² for good models"""
        print("Testing evaluate_model with good data...")
        self.setUp()
        
        # Use a simple linear model that should perform well on linear data
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        metrics = evaluate_model(model, self.X_test, self.y_test, "Linear Model")
        
        # Since our data has a linear relationship, R² should be positive
        assert metrics['R2'] > 0, f"Expected positive R², got {metrics['R2']}"
        
        print("PASS: evaluate_model returns positive R² for good models")
    
    def test_plot_model_comparison(self):
        """Test model comparison plotting (without actually showing plot)"""
        print("Testing plot_model_comparison...")
        self.setUp()
        
        # Create sample results
        sample_results = {
            'Random Forest': {'RMSE': 100, 'R2': 0.95, 'MAE': 80, 'Training_Time': 10},
            'Gradient Boosting': {'RMSE': 120, 'R2': 0.92, 'MAE': 95, 'Training_Time': 15},
            'Linear Regression': {'RMSE': 150, 'R2': 0.85, 'MAE': 120, 'Training_Time': 2}
        }
        
        # Test that function runs without errors
        try:
            results_df = plot_model_comparison(sample_results, save_path=None)
            
            # Check that results are sorted by RMSE
            rmse_values = results_df['RMSE'].tolist()
            assert rmse_values == sorted(rmse_values), "Results should be sorted by RMSE"
            
            print("PASS: plot_model_comparison runs without errors and sorts results correctly")
        except Exception as e:
            pytest.fail(f"plot_model_comparison failed with error: {e}")

def test_model_performance_threshold():
    """Test that models achieve reasonable performance"""
    print("Testing model performance thresholds...")
    
    # Create simple linear data where models should perform well
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 50),
        'feature2': np.random.normal(0, 1, 50)
    })
    y = 2 * X['feature1'] + 1.5 * X['feature2'] + np.random.normal(0, 0.1, 50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from src.model_training import compare_models
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Check that at least one model has decent performance
    best_r2 = max(results[model]['R2'] for model in results)
    assert best_r2 > 0.5, f"Expected at least one model with R² > 0.5, got {best_r2}"
    
    print("PASS: Models achieve reasonable performance on simple data")

def run_all_model_tests():
    """Run all model tests"""
    print("Running Model Training Tests...")
    print("=" * 50)
    
    test_class = TestModelTraining()
    
    tests = [
        'test_compare_models',
        'test_train_final_model',
        'test_evaluate_model',
        'test_evaluate_model_with_positive_r2',
        'test_plot_model_comparison'
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
    
    # Run additional tests
    try:
        test_model_performance_threshold()
        passed += 1
        print("PASS: test_model_performance_threshold")
    except Exception as e:
        failed += 1
        print(f"FAIL: test_model_performance_threshold - {e}")
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("SUCCESS: All model training tests passed!")
    else:
        print(f"WARNING: {failed} tests failed")

if __name__ == "__main__":
    run_all_model_tests()