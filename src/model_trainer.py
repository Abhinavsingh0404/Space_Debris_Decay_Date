# src/model_trainer.py
"""
Advanced model training with hyperparameter tuning and optimization
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import time

from .logger import get_logger
from .exception import ModelTrainingError, log_exception

logger = get_logger('model_trainer')

class ModelTrainer:
    """
    Advanced model trainer with hyperparameter tuning capabilities
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.training_history = []
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='random_forest', cv=5):
        """
        Perform hyperparameter tuning for specified model type
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to tune ('random_forest', 'xgboost', 'gradient_boosting')
            cv: Number of cross-validation folds
            
        Returns:
            Best model and parameters
        """
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        try:
            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
                model = RandomForestRegressor(random_state=self.random_state)
                
            elif model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
                model = xgb.XGBRegressor(random_state=self.random_state)
                
            elif model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                }
                model = GradientBoostingRegressor(random_state=self.random_state)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Use RandomizedSearchCV for faster tuning
            grid_search = RandomizedSearchCV(
                model, param_grid, n_iter=20, cv=cv, scoring='r2',
                n_jobs=-1, random_state=self.random_state
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            
            logger.info(f"Hyperparameter tuning completed in {training_time:.2f} seconds")
            logger.info(f"Best {model_type} parameters: {self.best_params}")
            logger.info(f"Best cross-validation R²: {self.best_score:.4f}")
            
            # Store training history
            self.training_history.append({
                'model_type': model_type,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'training_time': training_time
            })
            
            return self.best_model, self.best_params
            
        except Exception as e:
            error_msg = f"Error in hyperparameter tuning for {model_type}"
            log_exception(logger, e, error_msg)
            raise ModelTrainingError(error_msg) from e
    
    def train_ensemble(self, X_train, y_train, models_to_use=None):
        """
        Train an ensemble of models
        
        Args:
            X_train: Training features
            y_train: Training target
            models_to_use: List of model types to include in ensemble
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training ensemble of models...")
        
        if models_to_use is None:
            models_to_use = ['random_forest', 'xgboost', 'gradient_boosting']
        
        ensemble_models = {}
        
        for model_type in models_to_use:
            try:
                logger.info(f"Training {model_type} for ensemble...")
                model, params = self.hyperparameter_tuning(X_train, y_train, model_type)
                ensemble_models[model_type] = {
                    'model': model,
                    'params': params,
                    'type': model_type
                }
            except Exception as e:
                logger.warning(f"Failed to train {model_type} for ensemble: {e}")
                continue
        
        logger.info(f"Ensemble training completed. Successfully trained {len(ensemble_models)} models")
        return ensemble_models
    
    def evaluate_ensemble(self, ensemble_models, X_test, y_test):
        """
        Evaluate ensemble models on test data
        
        Args:
            ensemble_models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with model performance metrics
        """
        logger.info("Evaluating ensemble models...")
        
        results = []
        
        for model_name, model_info in ensemble_models.items():
            model = model_info['model']
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'model': model_name,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'parameters': model_info['params']
                })
                
                logger.info(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.2f}")
                
            except Exception as e:
                logger.warning(f"Error evaluating {model_name}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('r2', ascending=False)
        
        return results_df
    
    def create_weighted_ensemble(self, ensemble_models, X_test, y_test, method='performance'):
        """
        Create a weighted ensemble based on model performance
        
        Args:
            ensemble_models: Dictionary of trained models
            X_test: Test features for weight calculation
            y_test: Test target for weight calculation
            method: Weighting method ('performance', 'equal')
            
        Returns:
            Function that makes weighted predictions
        """
        logger.info("Creating weighted ensemble...")
        
        # Get individual predictions
        predictions = {}
        performances = {}
        
        for model_name, model_info in ensemble_models.items():
            try:
                y_pred = model_info['model'].predict(X_test)
                predictions[model_name] = y_pred
                performances[model_name] = r2_score(y_test, y_pred)
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
                continue
        
        if method == 'performance':
            # Weight by R² score (negative scores get zero weight)
            weights = {name: max(0, perf) for name, perf in performances.items()}
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: weight/total_weight for name, weight in weights.items()}
            else:
                # Fallback to equal weights
                weights = {name: 1/len(predictions) for name in predictions.keys()}
        else:
            # Equal weights
            weights = {name: 1/len(predictions) for name in predictions.keys()}
        
        logger.info(f"Ensemble weights: {weights}")
        
        def weighted_predict(X):
            """Make weighted ensemble predictions"""
            ensemble_pred = np.zeros(len(X))
            for model_name, weight in weights.items():
                model_pred = ensemble_models[model_name]['model'].predict(X)
                ensemble_pred += weight * model_pred
            return ensemble_pred
        
        return weighted_predict, weights
    
    def save_model(self, filepath):
        """Save the best model to file"""
        if self.best_model is None:
            raise ModelTrainingError("No model has been trained yet")
        
        try:
            joblib.dump(self.best_model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            error_msg = f"Error saving model to {filepath}"
            log_exception(logger, e, error_msg)
            raise ModelTrainingError(error_msg) from e
    
    def load_model(self, filepath):
        """Load a model from file"""
        try:
            self.best_model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            error_msg = f"Error loading model from {filepath}"
            log_exception(logger, e, error_msg)
            raise ModelTrainingError(error_msg) from e

# Convenience functions
def optimize_model(X_train, y_train, model_type='random_forest', cv=5):
    """
    Convenience function for quick model optimization
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to optimize
        cv: Cross-validation folds
        
    Returns:
        Optimized model and parameters
    """
    trainer = ModelTrainer()
    return trainer.hyperparameter_tuning(X_train, y_train, model_type, cv)

def create_ensemble_predictor(X_train, y_train, X_test, y_test, models=None):
    """
    Convenience function for creating a weighted ensemble predictor
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data for weight calculation
        models: List of model types to include
        
    Returns:
        Ensemble prediction function and weights
    """
    trainer = ModelTrainer()
    ensemble_models = trainer.train_ensemble(X_train, y_train, models)
    return trainer.create_weighted_ensemble(ensemble_models, X_test, y_test)

if __name__ == "__main__":
    # Test the model trainer
    logger.info("Testing ModelTrainer class...")
    
    # Create sample data
    np.random.seed(42)
    X_test = np.random.randn(100, 5)
    y_test = X_test[:, 0] + 2*X_test[:, 1] + np.random.randn(100)*0.1
    
    # Test hyperparameter tuning
    trainer = ModelTrainer()
    try:
        model, params = trainer.hyperparameter_tuning(X_test, y_test, 'random_forest')
        logger.info("ModelTrainer test completed successfully!")
    except Exception as e:
        logger.error(f"ModelTrainer test failed: {e}")