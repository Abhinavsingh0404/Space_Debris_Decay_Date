# src/exception.py
"""
Custom exception classes for Space Debris Decay Prediction Project
"""
import sys
import traceback

class SpaceDebrisException(Exception):
    """Base exception class for space debris project"""
    
    def __init__(self, error_message, error_detail=None):
        """
        Initialize the exception
        
        Args:
            error_message (str): Human readable error message
            error_detail: Exception details from sys.exc_info()
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail
    
    def get_detailed_message(self):
        """Get detailed error message with traceback"""
        if self.error_detail:
            error_class, error, tb = self.error_detail
            filename = tb.tb_frame.f_code.co_filename
            line_number = tb.tb_lineno
            error_message = f"Error occurred in file: {filename} at line: {line_number}\n"
            error_message += f"Error class: {error_class.__name__}\n"
            error_message += f"Error message: {self.error_message}"
            return error_message
        return self.error_message
    
    def __str__(self):
        return self.get_detailed_message()

class DataLoadingError(SpaceDebrisException):
    """Exception raised for errors in data loading"""
    pass

class DataValidationError(SpaceDebrisException):
    """Exception raised for data validation errors"""
    pass

class PreprocessingError(SpaceDebrisException):
    """Exception raised for errors in data preprocessing"""
    pass

class FeatureEngineeringError(SpaceDebrisException):
    """Exception raised for errors in feature engineering"""
    pass

class ModelTrainingError(SpaceDebrisException):
    """Exception raised for errors in model training"""
    pass

class ModelEvaluationError(SpaceDebrisException):
    """Exception raised for errors in model evaluation"""
    pass

class ModelSavingError(SpaceDebrisException):
    """Exception raised for errors in saving models"""
    pass

class ConfigurationError(SpaceDebrisException):
    """Exception raised for configuration errors"""
    pass

class PipelineExecutionError(SpaceDebrisException):
    """Exception raised for errors in pipeline execution"""
    pass

def log_exception(logger, exception, custom_message=None):
    """
    Log exception with custom message
    
    Args:
        logger: Logger instance
        exception: Exception object
        custom_message (str, optional): Additional context message
    """
    if custom_message:
        logger.error(f"{custom_message}: {str(exception)}")
    else:
        logger.error(f"Exception occurred: {str(exception)}")
    
    # Log traceback for debugging
    logger.error(f"Traceback: {traceback.format_exc()}")