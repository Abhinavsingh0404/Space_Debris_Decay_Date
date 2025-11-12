# main.py
"""
Main entry point for the Space Debris Decay Prediction Project
"""
import sys
import os

# Add src to Python path
sys.path.append('src')

from src.logger import setup_logging, get_logger
from src.exception import PipelineExecutionError, log_exception
from src.pipeline import run_full_pipeline

def main():
    """
    Main function to run the complete pipeline
    """
    # Setup logging at the start
    setup_logging()
    logger = get_logger('main')
    
    logger.info("Space Debris Decay Prediction System")
    logger.info("=" * 50)
    
    try:
        # Run the complete pipeline
        logger.info("Starting pipeline execution...")
        results = run_full_pipeline()
        
        if results is not None:
            logger.info("Pipeline executed successfully!")
            logger.info("Final Model Performance:")
            logger.info(f"  RMSE: {results['metrics']['RMSE']:.2f} days")
            logger.info(f"  MAE:  {results['metrics']['MAE']:.2f} days")
            logger.info(f"  R²:   {results['metrics']['R2']:.4f}")
            logger.info(f"  Models saved in: models/")
            logger.info(f"  Plots saved in: output/plots/")
            
            # Log performance summary
            if results['metrics']['R2'] > 0.95:
                logger.info("EXCELLENT: Model performance is outstanding!")
            elif results['metrics']['R2'] > 0.85:
                logger.info("GOOD: Model performance is very good!")
            else:
                logger.warning("MODERATE: Model performance could be improved.")
                
        else:
            logger.error("Pipeline failed to complete - returned None results")
            raise PipelineExecutionError("Pipeline execution returned no results")
            
    except PipelineExecutionError as e:
        logger.error("Pipeline execution failed with custom error")
        log_exception(logger, e)
        sys.exit(1)
        
    except Exception as e:
        error_msg = "Unexpected error during pipeline execution"
        logger.error(error_msg)
        log_exception(logger, e, error_msg)
        raise PipelineExecutionError(error_msg) from e

if __name__ == "__main__":
    main()