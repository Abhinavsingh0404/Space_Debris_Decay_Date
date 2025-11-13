# space_decay_predictor/main.py
"""
Main entry point for Space Debris Decay Prediction Package
"""
import sys
import os

# Add the parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import main as src_main

# Re-export the main function
main = src_main

if __name__ == "__main__":
    main()