# space_decay_predictor/config.py
"""
Configuration for Space Debris Decay Prediction Package
"""
import os

# Paths
DATA_PATH = 'data/space_decay.csv'
MODEL_SAVE_PATH = 'models/'
OUTPUT_PATH = 'output/'
PLOTS_PATH = 'output/plots/'

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
PCA_VARIANCE_THRESHOLD = 0.9

# Feature Lists
NUMERIC_FEATURES = [
    'SEMIMAJOR_AXIS', 'PERIOD', 'MEAN_MOTION', 'BSTAR', 
    'APOAPSIS', 'PERIAPSIS', 'INCLINATION', 'ECCENTRICITY',
    'RA_OF_ASC_NODE', 'MEAN_ANOMALY', 'ARG_OF_PERICENTER'
]

EXCLUDE_FEATURES = ['OBJECT_ID', 'NORAD_CAT_ID', 'DECAY_DATE', 'LAUNCH_DATE', 'CREATION_DATE']

# Plotting Configuration
PLOT_CONFIG = {
    'FIGSIZE_MEDIUM': (12, 6),
    'FIGSIZE_LARGE': (15, 8),
    'FIGSIZE_SMALL': (10, 6),
    'DPI': 300
}