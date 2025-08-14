"""
Drug-Drug Interaction Prediction Package

A comprehensive machine learning pipeline for predicting drug-drug interactions
using Graph Neural Networks with rigorous statistical validation.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

from . import data, features, models, utils, visualization

__all__ = ["data", "features", "models", "utils", "visualization"]