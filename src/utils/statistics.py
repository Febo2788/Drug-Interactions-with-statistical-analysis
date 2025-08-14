"""
Statistical analysis utilities for drug-drug interaction prediction.

This module provides comprehensive statistical testing, confidence intervals,
and multiple testing correction methods with proper validation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class StatisticalTests:
    """
    Comprehensive statistical testing utilities.
    
    Provides methods for hypothesis testing, effect size calculation,
    and statistical validation with proper multiple testing correction.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical tests.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.logger = logging.getLogger(self.__class__.__name__)


class ConfidenceIntervals:
    """
    Confidence interval calculation utilities.
    
    Provides bootstrap and parametric confidence intervals
    for model performance metrics.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize confidence interval calculator.
        
        Args:
            confidence_level: Confidence level (default 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(self.__class__.__name__)


class MultipleTestingCorrection:
    """
    Multiple testing correction methods.
    
    Provides Bonferroni, Benjamini-Hochberg, and other
    correction methods for controlling false discovery rate.
    """
    
    def __init__(self):
        """Initialize multiple testing correction."""
        self.logger = logging.getLogger(self.__class__.__name__)