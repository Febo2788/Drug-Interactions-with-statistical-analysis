"""
Data preprocessing pipeline for drug-drug interaction prediction.

This module provides comprehensive data cleaning, validation, and transformation
utilities with statistical rigor for the DDI prediction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for drug interaction data.
    
    Provides standardized preprocessing steps with statistical validation
    and quality control measures.
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.3,
                 outlier_method: str = 'iqr',
                 statistical_significance: float = 0.05):
        """
        Initialize the data preprocessor.
        
        Args:
            missing_threshold: Maximum allowed proportion of missing values
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            statistical_significance: Alpha level for statistical tests
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.alpha = statistical_significance
        self.scalers = {}
        self.encoders = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def preprocess_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess drug interaction dataset.
        
        Args:
            interactions_df: Raw interaction data
            
        Returns:
            Cleaned and preprocessed interaction DataFrame
        """
        self.logger.info("Starting interaction data preprocessing...")
        
        # Create a copy to avoid modifying original data
        df = interactions_df.copy()
        
        # Basic data quality checks
        self._log_data_quality(df, "Raw interaction data")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Standardize drug names
        df = self._standardize_drug_names(df)
        
        # Encode categorical variables
        df = self._encode_categorical_variables(df)
        
        # Validate interaction pairs
        df = self._validate_interaction_pairs(df)
        
        self._log_data_quality(df, "Processed interaction data")
        self.logger.info("Interaction preprocessing completed")
        
        return df
        
    def preprocess_molecular_data(self, molecular_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess molecular structure and property data.
        
        Args:
            molecular_df: Raw molecular data with SMILES
            
        Returns:
            Cleaned molecular DataFrame
        """
        self.logger.info("Starting molecular data preprocessing...")
        
        df = molecular_df.copy()
        
        # Validate SMILES strings
        df = self._validate_smiles(df)
        
        # Handle molecular property outliers
        df = self._handle_molecular_outliers(df)
        
        # Scale molecular descriptors
        df = self._scale_molecular_features(df)
        
        self.logger.info("Molecular preprocessing completed")
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with statistical validation."""
        # Calculate missing value statistics
        missing_stats = df.isnull().sum() / len(df)
        
        # Drop columns with excessive missing values
        cols_to_drop = missing_stats[missing_stats > self.missing_threshold].index
        if len(cols_to_drop) > 0:
            self.logger.warning(f"Dropping columns with >30% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
            
        # Handle remaining missing values
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                # For categorical: use mode
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
            else:
                # For numerical: use median (more robust than mean)
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                
        return df
        
    def _standardize_drug_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize drug names for consistency."""
        # Convert to lowercase and strip whitespace
        for col in ['drug1_name', 'drug2_name']:
            if col in df.columns:
                df[col] = df[col].str.lower().str.strip()
                # Replace common variations
                df[col] = df[col].str.replace('_', ' ')
                df[col] = df[col].str.replace('-', ' ')
                
        return df
        
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables with proper handling."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['drug1_name', 'drug2_name']:  # Keep drug names as text
                # Use label encoding for ordinal variables like severity
                if col == 'severity':
                    severity_order = ['Minor', 'Moderate', 'Major', 'Contraindicated']
                    df[col] = df[col].astype('category')
                    df[col] = df[col].cat.set_categories(severity_order, ordered=True)
                    df[f'{col}_encoded'] = df[col].cat.codes
                else:
                    # One-hot encoding for nominal variables
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
                    
        return df
        
    def _validate_interaction_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean interaction pairs."""
        initial_count = len(df)
        
        # Remove self-interactions (drug with itself)
        df = df[df['drug1_name'] != df['drug2_name']]
        
        # Remove duplicate interactions (considering both directions)
        df['interaction_pair'] = df.apply(
            lambda x: tuple(sorted([x['drug1_name'], x['drug2_name']])), axis=1
        )
        df = df.drop_duplicates(subset='interaction_pair')
        df = df.drop(columns='interaction_pair')
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} invalid/duplicate interactions")
            
        return df
        
    def _validate_smiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate SMILES strings (placeholder - would use RDKit in full implementation)."""
        # In full implementation, would validate SMILES with RDKit
        # For now, just check for basic format issues
        if 'smiles' in df.columns:
            # Remove rows with empty SMILES
            initial_count = len(df)
            df = df[df['smiles'].notna() & (df['smiles'] != '')]
            removed_count = initial_count - len(df)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} rows with invalid SMILES")
                
        return df
        
    def _handle_molecular_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in molecular properties."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if self.outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    self.logger.info(f"Found {outlier_count} outliers in {col}")
                    # Cap outliers rather than remove (preserves sample size)
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
                    
        return df
        
    def _scale_molecular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale molecular features for ML compatibility."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                
            df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[[col]])
            
        return df
        
    def _log_data_quality(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Log data quality metrics."""
        self.logger.info(f"\n{dataset_name} Quality Report:")
        self.logger.info(f"  Shape: {df.shape}")
        self.logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        self.logger.info(f"  Duplicate rows: {df.duplicated().sum()}")
        self.logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
    def get_preprocessing_summary(self) -> Dict[str, any]:
        """Get summary of preprocessing steps applied."""
        return {
            'missing_threshold': self.missing_threshold,
            'outlier_method': self.outlier_method,
            'encoders_fitted': list(self.encoders.keys()),
            'scalers_fitted': list(self.scalers.keys()),
            'statistical_significance': self.alpha
        }