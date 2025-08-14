#!/usr/bin/env python3
"""
Exploratory Data Analysis Runner for Drug-Drug Interactions

This script performs comprehensive EDA on the drug interaction datasets
with statistical analysis and visualization generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import DrugDataLoader
from data.preprocessor import DataPreprocessor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('eda_analysis.log'),
            logging.StreamHandler()
        ]
    )
    

def perform_basic_eda():
    """Perform exploratory data analysis on drug interaction data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting EDA analysis...")
    
    # Load data
    interactions_df = pd.read_csv('../../data/raw/sample_drug_interactions.csv')
    molecular_df = pd.read_csv('../../data/raw/drug_smiles.csv')
    
    logger.info(f"Loaded {len(interactions_df)} drug interactions")
    logger.info(f"Loaded {len(molecular_df)} molecular structures")
    
    # Basic statistics
    print("\n=== DRUG INTERACTION ANALYSIS ===")
    print(f"Total interactions: {len(interactions_df)}")
    print(f"Unique drugs involved: {len(set(interactions_df['drug1_name'].tolist() + interactions_df['drug2_name'].tolist()))}")
    print(f"Severity distribution:")
    print(interactions_df['severity'].value_counts())
    print(f"\nEvidence level distribution:")
    print(interactions_df['evidence_level'].value_counts())
    
    # Molecular properties analysis
    print("\n=== MOLECULAR PROPERTIES ANALYSIS ===")
    print(f"Molecular weight range: {molecular_df['molecular_weight'].min():.1f} - {molecular_df['molecular_weight'].max():.1f}")
    print(f"LogP range: {molecular_df['logp'].min():.1f} - {molecular_df['logp'].max():.1f}")
    print(f"H-bond donors range: {molecular_df['hbd'].min()} - {molecular_df['hbd'].max()}")
    print(f"H-bond acceptors range: {molecular_df['hba'].min()} - {molecular_df['hba'].max()}")
    
    # Statistical analysis
    print("\n=== STATISTICAL ANALYSIS ===")
    
    # Test for normality of molecular weight
    from scipy import stats
    stat, p_value = stats.shapiro(molecular_df['molecular_weight'][:50])  # Shapiro-Wilk for small samples
    print(f"Molecular weight normality test (Shapiro-Wilk): statistic={stat:.4f}, p-value={p_value:.4f}")
    
    # Correlation analysis
    numerical_cols = ['molecular_weight', 'logp', 'hbd', 'hba']
    correlation_matrix = molecular_df[numerical_cols].corr()
    print(f"\nMolecular property correlations:")
    print(correlation_matrix.round(3))
    
    # Chi-square test for categorical associations
    severity_evidence_crosstab = pd.crosstab(interactions_df['severity'], interactions_df['evidence_level'])
    chi2, p_val, dof, expected = stats.chi2_contingency(severity_evidence_crosstab)
    print(f"\nSeverity-Evidence association test:")
    print(f"Chi-square statistic: {chi2:.4f}, p-value: {p_val:.4f}")
    
    logger.info("EDA analysis completed successfully")
    
    return {
        'interactions_count': len(interactions_df),
        'unique_drugs': len(set(interactions_df['drug1_name'].tolist() + interactions_df['drug2_name'].tolist())),
        'severity_distribution': interactions_df['severity'].value_counts().to_dict(),
        'molecular_stats': molecular_df[numerical_cols].describe().to_dict()
    }


def main():
    """Main execution function."""
    setup_logging()
    
    try:
        results = perform_basic_eda()
        print(f"\n=== EDA SUMMARY ===")
        print(f"Analysis completed successfully!")
        print(f"Key findings documented in eda_analysis.log")
        
    except Exception as e:
        logging.error(f"EDA analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()