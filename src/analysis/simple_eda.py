#!/usr/bin/env python3
"""
Simple EDA script for drug interaction analysis without heavy dependencies.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_data():
    """Perform basic exploratory data analysis."""
    print("=== Drug-Drug Interaction EDA ===\n")
    
    # Load data
    try:
        interactions_df = pd.read_csv('../../data/raw/sample_drug_interactions.csv')
        molecular_df = pd.read_csv('../../data/raw/drug_smiles.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("1. DATASET OVERVIEW")
    print(f"   Interactions dataset: {interactions_df.shape[0]} rows, {interactions_df.shape[1]} columns")
    print(f"   Molecular dataset: {molecular_df.shape[0]} rows, {molecular_df.shape[1]} columns")
    
    print("\n2. INTERACTION ANALYSIS")
    # Count unique drugs
    all_drugs = set(interactions_df['drug1_name'].tolist() + interactions_df['drug2_name'].tolist())
    print(f"   Total unique drugs involved: {len(all_drugs)}")
    
    # Severity distribution
    print(f"   Severity distribution:")
    severity_counts = interactions_df['severity'].value_counts()
    for severity, count in severity_counts.items():
        print(f"     {severity}: {count} ({count/len(interactions_df)*100:.1f}%)")
    
    # Evidence level distribution
    print(f"   Evidence level distribution:")
    evidence_counts = interactions_df['evidence_level'].value_counts()
    for evidence, count in evidence_counts.items():
        print(f"     {evidence}: {count} ({count/len(interactions_df)*100:.1f}%)")
    
    print("\n3. MOLECULAR PROPERTIES ANALYSIS")
    print(f"   Molecular weight statistics:")
    mw_stats = molecular_df['molecular_weight'].describe()
    print(f"     Mean: {mw_stats['mean']:.1f}")
    print(f"     Std: {mw_stats['std']:.1f}")
    print(f"     Range: {mw_stats['min']:.1f} - {mw_stats['max']:.1f}")
    
    print(f"   LogP statistics:")
    logp_stats = molecular_df['logp'].describe()
    print(f"     Mean: {logp_stats['mean']:.2f}")
    print(f"     Std: {logp_stats['std']:.2f}")
    print(f"     Range: {logp_stats['min']:.1f} - {logp_stats['max']:.1f}")
    
    print("\n4. DATA QUALITY ASSESSMENT")
    print(f"   Interactions missing values:")
    interactions_missing = interactions_df.isnull().sum()
    for col, missing in interactions_missing.items():
        if missing > 0:
            print(f"     {col}: {missing} ({missing/len(interactions_df)*100:.1f}%)")
        
    print(f"   Molecular data missing values:")
    molecular_missing = molecular_df.isnull().sum()
    for col, missing in molecular_missing.items():
        if missing > 0:
            print(f"     {col}: {missing} ({missing/len(molecular_df)*100:.1f}%)")
    
    print("\n5. KEY INSIGHTS")
    print(f"   - Dataset contains {len(interactions_df)} drug-drug interactions")
    print(f"   - {severity_counts.get('Major', 0)} major severity interactions require attention")
    print(f"   - {evidence_counts.get('High', 0)} interactions have high-quality evidence")
    print(f"   - Molecular weight range suggests diverse drug types")
    print(f"   - Ready for feature engineering and model development")
    
    print("\n=== EDA Complete ===")

if __name__ == "__main__":
    analyze_data()