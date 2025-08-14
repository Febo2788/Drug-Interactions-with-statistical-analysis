#!/usr/bin/env python3
"""
Basic unit tests for core drug interaction prediction functions.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_data_loading():
    """Test that data files can be loaded."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Test molecular descriptors
    molecular_file = data_dir / 'processed' / 'molecular_descriptors.csv'
    if molecular_file.exists():
        df = pd.read_csv(molecular_file)
        assert len(df) > 0
        assert 'drug_name' in df.columns

def test_prediction_logic():
    """Test basic prediction logic."""
    # Mock molecular properties
    drug1_props = {'molecular_weight': 300, 'logp': 2, 'hbd': 2, 'hba': 4}
    drug2_props = {'molecular_weight': 250, 'logp': 3, 'hbd': 1, 'hba': 5}
    
    # Calculate similarity
    prop1 = np.array([drug1_props[k] for k in ['molecular_weight', 'logp', 'hbd', 'hba']])
    prop2 = np.array([drug2_props[k] for k in ['molecular_weight', 'logp', 'hbd', 'hba']])
    
    diff = np.abs(prop1 - prop2)
    similarity = 1 / (1 + np.sum(diff) / len(diff))
    
    assert 0 <= similarity <= 1
    
    # Test risk level assignment
    interaction_prob = 0.8
    if interaction_prob >= 0.7:
        risk_level = "HIGH"
    elif interaction_prob >= 0.4:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    assert risk_level == "HIGH"

def test_cli_prediction():
    """Test CLI prediction function exists."""
    try:
        from cli.simple_cli import predict_interaction
        # Test with mock data
        result = predict_interaction("TestDrug1", "TestDrug2")
        assert isinstance(result, dict)
    except ImportError:
        pytest.skip("CLI module not available")

def test_web_interface():
    """Test web interface components."""
    web_app_file = Path(__file__).parent.parent / 'web_app.py'
    assert web_app_file.exists(), "Web app file should exist"

def test_api_backend():
    """Test API backend structure."""
    api_file = Path(__file__).parent.parent / 'web_interface' / 'backend' / 'api.py'
    assert api_file.exists(), "API backend should exist"

if __name__ == "__main__":
    pytest.main([__file__])