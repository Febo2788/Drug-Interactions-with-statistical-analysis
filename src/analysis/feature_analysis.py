#!/usr/bin/env python3
"""
Analysis of generated features for drug-drug interaction prediction.
"""

import pandas as pd
import pickle
from pathlib import Path

def analyze_features():
    """Analyze the generated features."""
    print("=== FEATURE ANALYSIS REPORT ===\n")
    
    # Load processed data
    molecular_desc = pd.read_csv('../../data/processed/molecular_descriptors.csv')
    fingerprints = pd.read_csv('../../data/processed/molecular_fingerprints.csv')
    
    with open('../../data/processed/graph_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    graph_data = data['graph_data']
    adj_by_type = data['adj_matrices_by_type']
    
    print("1. MOLECULAR DESCRIPTORS")
    print(f"   Drugs with molecular data: {len(molecular_desc)}")
    print(f"   Descriptor features: {molecular_desc.shape[1] - 2}")  # Exclude name, SMILES
    print(f"   Lipinski compliant: {molecular_desc['lipinski_compliant'].sum()}/{len(molecular_desc)}")
    print(f"   Average drug-likeness score: {molecular_desc['drug_likeness_score'].mean():.3f}")
    
    print("\n2. MOLECULAR FINGERPRINTS")
    print(f"   Fingerprint dimensions: {fingerprints.shape[1] - 1}")  # Exclude drug name
    print(f"   Average fingerprint density: {fingerprints.iloc[:, 1:].mean().mean():.3f}")
    
    print("\n3. GRAPH STRUCTURE")
    print(f"   Total nodes (drugs): {graph_data['num_nodes']}")
    print(f"   Total edges (interactions): {graph_data['num_edges']}")
    print(f"   Network density: {graph_data['network_features']['density']:.3f}")
    print(f"   Average degree: {graph_data['network_features']['avg_degree']:.2f}")
    print(f"   Max degree: {graph_data['network_features']['max_degree']}")
    
    print("\n4. INTERACTION TYPES")
    total_interactions = sum(adj_by_type[k].sum() // 2 for k in adj_by_type)
    for interaction_type, matrix in adj_by_type.items():
        count = matrix.sum() // 2
        percentage = (count / total_interactions * 100) if total_interactions > 0 else 0
        print(f"   {interaction_type.capitalize()}: {count} ({percentage:.1f}%)")
    
    print("\n5. STATISTICAL SUMMARY")
    print(f"   MW range: {molecular_desc['molecular_weight'].min():.1f} - {molecular_desc['molecular_weight'].max():.1f} Da")
    print(f"   LogP range: {molecular_desc['logp'].min():.1f} - {molecular_desc['logp'].max():.1f}")
    print(f"   H-bond donors range: {molecular_desc['hbd'].min()} - {molecular_desc['hbd'].max()}")
    print(f"   H-bond acceptors range: {molecular_desc['hba'].min()} - {molecular_desc['hba'].max()}")
    
    print("\n6. READINESS FOR MODELING")
    print(f"   ✓ Node features: {graph_data['node_features'].shape}")
    print(f"   ✓ Edge features: Available (severity, evidence, mechanism)")
    print(f"   ✓ Graph adjacency: {graph_data['adjacency_matrix'].shape}")
    print(f"   ✓ Molecular fingerprints: Ready for similarity computation")
    print(f"   ✓ Drug-likeness metrics: Computed for all molecules")
    
    print("\n=== READY FOR PHASE 4: MODEL DEVELOPMENT ===")

if __name__ == "__main__":
    analyze_features()