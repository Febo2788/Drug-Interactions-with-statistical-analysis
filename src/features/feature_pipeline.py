#!/usr/bin/env python3
"""
Feature engineering pipeline for drug-drug interaction prediction.

This script processes molecular data and builds graph representations
for GNN training with comprehensive feature extraction.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from features.molecular_features import MolecularFeatureExtractor
from features.graph_features import GraphFeatureExtractor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feature_engineering.log'),
            logging.StreamHandler()
        ]
    )


def run_feature_pipeline():
    """Run the complete feature engineering pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting feature engineering pipeline...")
    
    # Load raw data
    print("Loading raw datasets...")
    interactions_df = pd.read_csv('../../data/raw/sample_drug_interactions.csv')
    molecular_df = pd.read_csv('../../data/raw/drug_smiles.csv')
    
    # Initialize feature extractors
    mol_extractor = MolecularFeatureExtractor(use_rdkit=False)  # Use fallback methods
    graph_extractor = GraphFeatureExtractor()
    
    # 1. Extract molecular features
    print("\n1. Extracting molecular descriptors...")
    molecular_descriptors = mol_extractor.calculate_drug_properties(molecular_df)
    print(f"   Generated descriptors for {len(molecular_descriptors)} drugs")
    
    # 2. Generate molecular fingerprints
    print("\n2. Generating Morgan fingerprints...")
    fingerprints = mol_extractor.generate_morgan_fingerprints(molecular_df, radius=2, n_bits=512)
    print(f"   Generated {fingerprints.shape[1]-1} fingerprint features")
    
    # 3. Build interaction graph
    print("\n3. Building drug interaction graph...")
    graph_data = graph_extractor.build_interaction_graph(interactions_df, molecular_descriptors)
    print(f"   Graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
    print(f"   Network density: {graph_data['network_features']['density']:.3f}")
    print(f"   Average degree: {graph_data['network_features']['avg_degree']:.2f}")
    print(f"   Average clustering: {graph_data['network_features']['avg_clustering']:.3f}")
    
    # 4. Create adjacency matrices by interaction type
    print("\n4. Creating type-specific adjacency matrices...")
    adj_matrices_by_type = graph_extractor.create_adjacency_matrices_by_type(interactions_df)
    for interaction_type, matrix in adj_matrices_by_type.items():
        edge_count = np.sum(matrix) // 2
        print(f"   {interaction_type.capitalize()}: {edge_count} interactions")
    
    # 5. Save processed features
    print("\n5. Saving processed features...")
    
    # Ensure output directory exists
    Path('../../data/processed').mkdir(parents=True, exist_ok=True)
    
    # Save molecular features
    molecular_descriptors.to_csv('../../data/processed/molecular_descriptors.csv', index=False)
    fingerprints.to_csv('../../data/processed/molecular_fingerprints.csv', index=False)
    
    # Save graph data
    with open('../../data/processed/graph_data.pkl', 'wb') as f:
        pickle.dump({
            'graph_data': graph_data,
            'adj_matrices_by_type': adj_matrices_by_type,
            'molecular_descriptors': molecular_descriptors,
            'fingerprints': fingerprints
        }, f)
    
    # 6. Generate feature summary report
    print("\n6. Generating feature summary report...")
    
    feature_summary = {
        'molecular_features': {
            'num_drugs': len(molecular_descriptors),
            'descriptor_features': molecular_descriptors.shape[1] - 2,  # Excluding name and SMILES
            'fingerprint_features': fingerprints.shape[1] - 1,  # Excluding drug name
            'lipinski_compliant_drugs': molecular_descriptors['lipinski_compliant'].sum() if 'lipinski_compliant' in molecular_descriptors.columns else 'N/A',
            'avg_drug_likeness': molecular_descriptors['drug_likeness_score'].mean() if 'drug_likeness_score' in molecular_descriptors.columns else 'N/A'
        },
        'graph_features': graph_data['network_features'],
        'interaction_types': {k: int(np.sum(v) // 2) for k, v in adj_matrices_by_type.items()}
    }
    
    # Save feature summary
    with open('../../data/processed/feature_summary.pkl', 'wb') as f:
        pickle.dump(feature_summary, f)
    
    print("\n=== FEATURE ENGINEERING COMPLETE ===")
    print(f"Molecular descriptors: {feature_summary['molecular_features']['descriptor_features']} features")
    print(f"Molecular fingerprints: {feature_summary['molecular_features']['fingerprint_features']} features")
    print(f"Graph nodes: {feature_summary['graph_features']['num_nodes']}")
    print(f"Graph edges: {feature_summary['graph_features']['num_edges']}")
    print(f"Network density: {feature_summary['graph_features']['density']:.3f}")
    
    print(f"\nInteraction breakdown:")
    for interaction_type, count in feature_summary['interaction_types'].items():
        print(f"  {interaction_type.capitalize()}: {count}")
    
    print(f"\nAll processed features saved to data/processed/")
    
    return feature_summary


def main():
    """Main execution function."""
    setup_logging()
    
    try:
        summary = run_feature_pipeline()
        print(f"\nFeature engineering completed successfully!")
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()