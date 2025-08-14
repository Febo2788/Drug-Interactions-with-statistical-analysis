"""
Graph feature extraction and network construction for drug interactions.

This module provides functionality to build and analyze drug interaction
networks with proper node and edge features for GNN training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """
    Extract graph-based features from drug interaction networks.
    
    Creates network representations suitable for Graph Neural Networks
    with comprehensive node and edge attributes.
    """
    
    def __init__(self):
        """Initialize graph feature extractor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.drug_to_idx = {}
        self.idx_to_drug = {}
        
    def build_interaction_graph(self, interactions_df: pd.DataFrame, 
                              molecular_df: pd.DataFrame) -> Dict[str, any]:
        """
        Build drug interaction graph with molecular features.
        
        Args:
            interactions_df: Drug interaction data
            molecular_df: Molecular property data
            
        Returns:
            Dictionary containing graph structure and features
        """
        self.logger.info("Building drug interaction graph...")
        
        # Create drug index mapping
        all_drugs = set(interactions_df['drug1_name'].tolist() + 
                       interactions_df['drug2_name'].tolist())
        
        self.drug_to_idx = {drug: idx for idx, drug in enumerate(sorted(all_drugs))}
        self.idx_to_drug = {idx: drug for drug, idx in self.drug_to_idx.items()}
        
        self.logger.info(f"Graph contains {len(all_drugs)} nodes (drugs)")
        
        # Build adjacency matrix and edge features
        adj_matrix, edge_features, edge_indices = self._build_adjacency_matrix(interactions_df)
        
        # Build node features
        node_features = self._build_node_features(molecular_df, all_drugs)
        
        # Calculate network topology features
        network_features = self._calculate_network_features(adj_matrix)
        
        graph_data = {
            'adjacency_matrix': adj_matrix,
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_indices': edge_indices,
            'network_features': network_features,
            'drug_to_idx': self.drug_to_idx,
            'idx_to_drug': self.idx_to_drug,
            'num_nodes': len(all_drugs),
            'num_edges': len(edge_indices[0])
        }
        
        self.logger.info(f"Graph built: {len(all_drugs)} nodes, {len(edge_indices[0])} edges")
        return graph_data
    
    def _build_adjacency_matrix(self, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, Dict, Tuple]:
        """Build adjacency matrix and edge features."""
        num_drugs = len(self.drug_to_idx)
        adj_matrix = np.zeros((num_drugs, num_drugs), dtype=int)
        
        edge_features = defaultdict(list)
        edge_indices = ([], [])  # Source and target node indices
        
        # Encode severity levels
        severity_encoding = {
            'Minor': 1, 'Moderate': 2, 'Major': 3, 'Contraindicated': 4
        }
        
        # Encode evidence levels
        evidence_encoding = {
            'Low': 1, 'Medium': 2, 'High': 3
        }
        
        for _, row in interactions_df.iterrows():
            drug1_idx = self.drug_to_idx[row['drug1_name']]
            drug2_idx = self.drug_to_idx[row['drug2_name']]
            
            # Set adjacency (undirected graph)
            adj_matrix[drug1_idx, drug2_idx] = 1
            adj_matrix[drug2_idx, drug1_idx] = 1
            
            # Add edge features for both directions
            severity_code = severity_encoding.get(row['severity'], 1)
            evidence_code = evidence_encoding.get(row['evidence_level'], 1)
            
            for src, dst in [(drug1_idx, drug2_idx), (drug2_idx, drug1_idx)]:
                edge_indices[0].append(src)
                edge_indices[1].append(dst)
                edge_features['severity'].append(severity_code)
                edge_features['evidence'].append(evidence_code)
                edge_features['mechanism'].append(self._encode_mechanism(row['mechanism']))
        
        return adj_matrix, dict(edge_features), edge_indices
    
    def _build_node_features(self, molecular_df: pd.DataFrame, all_drugs: set) -> np.ndarray:
        """Build node feature matrix from molecular properties."""
        num_drugs = len(all_drugs)
        
        # Define feature columns (handle missing molecular data gracefully)
        feature_cols = ['molecular_weight', 'logp', 'hbd', 'hba']
        num_features = len(feature_cols)
        
        node_features = np.zeros((num_drugs, num_features))
        
        # Create mapping of drug names to molecular data
        mol_dict = {}
        if not molecular_df.empty:
            mol_dict = molecular_df.set_index('drug_name').to_dict('index')
        
        for drug, idx in self.drug_to_idx.items():
            if drug in mol_dict:
                # Use actual molecular data
                mol_data = mol_dict[drug]
                for i, col in enumerate(feature_cols):
                    node_features[idx, i] = mol_data.get(col, 0.0)
            else:
                # Use default values for drugs without molecular data
                node_features[idx] = [300.0, 2.0, 2.0, 4.0]  # Typical drug-like values
                
        # Standardize features
        node_features = self._standardize_features(node_features)
        
        return node_features
    
    def _calculate_network_features(self, adj_matrix: np.ndarray) -> Dict[str, any]:
        """Calculate network topology features."""
        # Basic network statistics
        num_nodes = adj_matrix.shape[0]
        num_edges = np.sum(adj_matrix) // 2  # Undirected graph
        
        # Degree statistics
        degrees = np.sum(adj_matrix, axis=1)
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        
        # Density
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0
        
        # Clustering coefficient approximation
        clustering_coeffs = []
        for i in range(num_nodes):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) > 1:
                # Count edges among neighbors
                edges_among_neighbors = 0
                for j in neighbors:
                    for k in neighbors:
                        if j < k and adj_matrix[j, k] == 1:
                            edges_among_neighbors += 1
                
                max_edges_neighbors = len(neighbors) * (len(neighbors) - 1) / 2
                clustering = edges_among_neighbors / max_edges_neighbors if max_edges_neighbors > 0 else 0
                clustering_coeffs.append(clustering)
        
        avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'density': density,
            'avg_clustering': avg_clustering,
            'degree_distribution': degrees.tolist()
        }
    
    def _encode_mechanism(self, mechanism: str) -> int:
        """Encode interaction mechanism as integer."""
        mechanism_encoding = {
            'Pharmacodynamic synergism': 1,
            'CYP3A4 inhibition': 2,
            'Renal clearance reduction': 3,
            'P-glycoprotein inhibition': 4,
            'CYP2C19 interaction': 5,
            'Reduced renal clearance': 6,
            'Nitric oxide pathway': 7,
            'CYP1A2 inhibition': 8,
            'CYP2C19 inhibition': 9,
            'Serotonergic synergism': 10,
            'TPMT enzyme inhibition': 11,
            'CYP3A4 induction': 12,
            'Additive glucose lowering': 13
        }
        
        return mechanism_encoding.get(mechanism, 0)
    
    def _standardize_features(self, features: np.ndarray) -> np.ndarray:
        """Standardize feature matrix (z-score normalization)."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        return (features - mean) / std
    
    def create_adjacency_matrices_by_type(self, interactions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create separate adjacency matrices for different interaction types."""
        num_drugs = len(self.drug_to_idx)
        
        # Create matrices for each severity level
        matrices = {
            'minor': np.zeros((num_drugs, num_drugs), dtype=int),
            'moderate': np.zeros((num_drugs, num_drugs), dtype=int),
            'major': np.zeros((num_drugs, num_drugs), dtype=int),
            'contraindicated': np.zeros((num_drugs, num_drugs), dtype=int)
        }
        
        for _, row in interactions_df.iterrows():
            drug1_idx = self.drug_to_idx[row['drug1_name']]
            drug2_idx = self.drug_to_idx[row['drug2_name']]
            
            severity = row['severity'].lower()
            if severity in matrices:
                matrices[severity][drug1_idx, drug2_idx] = 1
                matrices[severity][drug2_idx, drug1_idx] = 1
        
        self.logger.info("Created separate adjacency matrices by interaction type")
        return matrices