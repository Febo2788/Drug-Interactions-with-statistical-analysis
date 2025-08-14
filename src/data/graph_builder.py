"""
Graph construction utilities for drug-drug interaction networks.

This module provides functionality to build graph structures from drug
interaction data for use with Graph Neural Networks.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build graph structures from drug interaction data.
    
    Creates networkx graphs suitable for GNN training with proper
    node and edge attributes.
    """
    
    def __init__(self):
        """Initialize graph builder."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def build_interaction_graph(self, interactions_df: pd.DataFrame) -> nx.Graph:
        """
        Build drug interaction network from interaction data.
        
        Args:
            interactions_df: DataFrame with drug interaction data
            
        Returns:
            NetworkX graph with drug interactions
        """
        self.logger.info("Building drug interaction network...")
        
        # Create undirected graph
        G = nx.Graph()
        
        # Add edges for each interaction
        for _, row in interactions_df.iterrows():
            drug1, drug2 = row['drug1_name'], row['drug2_name']
            
            # Add edge with attributes
            G.add_edge(drug1, drug2, 
                      severity=row.get('severity', 'Unknown'),
                      mechanism=row.get('mechanism', 'Unknown'),
                      evidence=row.get('evidence_level', 'Low'))
        
        self.logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G