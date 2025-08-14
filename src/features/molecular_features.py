"""
Molecular feature extraction from SMILES strings.

This module provides comprehensive molecular descriptor calculation
and fingerprint generation for drug molecules using RDKit (when available)
or fallback methods for development.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings

logger = logging.getLogger(__name__)


class MolecularFeatureExtractor:
    """
    Extract molecular features and descriptors from SMILES strings.
    
    Provides both RDKit-based extraction (when available) and fallback
    methods for development without full cheminformatics dependencies.
    """
    
    def __init__(self, use_rdkit: bool = True):
        """
        Initialize molecular feature extractor.
        
        Args:
            use_rdkit: Whether to use RDKit for feature extraction
        """
        self.use_rdkit = use_rdkit
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Try to import RDKit
        self.rdkit_available = False
        if use_rdkit:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
                self.Chem = Chem
                self.Descriptors = Descriptors
                self.Crippen = Crippen
                self.rdMolDescriptors = rdMolDescriptors
                self.rdkit_available = True
                self.logger.info("RDKit available - using full cheminformatics features")
            except ImportError:
                self.logger.warning("RDKit not available - using fallback methods")
    
    def extract_basic_descriptors(self, smiles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic molecular descriptors from SMILES.
        
        Args:
            smiles_df: DataFrame with 'smiles' column
            
        Returns:
            DataFrame with molecular descriptors
        """
        self.logger.info("Extracting basic molecular descriptors...")
        
        if self.rdkit_available:
            return self._extract_rdkit_descriptors(smiles_df)
        else:
            return self._extract_fallback_descriptors(smiles_df)
    
    def generate_morgan_fingerprints(self, smiles_df: pd.DataFrame, 
                                   radius: int = 2, n_bits: int = 1024) -> pd.DataFrame:
        """
        Generate Morgan fingerprints for molecular similarity.
        
        Args:
            smiles_df: DataFrame with SMILES strings
            radius: Morgan fingerprint radius
            n_bits: Number of bits in fingerprint
            
        Returns:
            DataFrame with fingerprint vectors
        """
        self.logger.info(f"Generating Morgan fingerprints (radius={radius}, bits={n_bits})...")
        
        if self.rdkit_available:
            return self._generate_rdkit_fingerprints(smiles_df, radius, n_bits)
        else:
            return self._generate_mock_fingerprints(smiles_df, n_bits)
    
    def calculate_drug_properties(self, smiles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive drug properties.
        
        Args:
            smiles_df: DataFrame with SMILES strings
            
        Returns:
            DataFrame with calculated properties
        """
        self.logger.info("Calculating comprehensive drug properties...")
        
        # Start with basic descriptors
        props_df = self.extract_basic_descriptors(smiles_df.copy())
        
        # Add Lipinski's Rule of Five compliance
        props_df = self._calculate_lipinski_compliance(props_df)
        
        # Add drug-likeness scores
        props_df = self._calculate_drug_likeness(props_df)
        
        return props_df
    
    def _extract_rdkit_descriptors(self, smiles_df: pd.DataFrame) -> pd.DataFrame:
        """Extract descriptors using RDKit (when available)."""
        descriptors = []
        
        for _, row in smiles_df.iterrows():
            try:
                mol = self.Chem.MolFromSmiles(row['smiles'])
                if mol is not None:
                    desc = {
                        'drug_name': row['drug_name'],
                        'smiles': row['smiles'],
                        'molecular_weight': self.Descriptors.MolWt(mol),
                        'logp': self.Descriptors.MolLogP(mol),
                        'hbd': self.Descriptors.NumHDonors(mol),
                        'hba': self.Descriptors.NumHAcceptors(mol),
                        'tpsa': self.Descriptors.TPSA(mol),
                        'rotatable_bonds': self.Descriptors.NumRotatableBonds(mol),
                        'aromatic_rings': self.Descriptors.NumAromaticRings(mol),
                        'aliphatic_rings': self.Descriptors.NumAliphaticRings(mol),
                        'formal_charge': self.Chem.rdmolops.GetFormalCharge(mol),
                        'num_heavy_atoms': mol.GetNumHeavyAtoms()
                    }
                else:
                    # Invalid SMILES
                    desc = self._get_invalid_descriptor_dict(row)
                    
            except Exception as e:
                self.logger.warning(f"Error processing {row['drug_name']}: {e}")
                desc = self._get_invalid_descriptor_dict(row)
                
            descriptors.append(desc)
        
        return pd.DataFrame(descriptors)
    
    def _extract_fallback_descriptors(self, smiles_df: pd.DataFrame) -> pd.DataFrame:
        """Extract descriptors using fallback methods (no RDKit)."""
        self.logger.info("Using fallback descriptor calculation...")
        
        descriptors = []
        for _, row in smiles_df.iterrows():
            # Simple SMILES-based approximations
            smiles = row['smiles']
            desc = {
                'drug_name': row['drug_name'],
                'smiles': smiles,
                'molecular_weight': row.get('molecular_weight', self._estimate_mw_from_smiles(smiles)),
                'logp': row.get('logp', self._estimate_logp_from_smiles(smiles)),
                'hbd': row.get('hbd', smiles.count('O') + smiles.count('N')),  # Rough approximation
                'hba': row.get('hba', smiles.count('O') + smiles.count('N')),
                'tpsa': self._estimate_tpsa(smiles),
                'rotatable_bonds': smiles.count('C') // 4,  # Very rough approximation
                'aromatic_rings': smiles.count('c'),  # Lowercase c indicates aromatic
                'aliphatic_rings': smiles.count('C') // 6,  # Rough approximation
                'formal_charge': smiles.count('+') - smiles.count('-'),
                'num_heavy_atoms': len([c for c in smiles if c.isalpha() and c != 'H'])
            }
            descriptors.append(desc)
            
        return pd.DataFrame(descriptors)
    
    def _generate_rdkit_fingerprints(self, smiles_df: pd.DataFrame, 
                                   radius: int, n_bits: int) -> pd.DataFrame:
        """Generate Morgan fingerprints using RDKit."""
        fingerprints = []
        
        for _, row in smiles_df.iterrows():
            try:
                mol = self.Chem.MolFromSmiles(row['smiles'])
                if mol is not None:
                    fp = self.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        mol, radius, nBits=n_bits
                    )
                    fp_array = np.array(fp)
                else:
                    fp_array = np.zeros(n_bits)
            except:
                fp_array = np.zeros(n_bits)
                
            fp_dict = {'drug_name': row['drug_name']}
            fp_dict.update({f'fp_bit_{i}': fp_array[i] for i in range(n_bits)})
            fingerprints.append(fp_dict)
            
        return pd.DataFrame(fingerprints)
    
    def _generate_mock_fingerprints(self, smiles_df: pd.DataFrame, n_bits: int) -> pd.DataFrame:
        """Generate mock fingerprints for development (no RDKit)."""
        fingerprints = []
        
        for _, row in smiles_df.iterrows():
            # Create deterministic "fingerprint" based on SMILES string
            smiles = row['smiles']
            np.random.seed(hash(smiles) % 2**32)  # Deterministic seed
            fp_array = np.random.binomial(1, 0.1, n_bits)  # Sparse binary vector
            
            fp_dict = {'drug_name': row['drug_name']}
            fp_dict.update({f'fp_bit_{i}': fp_array[i] for i in range(n_bits)})
            fingerprints.append(fp_dict)
            
        return pd.DataFrame(fingerprints)
    
    def _calculate_lipinski_compliance(self, props_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Lipinski's Rule of Five compliance."""
        props_df['lipinski_violations'] = (
            (props_df['molecular_weight'] > 500).astype(int) +
            (props_df['logp'] > 5).astype(int) +
            (props_df['hbd'] > 5).astype(int) +
            (props_df['hba'] > 10).astype(int)
        )
        
        props_df['lipinski_compliant'] = props_df['lipinski_violations'] <= 1
        return props_df
    
    def _calculate_drug_likeness(self, props_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple drug-likeness score."""
        # Simple drug-likeness based on common ranges
        score = 0
        
        # Molecular weight penalty
        mw_penalty = np.abs(props_df['molecular_weight'] - 350) / 350
        
        # LogP penalty
        logp_penalty = np.abs(props_df['logp'] - 2.5) / 2.5
        
        # Drug-likeness score (0-1, higher is better)
        props_df['drug_likeness_score'] = np.exp(-(mw_penalty + logp_penalty))
        
        return props_df
    
    def _get_invalid_descriptor_dict(self, row: pd.Series) -> Dict:
        """Return dict with NaN values for invalid molecules."""
        return {
            'drug_name': row['drug_name'],
            'smiles': row['smiles'],
            'molecular_weight': np.nan,
            'logp': np.nan,
            'hbd': np.nan,
            'hba': np.nan,
            'tpsa': np.nan,
            'rotatable_bonds': np.nan,
            'aromatic_rings': np.nan,
            'aliphatic_rings': np.nan,
            'formal_charge': np.nan,
            'num_heavy_atoms': np.nan
        }
    
    def _estimate_mw_from_smiles(self, smiles: str) -> float:
        """Rough MW estimation from SMILES string."""
        # Very rough approximation based on atom counts
        c_count = smiles.count('C') + smiles.count('c')
        n_count = smiles.count('N') + smiles.count('n')
        o_count = smiles.count('O') + smiles.count('o')
        s_count = smiles.count('S') + smiles.count('s')
        
        return c_count * 12 + n_count * 14 + o_count * 16 + s_count * 32 + 50  # +50 for H
    
    def _estimate_logp_from_smiles(self, smiles: str) -> float:
        """Rough LogP estimation from SMILES string."""
        # Very rough approximation
        carbon_contribution = (smiles.count('C') + smiles.count('c')) * 0.2
        nitrogen_penalty = (smiles.count('N') + smiles.count('n')) * -0.5
        oxygen_penalty = (smiles.count('O') + smiles.count('o')) * -0.3
        
        return carbon_contribution + nitrogen_penalty + oxygen_penalty
    
    def _estimate_tpsa(self, smiles: str) -> float:
        """Rough TPSA estimation."""
        # Very rough approximation
        n_count = smiles.count('N') + smiles.count('n')
        o_count = smiles.count('O') + smiles.count('o')
        
        return n_count * 23.8 + o_count * 20.2  # Rough polar surface area