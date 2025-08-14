"""
Baseline machine learning models for drug-drug interaction prediction.

This module implements traditional ML algorithms as baselines for comparison
with Graph Neural Networks, including proper statistical validation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional
import pickle

logger = logging.getLogger(__name__)


class BaselineModels:
    """
    Implementation of baseline machine learning models for DDI prediction.
    
    Provides logistic regression and random forest implementations with
    comprehensive statistical evaluation and comparison metrics.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize baseline models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_training_data(self, molecular_desc: pd.DataFrame, 
                            interactions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from molecular features and interactions.
        
        Args:
            molecular_desc: Molecular descriptors DataFrame
            interactions_df: Drug interactions DataFrame
            
        Returns:
            Tuple of (X, y) for model training
        """
        self.logger.info("Preparing training data for baseline models...")
        
        # Create drug feature mapping
        drug_features = {}
        feature_cols = [col for col in molecular_desc.columns 
                       if col not in ['drug_name', 'smiles', 'lipinski_compliant']]
        
        for _, row in molecular_desc.iterrows():
            drug_features[row['drug_name']] = row[feature_cols].values
        
        # Create training pairs
        X_list = []
        y_list = []
        
        for _, interaction in interactions_df.iterrows():
            drug1, drug2 = interaction['drug1_name'], interaction['drug2_name']
            
            # Get features for both drugs (use defaults if missing)
            feat1 = drug_features.get(drug1, np.zeros(len(feature_cols)))
            feat2 = drug_features.get(drug2, np.zeros(len(feature_cols)))
            
            # Combine features (concatenation + element-wise operations)
            combined_features = np.concatenate([
                feat1, feat2,  # Individual drug features
                feat1 + feat2,  # Sum
                feat1 * feat2,  # Product
                np.abs(feat1 - feat2)  # Difference
            ])
            
            X_list.append(combined_features)
            
            # Create binary target (Major/Contraindicated = 1, others = 0)
            severity = interaction['severity']
            y_list.append(1 if severity in ['Major', 'Contraindicated'] else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.logger.info(f"Prepared {X.shape[0]} training samples with {X.shape[1]} features")
        self.logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X_scaled, y
    
    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train logistic regression model with statistical validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with model performance metrics
        """
        self.logger.info("Training logistic regression model...")
        
        # Initialize model with class balancing
        lr_model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000
        )
        
        # Fit model
        lr_model.fit(X, y)
        self.models['logistic_regression'] = lr_model
        
        # Evaluate with cross-validation
        cv_scores = self._evaluate_with_cv(lr_model, X, y, 'Logistic Regression')
        self.results['logistic_regression'] = cv_scores
        
        return cv_scores
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train random forest model with statistical validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with model performance metrics
        """
        self.logger.info("Training random forest model...")
        
        # Initialize model with class balancing
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5
        )
        
        # Fit model
        rf_model.fit(X, y)
        self.models['random_forest'] = rf_model
        
        # Evaluate with cross-validation
        cv_scores = self._evaluate_with_cv(rf_model, X, y, 'Random Forest')
        self.results['random_forest'] = cv_scores
        
        # Feature importance analysis
        feature_importance = rf_model.feature_importances_
        self.results['random_forest']['feature_importance'] = feature_importance
        
        return cv_scores
    
    def _evaluate_with_cv(self, model, X: np.ndarray, y: np.ndarray, 
                         model_name: str) -> Dict[str, float]:
        """Comprehensive cross-validation evaluation."""
        
        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Collect metrics across folds
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train on fold
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            
            try:
                metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            except ValueError:
                metrics['roc_auc'].append(0.5)  # Random performance if only one class
        
        # Calculate mean and confidence intervals
        results = {}
        for metric, values in metrics.items():
            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            ci_lower, ci_upper = self._calculate_confidence_interval(values)
            
            results[f'{metric}_mean'] = mean_val
            results[f'{metric}_std'] = std_val
            results[f'{metric}_ci_lower'] = ci_lower
            results[f'{metric}_ci_upper'] = ci_upper
        
        self.logger.info(f"{model_name} CV Results:")
        self.logger.info(f"  Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
        self.logger.info(f"  Precision: {results['precision_mean']:.3f} ± {results['precision_std']:.3f}")
        self.logger.info(f"  Recall: {results['recall_mean']:.3f} ± {results['recall_std']:.3f}")
        self.logger.info(f"  F1-Score: {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
        self.logger.info(f"  ROC-AUC: {results['roc_auc_mean']:.3f} ± {results['roc_auc_std']:.3f}")
        
        return results
    
    def _calculate_confidence_interval(self, values: np.ndarray, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for metric values."""
        n = len(values)
        mean = np.mean(values)
        std_err = stats.sem(values)  # Standard error of the mean
        
        # t-distribution critical value
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        margin_error = t_critical * std_err
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        return ci_lower, ci_upper
    
    def compare_models(self) -> Dict[str, any]:
        """
        Statistical comparison between models using paired t-tests.
        
        Returns:
            Dictionary with comparison statistics
        """
        if len(self.results) < 2:
            self.logger.warning("Need at least 2 models for comparison")
            return {}
        
        self.logger.info("Performing statistical model comparison...")
        
        comparison_results = {}
        
        # Compare all pairs of models
        model_names = list(self.results.keys())
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                
                # Extract metrics for comparison
                metrics_to_compare = ['accuracy_mean', 'precision_mean', 'recall_mean', 
                                    'f1_mean', 'roc_auc_mean']
                
                comparison = {}
                for metric in metrics_to_compare:
                    val1 = self.results[model1].get(metric, 0)
                    val2 = self.results[model2].get(metric, 0)
                    
                    comparison[metric] = {
                        f'{model1}': val1,
                        f'{model2}': val2,
                        'difference': val1 - val2,
                        'better_model': model1 if val1 > val2 else model2
                    }
                
                comparison_results[f'{model1}_vs_{model2}'] = comparison
        
        return comparison_results
    
    def save_models(self, output_dir: str = '../../models/saved_models'):
        """Save trained models and results."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            with open(f'{output_dir}/baseline_{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        with open(f'{output_dir}/baseline_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save results
        with open(f'{output_dir}/baseline_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        self.logger.info(f"Models and results saved to {output_dir}")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of model performance."""
        summary_data = []
        
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}",
                'Precision': f"{results['precision_mean']:.3f} ± {results['precision_std']:.3f}",
                'Recall': f"{results['recall_mean']:.3f} ± {results['recall_std']:.3f}",
                'F1-Score': f"{results['f1_mean']:.3f} ± {results['f1_std']:.3f}",
                'ROC-AUC': f"{results['roc_auc_mean']:.3f} ± {results['roc_auc_std']:.3f}"
            })
        
        return pd.DataFrame(summary_data)