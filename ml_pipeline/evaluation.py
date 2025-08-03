"""
Evaluation Module for ODE Generation Models

Provides tools for evaluating generated ODEs and model performance.
"""

import numpy as np
import pandas as pd
import torch
import sympy as sp
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import json
from pathlib import Path

from verification.verifier import ODEVerifier
from core.symbols import SYMBOLS
from utils.features import FeatureExtractor

logger = logging.getLogger(__name__)


class ODEEvaluator:
    """
    Comprehensive evaluator for ODE generation models
    """
    
    def __init__(self, 
                 verifier_config: Optional[Dict] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator
        
        Args:
            verifier_config: Configuration for ODE verifier
            device: Device for model evaluation
        """
        self.device = device
        self.verifier = ODEVerifier(verifier_config or {})
        self.feature_extractor = FeatureExtractor()
        self.metrics_history = []
        
    def evaluate_model(self, 
                      model: torch.nn.Module,
                      test_dataset: torch.utils.data.Dataset,
                      batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate a trained model on test dataset
        
        Args:
            model: Trained PyTorch model
            test_dataset: Test dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        model.to(self.device)
        
        dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        all_predictions = []
        all_targets = []
        all_complexities = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                numeric_features = batch['numeric_features'].to(self.device)
                generator_id = batch['generator_id'].to(self.device)
                function_id = batch['function_id'].to(self.device)
                verified = batch['verified'].to(self.device)
                
                # Forward pass
                outputs = model(numeric_features, generator_id, function_id)
                
                # Collect predictions
                verification_preds = (outputs['verification'].squeeze() > 0.5).cpu().numpy()
                all_predictions.extend(verification_preds)
                all_targets.extend(batch['verified'].numpy())
                
                if 'complexity' in outputs:
                    all_complexities.extend(outputs['complexity'].squeeze().cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': len(all_targets),
            'positive_samples': sum(all_targets),
            'negative_samples': len(all_targets) - sum(all_targets)
        }
        
        # Add complexity metrics if available
        if all_complexities:
            metrics['complexity_mae'] = np.mean(np.abs(
                np.array(all_complexities) - 
                [batch['numeric_features'][i, 0] for i in range(len(all_complexities))]
            ))
        
        self.metrics_history.append(metrics)
        return metrics
    
    def evaluate_generated_odes(self, 
                               generated_odes: List[str],
                               reference_dataset: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate a list of generated ODEs
        
        Args:
            generated_odes: List of generated ODE strings
            reference_dataset: Optional reference dataset for comparison
            
        Returns:
            Evaluation results
        """
        results = {
            'total': len(generated_odes),
            'valid': 0,
            'verified': 0,
            'novel': 0,
            'diverse': 0,
            'complexity_stats': {},
            'verification_details': [],
            'novelty_scores': []
        }
        
        valid_odes = []
        complexities = []
        
        for i, ode_str in enumerate(generated_odes):
            try:
                # Parse ODE
                ode = sp.sympify(ode_str)
                
                # Check if valid equation
                if not isinstance(ode, sp.Eq):
                    continue
                
                results['valid'] += 1
                valid_odes.append(ode)
                
                # Analyze complexity
                complexity = len(str(ode))
                complexities.append(complexity)
                
                # Try to verify (need solution for verification)
                # This is a simplified check - in practice you'd need the solution
                properties = self.verifier.analyze_ode_properties(ode)
                
                results['verification_details'].append({
                    'ode': str(ode),
                    'complexity': complexity,
                    'properties': properties
                })
                
            except Exception as e:
                logger.debug(f"Failed to parse ODE {i}: {e}")
                continue
        
        # Calculate complexity statistics
        if complexities:
            results['complexity_stats'] = {
                'mean': np.mean(complexities),
                'std': np.std(complexities),
                'min': min(complexities),
                'max': max(complexities),
                'median': np.median(complexities)
            }
        
        # Calculate diversity (unique ODEs)
        unique_odes = len(set(str(ode) for ode in valid_odes))
        results['diverse'] = unique_odes / len(valid_odes) if valid_odes else 0
        
        # Check novelty against reference dataset
        if reference_dataset is not None:
            novelty_detector = NoveltyDetector(reference_dataset)
            for ode in valid_odes:
                novelty_score = novelty_detector.compute_novelty_score(str(ode))
                results['novelty_scores'].append(novelty_score)
                if novelty_score > 0.8:  # Threshold for novel
                    results['novel'] += 1
        
        return results
    
    def evaluate_generation_quality(self,
                                   model,
                                   n_samples: int = 100,
                                   generators: List[str] = None,
                                   functions: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate quality of ODEs generated by a model
        
        Args:
            model: Trained generation model
            n_samples: Number of samples to generate
            generators: List of generator names to use
            functions: List of function names to use
            
        Returns:
            Quality metrics
        """
        if generators is None:
            generators = ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3']
        
        if functions is None:
            functions = ['identity', 'exponential', 'sine', 'cosine']
        
        generated_odes = []
        generation_times = []
        
        model.eval()
        
        for _ in range(n_samples):
            # Random generator and function
            gen = np.random.choice(generators)
            func = np.random.choice(functions)
            
            # Generate ODE
            start_time = time.time()
            
            try:
                if hasattr(model, 'generate'):
                    ode = model.generate(generator=gen, function=func)
                    generated_odes.append(ode)
                    generation_times.append(time.time() - start_time)
            except Exception as e:
                logger.debug(f"Generation failed: {e}")
                continue
        
        # Evaluate generated ODEs
        evaluation = self.evaluate_generated_odes(generated_odes)
        
        # Add timing statistics
        if generation_times:
            evaluation['generation_time_stats'] = {
                'mean': np.mean(generation_times),
                'std': np.std(generation_times),
                'total': sum(generation_times)
            }
        
        return evaluation
    
    def compare_models(self,
                      models: Dict[str, torch.nn.Module],
                      test_dataset: torch.utils.data.Dataset) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model_name -> model
            test_dataset: Test dataset
            
        Returns:
            Comparison DataFrame
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            metrics = self.evaluate_model(model, test_dataset)
            metrics['model'] = model_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def save_evaluation_report(self, 
                              filepath: str,
                              metrics: Dict[str, Any],
                              model_info: Optional[Dict] = None):
        """
        Save detailed evaluation report
        
        Args:
            filepath: Path to save report
            metrics: Evaluation metrics
            model_info: Optional model information
        """
        report = {
            'evaluation_date': str(datetime.now()),
            'metrics': metrics,
            'model_info': model_info or {},
            'verifier_config': self.verifier.__dict__,
            'metrics_history': self.metrics_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {filepath}")


class NoveltyDetector:
    """
    Detect novel ODEs compared to reference dataset
    """
    
    def __init__(self, reference_dataset: pd.DataFrame):
        """
        Initialize novelty detector
        
        Args:
            reference_dataset: DataFrame with reference ODEs
        """
        self.reference_dataset = reference_dataset
        self._build_reference_features()
    
    def _build_reference_features(self):
        """Build feature representations of reference ODEs"""
        self.reference_features = []
        self.reference_odes = []
        
        for _, row in self.reference_dataset.iterrows():
            ode_str = row.get('ode_symbolic', '')
            if ode_str:
                self.reference_odes.append(ode_str)
                
                # Extract features (simplified)
                features = self._extract_simple_features(ode_str)
                self.reference_features.append(features)
    
    def _extract_simple_features(self, ode_str: str) -> np.ndarray:
        """Extract simple features from ODE string"""
        features = []
        
        # Length
        features.append(len(ode_str))
        
        # Character counts
        features.append(ode_str.count('+'))
        features.append(ode_str.count('-'))
        features.append(ode_str.count('*'))
        features.append(ode_str.count('/'))
        features.append(ode_str.count('^'))
        features.append(ode_str.count('('))
        features.append(ode_str.count('sin'))
        features.append(ode_str.count('cos'))
        features.append(ode_str.count('exp'))
        features.append(ode_str.count('log'))
        
        # Derivative counts
        features.append(ode_str.count("y'"))
        features.append(ode_str.count("y''"))
        
        return np.array(features)
    
    def compute_novelty_score(self, ode_str: str) -> float:
        """
        Compute novelty score for an ODE
        
        Args:
            ode_str: ODE string
            
        Returns:
            Novelty score (0-1, higher is more novel)
        """
        # Check exact match first
        if ode_str in self.reference_odes:
            return 0.0
        
        # Extract features
        features = self._extract_simple_features(ode_str)
        
        # Compute distances to all reference ODEs
        min_distance = float('inf')
        
        for ref_features in self.reference_features:
            # Euclidean distance
            distance = np.linalg.norm(features - ref_features)
            min_distance = min(min_distance, distance)
        
        # Convert to novelty score (sigmoid-like)
        novelty_score = 1 - np.exp(-min_distance / 100)
        
        return novelty_score
    
    def find_similar_odes(self, ode_str: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar ODEs in reference dataset
        
        Args:
            ode_str: ODE string
            top_k: Number of similar ODEs to return
            
        Returns:
            List of (ode, similarity_score) tuples
        """
        features = self._extract_simple_features(ode_str)
        
        similarities = []
        
        for i, (ref_ode, ref_features) in enumerate(
            zip(self.reference_odes, self.reference_features)
        ):
            # Cosine similarity
            similarity = np.dot(features, ref_features) / (
                np.linalg.norm(features) * np.linalg.norm(ref_features) + 1e-8
            )
            similarities.append((ref_ode, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class PerformanceAnalyzer:
    """
    Analyze model performance across different ODE types
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_by_generator(self, 
                            predictions: List[bool],
                            targets: List[bool],
                            generators: List[str]) -> pd.DataFrame:
        """Analyze performance by generator type"""
        df = pd.DataFrame({
            'prediction': predictions,
            'target': targets,
            'generator': generators
        })
        
        results = []
        
        for gen in df['generator'].unique():
            gen_df = df[df['generator'] == gen]
            
            accuracy = accuracy_score(gen_df['target'], gen_df['prediction'])
            precision, recall, f1, _ = precision_recall_fscore_support(
                gen_df['target'], 
                gen_df['prediction'],
                average='binary'
            )
            
            results.append({
                'generator': gen,
                'count': len(gen_df),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        return pd.DataFrame(results)
    
    def analyze_by_complexity(self,
                             predictions: List[bool],
                             targets: List[bool],
                             complexities: List[int],
                             bins: int = 5) -> pd.DataFrame:
        """Analyze performance by complexity bins"""
        df = pd.DataFrame({
            'prediction': predictions,
            'target': targets,
            'complexity': complexities
        })
        
        # Create complexity bins
        df['complexity_bin'] = pd.qcut(df['complexity'], bins, labels=False)
        
        results = []
        
        for bin_idx in range(bins):
            bin_df = df[df['complexity_bin'] == bin_idx]
            
            if len(bin_df) > 0:
                accuracy = accuracy_score(bin_df['target'], bin_df['prediction'])
                
                results.append({
                    'complexity_bin': bin_idx,
                    'min_complexity': bin_df['complexity'].min(),
                    'max_complexity': bin_df['complexity'].max(),
                    'count': len(bin_df),
                    'accuracy': accuracy
                })
        
        return pd.DataFrame(results)
    
    def generate_performance_report(self, 
                                   model_name: str,
                                   metrics: Dict[str, Any]) -> str:
        """Generate a formatted performance report"""
        report = f"""
# ODE Generation Model Performance Report

## Model: {model_name}
## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Overall Performance
- Accuracy: {metrics.get('accuracy', 0):.4f}
- Precision: {metrics.get('precision', 0):.4f}
- Recall: {metrics.get('recall', 0):.4f}
- F1 Score: {metrics.get('f1_score', 0):.4f}

### Dataset Statistics
- Total Samples: {metrics.get('total_samples', 0)}
- Positive Samples: {metrics.get('positive_samples', 0)}
- Negative Samples: {metrics.get('negative_samples', 0)}

### Generation Quality
- Valid ODEs: {metrics.get('valid', 0)} / {metrics.get('total', 0)}
- Verified ODEs: {metrics.get('verified', 0)}
- Novel ODEs: {metrics.get('novel', 0)}
- Diversity Score: {metrics.get('diverse', 0):.4f}

### Complexity Analysis
- Mean Complexity: {metrics.get('complexity_stats', {}).get('mean', 0):.2f}
- Std Complexity: {metrics.get('complexity_stats', {}).get('std', 0):.2f}
- Min Complexity: {metrics.get('complexity_stats', {}).get('min', 0)}
- Max Complexity: {metrics.get('complexity_stats', {}).get('max', 0)}
"""
        return report


# Utility functions
import time
from datetime import datetime

def evaluate_inference_speed(model, test_input, n_runs: int = 100) -> Dict[str, float]:
    """Evaluate model inference speed"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**test_input)
    
    # Time runs
    times = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(**test_input)
            times.append(time.time() - start)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': min(times),
        'max_time': max(times),
        'throughput': 1.0 / np.mean(times)
    }