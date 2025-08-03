import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from core.types import ODEInstance

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from ODE instances for ML applications"""
    
    @staticmethod
    def extract_features(dataset: List[ODEInstance]) -> pd.DataFrame:
        """Extract comprehensive features from ODE dataset"""
        features = []
        
        for ode in dataset:
            try:
                feature_dict = FeatureExtractor._extract_single_ode_features(ode)
                features.append(feature_dict)
            except Exception as e:
                logger.error(f"Error extracting features for ODE {ode.id}: {e}")
                continue
        
        df = pd.DataFrame(features)
        
        # Add derived features
        df = FeatureExtractor._add_derived_features(df)
        
        return df
    
    @staticmethod
    def _extract_single_ode_features(ode: ODEInstance) -> Dict[str, Any]:
        """Extract features from a single ODE instance"""
        feature_dict = {
            # Basic metadata
            'id': ode.id,
            'generator_type': ode.generator_type.value,
            'generator_name': ode.generator_name,
            'function_name': ode.function_name,
            
            # Complexity metrics
            'operation_count': ode.operation_count,
            'atom_count': ode.atom_count,
            'symbol_count': ode.symbol_count,
            'complexity_score': ode.complexity_score,
            
            # Structural features
            'has_exponential': int('exp' in ode.ode_symbolic),
            'has_sine': int('sin' in ode.ode_symbolic),
            'has_cosine': int('cos' in ode.ode_symbolic),
            'has_logarithm': int('log' in ode.ode_symbolic),
            'has_rational': int('/' in ode.ode_symbolic),
            'has_pantograph': int(ode.has_pantograph),
            
            # Nonlinearity metrics
            'nonlin_pow_deriv_max': ode.nonlinearity_metrics.pow_deriv_max,
            'nonlin_pow_yprime': ode.nonlinearity_metrics.pow_yprime,
            'nonlin_total_degree': ode.nonlinearity_metrics.total_nonlinear_degree,
            'is_nonlinear': int(ode.generator_type.value == 'nonlinear'),
            
            # Verification metrics
            'verified': int(ode.verified),
            'verification_method': ode.verification_method.value,
            'verification_confidence': ode.verification_confidence,
            
            # Performance metrics
            'generation_time_ms': ode.generation_time * 1000,
            
            # Parameter features
            'param_alpha': ode.parameters.get('alpha', 0),
            'param_beta': ode.parameters.get('beta', 0),
            'param_M': ode.parameters.get('M', 0),
            'param_q': ode.parameters.get('q', 1),
            'param_v': ode.parameters.get('v', 1),
            'param_a': ode.parameters.get('a', 1),
            
            # Initial conditions count
            'n_initial_conditions': len(ode.initial_conditions),
            
            # Text length features
            'ode_text_length': len(ode.ode_symbolic),
            'solution_text_length': len(ode.solution_symbolic),
        }
        
        return feature_dict
    
    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataframe"""
        # Complexity ratios
        df['complexity_per_atom'] = df['complexity_score'] / (df['atom_count'] + 1)
        df['ops_per_symbol'] = df['operation_count'] / (df['symbol_count'] + 1)
        df['text_complexity_ratio'] = df['ode_text_length'] / (df['solution_text_length'] + 1)
        
        # Function type features
        df['has_transcendental'] = (
            df['has_exponential'] | df['has_sine'] | 
            df['has_cosine'] | df['has_logarithm']
        ).astype(int)
        
        # Verification success by type
        df['linear_verified'] = (
            (df['generator_type'] == 'linear') & df['verified']
        ).astype(int)
        
        df['nonlinear_verified'] = (
            (df['generator_type'] == 'nonlinear') & df['verified']
        ).astype(int)
        
        # Parameter statistics
        df['param_sum'] = (
            df['param_alpha'] + df['param_beta'] + 
            df['param_M'] + df['param_q'] + df['param_v']
        )
        
        df['param_product'] = (
            df['param_alpha'] * df['param_beta'] * 
            (df['param_M'] + 1) * df['param_q'] * df['param_v']
        )
        
        # Efficiency metrics
        df['generation_efficiency'] = df['complexity_score'] / (df['generation_time_ms'] + 1)
        
        # Categorical encodings
        df['generator_type_encoded'] = pd.Categorical(df['generator_type']).codes
        df['verification_method_encoded'] = pd.Categorical(df['verification_method']).codes
        
        return df
    
    @staticmethod
    def save_features(
        df: pd.DataFrame, 
        filepath: str = "ode_features.parquet",
        include_stats: bool = True
    ):
        """Save features to parquet file with optional statistics"""
        # Save main features
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        logger.info(f"Features saved to {filepath}")
        
        # Save statistics if requested
        if include_stats:
            stats_path = Path(filepath).with_suffix('.stats.json')
            stats = {
                'shape': list(df.shape),
                'numeric_stats': df.describe().to_dict(),
                'categorical_stats': {},
                'missing_values': df.isnull().sum().to_dict()
            }
            
            # Add categorical statistics
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                stats['categorical_stats'][col] = df[col].value_counts().to_dict()
            
            import json
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"Feature statistics saved to {stats_path}")
    
    @staticmethod
    def load_features(filepath: str = "ode_features.parquet") -> pd.DataFrame:
        """Load features from parquet file"""
        return pd.read_parquet(filepath, engine='pyarrow')