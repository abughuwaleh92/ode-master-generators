"""
Utility Functions for ML Pipeline

Helper functions for data preparation, model management, and generation.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sympy as sp
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def prepare_ml_dataset(
    dataset_path: str,
    output_dir: str = 'ml_data',
    test_split: float = 0.2,
    val_split: float = 0.1,
    seed: int = 42
) -> Dict[str, str]:
    """
    Prepare ODE dataset for machine learning
    
    Args:
        dataset_path: Path to JSONL dataset
        output_dir: Directory to save processed data
        test_split: Fraction for test set
        val_split: Fraction for validation set
        seed: Random seed
        
    Returns:
        Dictionary with paths to processed files
    """
    np.random.seed(seed)
    Path(output_dir).mkdir(exist_ok=True)
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load dataset
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} ODEs")
    
    # Extract features
    features = extract_ml_features(df)
    
    # Encode categorical variables
    encoders = {}
    
    # Generator encoding
    gen_encoder = LabelEncoder()
    features['generator_encoded'] = gen_encoder.fit_transform(features['generator_name'])
    encoders['generator'] = gen_encoder
    
    # Function encoding
    func_encoder = LabelEncoder()
    features['function_encoded'] = func_encoder.fit_transform(features['function_name'])
    encoders['function'] = func_encoder
    
    # Normalize numeric features
    numeric_cols = [
        'complexity_score', 'operation_count', 'atom_count', 'symbol_count',
        'alpha', 'beta', 'M', 'q', 'v', 'a',
        'pow_deriv_max', 'pow_yprime'
    ]
    
    scaler = StandardScaler()
    features[numeric_cols] = scaler.fit_transform(features[numeric_cols].fillna(0))
    
    # Split data
    n = len(features)
    indices = np.random.permutation(n)
    
    test_size = int(n * test_split)
    val_size = int(n * val_split)
    
    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]
    
    # Save splits
    train_data = features.iloc[train_idx]
    val_data = features.iloc[val_idx]
    test_data = features.iloc[test_idx]
    
    # Save to files
    paths = {
        'train': f"{output_dir}/train_data.parquet",
        'val': f"{output_dir}/val_data.parquet",
        'test': f"{output_dir}/test_data.parquet",
        'encoders': f"{output_dir}/encoders.pkl",
        'scaler': f"{output_dir}/scaler.pkl"
    }
    
    train_data.to_parquet(paths['train'])
    val_data.to_parquet(paths['val'])
    test_data.to_parquet(paths['test'])
    
    # Save encoders and scaler
    with open(paths['encoders'], 'wb') as f:
        pickle.dump(encoders, f)
    
    with open(paths['scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Dataset prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'n_generators': len(gen_encoder.classes_),
        'n_functions': len(func_encoder.classes_),
        'generators': list(gen_encoder.classes_),
        'functions': list(func_encoder.classes_),
        'numeric_features': numeric_cols
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return paths


def extract_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ML-ready features from ODE dataset
    
    Args:
        df: Raw ODE DataFrame
        
    Returns:
        Feature DataFrame
    """
    features = []
    
    for _, row in df.iterrows():
        feature_dict = {
            'id': row['id'],
            'generator_name': row['generator_name'],
            'function_name': row['function_name'],
            'generator_type': row['generator_type'],
            'complexity_score': row['complexity_score'],
            'operation_count': row['operation_count'],
            'atom_count': row['atom_count'],
            'symbol_count': row['symbol_count'],
            'verified': int(row['verified']),
            'has_pantograph': int(row.get('has_pantograph', False)),
            'verification_confidence': row.get('verification_confidence', 0),
            'ode_symbolic': row['ode_symbolic'],
            'solution_symbolic': row['solution_symbolic'],
            'ode_latex': row['ode_latex'],
            'solution_latex': row['solution_latex']
        }
        
        # Extract parameters
        params = row.get('parameters', {})
        for param in ['alpha', 'beta', 'M', 'q', 'v', 'a']:
            feature_dict[param] = params.get(param, 0)
        
        # Extract nonlinearity metrics
        if 'nonlinearity_metrics' in row and row['nonlinearity_metrics']:
            metrics = row['nonlinearity_metrics']
            feature_dict['pow_deriv_max'] = metrics.get('pow_deriv_max', 1)
            feature_dict['pow_yprime'] = metrics.get('pow_yprime', 1)
            feature_dict['is_exponential_nonlinear'] = int(metrics.get('is_exponential_nonlinear', False))
            feature_dict['is_logarithmic_nonlinear'] = int(metrics.get('is_logarithmic_nonlinear', False))
        else:
            feature_dict['pow_deriv_max'] = 1
            feature_dict['pow_yprime'] = 1
            feature_dict['is_exponential_nonlinear'] = 0
            feature_dict['is_logarithmic_nonlinear'] = 0
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)


def load_pretrained_model(
    model_type: str,
    checkpoint_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.nn.Module:
    """
    Load a pretrained model
    
    Args:
        model_type: Type of model ('pattern_net', 'transformer', etc.)
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    from .models import get_model
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = get_model(model_type, **model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {checkpoint_path}")
    
    return model


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    checkpoint_path: str,
    model_config: Optional[Dict] = None
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint
        model_config: Model configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'model_config': model_config or {}
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def generate_novel_odes(
    model: torch.nn.Module,
    n_samples: int = 10,
    generators: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
    temperature: float = 0.8,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[Dict[str, Any]]:
    """
    Generate novel ODEs using a trained model
    
    Args:
        model: Trained generation model
        n_samples: Number of ODEs to generate
        generators: List of generators to use
        functions: List of functions to use
        temperature: Sampling temperature
        device: Device for generation
        
    Returns:
        List of generated ODEs with metadata
    """
    model.eval()
    model.to(device)
    
    if generators is None:
        generators = ['L1', 'L2', 'L3', 'N1', 'N2', 'N3']
    
    if functions is None:
        functions = ['identity', 'exponential', 'sine', 'cosine', 'quadratic']
    
    generated_odes = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Randomly select generator and function
            gen = np.random.choice(generators)
            func = np.random.choice(functions)
            
            try:
                # Generate based on model type
                if hasattr(model, 'generate'):
                    # Language model or VAE
                    result = model.generate(
                        generator=gen,
                        function=func,
                        temperature=temperature
                    )
                else:
                    # Pattern network - sample parameters
                    # This is a simplified approach
                    params = sample_ode_parameters(model, gen, func)
                    result = construct_ode_from_params(gen, func, params)
                
                generated_odes.append({
                    'id': i,
                    'generator': gen,
                    'function': func,
                    'ode': result,
                    'timestamp': str(datetime.now())
                })
                
            except Exception as e:
                logger.debug(f"Failed to generate ODE {i}: {e}")
                continue
    
    return generated_odes


def sample_ode_parameters(
    model: torch.nn.Module,
    generator: str,
    function: str,
    n_attempts: int = 10
) -> Dict[str, float]:
    """
    Sample ODE parameters using a pattern network
    
    Args:
        model: Trained pattern network
        generator: Generator name
        function: Function name
        n_attempts: Number of sampling attempts
        
    Returns:
        Sampled parameters
    """
    # This is a placeholder - implement based on your model architecture
    # Could use the model's parameter head to predict optimal parameters
    
    params = {
        'alpha': np.random.uniform(0, 2),
        'beta': np.random.uniform(0.5, 2),
        'M': np.random.uniform(0, 1),
        'q': np.random.choice([2, 3]),
        'v': np.random.choice([2, 3, 4]),
        'a': np.random.choice([2, 3, 4])
    }
    
    return params


def construct_ode_from_params(
    generator: str,
    function: str,
    params: Dict[str, float]
) -> str:
    """
    Construct ODE string from generator, function, and parameters
    
    Args:
        generator: Generator name
        function: Function name
        params: Parameter values
        
    Returns:
        ODE string
    """
    # This is a simplified version - you'd need to implement the actual
    # ODE construction based on your generators
    
    # Example for L1 generator
    if generator == 'L1':
        ode_template = "y''(x) + y(x) = {rhs}"
        # Construct RHS based on function and parameters
        rhs = f"pi*{function}(alpha + beta*exp(-x))"
        return ode_template.format(rhs=rhs)
    
    # Add more generators as needed
    return f"{generator} with {function}"


def create_ode_tokenizer(vocab_path: Optional[str] = None) -> Any:
    """
    Create or load tokenizer for ODE text
    
    Args:
        vocab_path: Path to vocabulary file
        
    Returns:
        Tokenizer instance
    """
    if vocab_path and Path(vocab_path).exists():
        # Load custom vocabulary
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        # Create custom tokenizer
        # This is a placeholder - implement based on your needs
        tokenizer = None
    else:
        # Use pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Add ODE-specific tokens
        special_tokens = {
            'additional_special_tokens': [
                '<ode>', '</ode>',
                '<solution>', '</solution>',
                '<generator>', '</generator>',
                '<function>', '</function>',
                "y'", "y''", "y'''",
                'alpha', 'beta', 'pi'
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer


def visualize_latent_space(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    output_path: str = 'latent_visualization.png',
    n_samples: int = 1000
):
    """
    Visualize latent space for VAE models
    
    Args:
        model: Trained VAE model
        dataset: Dataset to visualize
        output_path: Path to save visualization
        n_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    model.eval()
    
    # Extract latent representations
    latent_vecs = []
    labels = []
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            data = dataset[i]
            
            # Get latent representation
            mu, _ = model.encode(data['features'])
            latent_vecs.append(mu.cpu().numpy())
            labels.append(data['generator_id'])
    
    latent_vecs = np.array(latent_vecs)
    labels = np.array(labels)
    
    # Reduce dimensionality if needed
    if latent_vecs.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vecs)
    else:
        latent_2d = latent_vecs
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('ODE Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Latent space visualization saved to {output_path}")


def analyze_generation_diversity(generated_odes: List[str]) -> Dict[str, Any]:
    """
    Analyze diversity of generated ODEs
    
    Args:
        generated_odes: List of generated ODE strings
        
    Returns:
        Diversity metrics
    """
    from collections import Counter
    
    # Basic uniqueness
    unique_odes = set(generated_odes)
    uniqueness_ratio = len(unique_odes) / len(generated_odes)
    
    # Structural diversity
    structures = []
    operators = []
    functions = []
    
    for ode in generated_odes:
        # Count operators
        ops = Counter()
        ops['+'] = ode.count('+')
        ops['-'] = ode.count('-')
        ops['*'] = ode.count('*')
        ops['/'] = ode.count('/')
        ops['^'] = ode.count('^') + ode.count('**')
        operators.append(tuple(sorted(ops.items())))
        
        # Extract functions
        func_list = []
        for func in ['sin', 'cos', 'exp', 'log', 'sinh', 'cosh', 'tanh']:
            if func in ode:
                func_list.append(func)
        functions.append(tuple(sorted(func_list)))
        
        # Simplified structure (length category)
        if len(ode) < 50:
            structures.append('short')
        elif len(ode) < 100:
            structures.append('medium')
        else:
            structures.append('long')
    
    # Calculate diversity metrics
    operator_diversity = len(set(operators)) / len(operators)
    function_diversity = len(set(functions)) / len(functions)
    structure_diversity = len(set(structures)) / len(structures)
    
    # Character-level diversity (entropy)
    all_chars = ''.join(generated_odes)
    char_counts = Counter(all_chars)
    total_chars = sum(char_counts.values())
    char_probs = [count/total_chars for count in char_counts.values()]
    char_entropy = -sum(p * np.log2(p) for p in char_probs if p > 0)
    
    return {
        'uniqueness_ratio': uniqueness_ratio,
        'unique_count': len(unique_odes),
        'total_count': len(generated_odes),
        'operator_diversity': operator_diversity,
        'function_diversity': function_diversity,
        'structure_diversity': structure_diversity,
        'character_entropy': char_entropy,
        'avg_length': np.mean([len(ode) for ode in generated_odes]),
        'std_length': np.std([len(ode) for ode in generated_odes])
    }


# Main utility function for complete ML pipeline
def run_ml_pipeline(
    dataset_path: str,
    model_type: str = 'pattern_net',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    output_dir: str = 'ml_output'
) -> Dict[str, Any]:
    """
    Run complete ML pipeline
    
    Args:
        dataset_path: Path to ODE dataset
        model_type: Type of model to train
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Output directory
        
    Returns:
        Training results
    """
    from .train_ode_generator import ODEGeneratorTrainer
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prepare data
    logger.info("Preparing dataset...")
    data_paths = prepare_ml_dataset(dataset_path, f"{output_dir}/data")
    
    # Create trainer
    trainer = ODEGeneratorTrainer(dataset_path)
    
    # Train model based on type
    if model_type == 'pattern_net':
        model = trainer.train_pattern_network(epochs=epochs, batch_size=batch_size)
    elif model_type == 'language':
        model = trainer.train_language_model(epochs=epochs, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ODEEvaluator()
    
    # Load test data
    test_data = pd.read_parquet(data_paths['test'])
    
    # Create test dataset
    from torch.utils.data import TensorDataset
    # Simplified - you'd need to properly create the dataset
    
    # Generate novel ODEs
    logger.info("Generating novel ODEs...")
    novel_odes = generate_novel_odes(model, n_samples=100)
    
    # Analyze diversity
    diversity_metrics = analyze_generation_diversity(
        [ode['ode'] for ode in novel_odes]
    )
    
    results = {
        'model_type': model_type,
        'training_epochs': epochs,
        'novel_odes_generated': len(novel_odes),
        'diversity_metrics': diversity_metrics,
        'output_dir': output_dir
    }
    
    # Save results
    with open(f"{output_dir}/pipeline_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"ML pipeline complete! Results saved to {output_dir}")
    
    return results


# Import datetime for timestamp generation
from datetime import datetime