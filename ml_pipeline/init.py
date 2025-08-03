"""
Machine Learning Pipeline for ODE Generation

This module provides tools for training ML models on ODE datasets
to learn patterns and generate novel equations.
"""

from .models import (
    ODEPatternNet,
    ODELanguageModel,
    ODETransformer,
    ODEVAE
)
from .train_ode_generator import ODEGeneratorTrainer
from .evaluation import ODEEvaluator, NoveltyDetector
from .utils import (
    prepare_ml_dataset,
    load_pretrained_model,
    generate_novel_odes
)

__all__ = [
    'ODEPatternNet',
    'ODELanguageModel',
    'ODETransformer',
    'ODEVAE',
    'ODEGeneratorTrainer',
    'ODEEvaluator',
    'NoveltyDetector',
    'prepare_ml_dataset',
    'load_pretrained_model',
    'generate_novel_odes'
]

__version__ = '1.0.0'