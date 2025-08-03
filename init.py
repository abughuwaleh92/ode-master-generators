"""ODE Master Generators - Root Package"""

__version__ = "2.0.0"
__author__ = "Your Name"

from .core import ODEInstance, HybridODE, GeneratorResult, SYMBOLS
from .pipeline.generator import ODEDatasetGenerator
from .utils.config import ConfigManager

__all__ = [
    'ODEDatasetGenerator',
    'ConfigManager',
    'ODEInstance',
    'HybridODE',
    'GeneratorResult',
    'SYMBOLS'
]