# core/types.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import time
import json
import numpy as np

class GeneratorType(Enum):
    """Type of ODE generator"""
    LINEAR = "linear"
    NONLINEAR = "nonlinear"

class VerificationMethod(Enum):
    """Method used for ODE verification"""
    SUBSTITUTION = "substitution"
    CHECKODESOL = "checkodesol"
    NUMERIC = "numeric"
    FAILED = "failed"
    PENDING = "pending"

@dataclass
class NonlinearityMetrics:
    """Metrics for tracking nonlinearity in ODEs"""
    pow_deriv_max: int = 1
    pow_yprime: int = 1
    has_pantograph: bool = False
    is_exponential_nonlinear: bool = False
    is_logarithmic_nonlinear: bool = False
    total_nonlinear_degree: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'pow_deriv_max': self.pow_deriv_max,
            'pow_yprime': self.pow_yprime,
            'has_pantograph': self.has_pantograph,
            'is_exponential_nonlinear': self.is_exponential_nonlinear,
            'is_logarithmic_nonlinear': self.is_logarithmic_nonlinear,
            'total_nonlinear_degree': self.total_nonlinear_degree
        }

@dataclass
class ODEInstance:
    """Complete ODE instance with metadata"""
    # Core identifiers
    id: int
    generator_type: GeneratorType
    generator_name: str
    function_name: str
    
    # Symbolic representations
    ode_symbolic: str
    ode_latex: str
    solution_symbolic: str
    solution_latex: str
    
    # Initial conditions and parameters
    initial_conditions: Dict[str, str]
    parameters: Dict[str, float]
    
    # Complexity metrics
    complexity_score: int
    operation_count: int
    atom_count: int
    symbol_count: int
    
    # Verification status
    verified: bool
    verification_method: VerificationMethod
    verification_confidence: float
    
    # Special properties
    has_pantograph: bool
    generation_time: float
    
    # Optional fields with defaults
    ode_numeric: Optional[Callable] = field(default=None, repr=False, compare=False)
    solution_numeric: Optional[Callable] = field(default=None, repr=False, compare=False)
    nonlinearity_metrics: Optional[NonlinearityMetrics] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        base_dict = {
            'id': self.id,
            'generator_type': self.generator_type.value,
            'generator_name': self.generator_name,
            'function_name': self.function_name,
            'ode_symbolic': self.ode_symbolic,
            'ode_latex': self.ode_latex,
            'solution_symbolic': self.solution_symbolic,
            'solution_latex': self.solution_latex,
            'initial_conditions': self.initial_conditions,
            'parameters': self.parameters,
            'complexity_score': self.complexity_score,
            'operation_count': self.operation_count,
            'atom_count': self.atom_count,
            'symbol_count': self.symbol_count,
            'has_pantograph': self.has_pantograph,
            'verified': self.verified,
            'verification_method': self.verification_method.value,
            'verification_confidence': self.verification_confidence,
            'generation_time': self.generation_time,
            'metadata': self.metadata
        }
        
        # Add nonlinearity metrics if present
        if self.nonlinearity_metrics:
            base_dict['nonlinearity_metrics'] = self.nonlinearity_metrics.to_dict()
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ODEInstance':
        """Create instance from dictionary"""
        # Convert enums
        data['generator_type'] = GeneratorType(data['generator_type'])
        data['verification_method'] = VerificationMethod(data['verification_method'])
        
        # Handle nonlinearity metrics
        if 'nonlinearity_metrics' in data and data['nonlinearity_metrics']:
            data['nonlinearity_metrics'] = NonlinearityMetrics(**data['nonlinearity_metrics'])
        
        # Remove any extra fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    def get_equation_string(self) -> str:
        """Get a readable string representation of the ODE"""
        return f"{self.ode_symbolic}"
    
    def get_solution_string(self) -> str:
        """Get a readable string representation of the solution"""
        return f"y(x) = {self.solution_symbolic}"
    
    def evaluate_solution(self, x_value: float) -> Optional[float]:
        """Evaluate the solution at a given x value"""
        if self.solution_numeric is None:
            return None
        try:
            return float(self.solution_numeric(x_value))
        except Exception:
            return None
    
    def evaluate_residual(self, x_value: float) -> Optional[float]:
        """Evaluate the ODE residual at a given x value"""
        if self.ode_numeric is None or self.solution_numeric is None:
            return None
        try:
            return float(self.ode_numeric(x_value))
        except Exception:
            return None

@dataclass
class GeneratorResult:
    """Result of ODE generation attempt"""
    success: bool
    ode_instance: Optional[ODEInstance] = None
    error: Optional[str] = None
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'ode_instance': self.ode_instance.to_dict() if self.ode_instance else None,
            'error': self.error,
            'generation_time': self.generation_time,
            'metadata': self.metadata
        }

@dataclass
class DatasetStatistics:
    """Statistics for an ODE dataset"""
    total_odes: int = 0
    verified_odes: int = 0
    linear_odes: int = 0
    nonlinear_odes: int = 0
    pantograph_odes: int = 0
    
    # Complexity statistics
    complexity_mean: float = 0.0
    complexity_std: float = 0.0
    complexity_min: int = 0
    complexity_max: int = 0
    
    # Verification statistics
    verification_rate: float = 0.0
    verification_methods: Dict[str, int] = field(default_factory=dict)
    
    # Generator performance
    generator_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    function_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Timing statistics
    total_generation_time: float = 0.0
    average_generation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_odes': self.total_odes,
            'verified_odes': self.verified_odes,
            'linear_odes': self.linear_odes,
            'nonlinear_odes': self.nonlinear_odes,
            'pantograph_odes': self.pantograph_odes,
            'complexity_mean': self.complexity_mean,
            'complexity_std': self.complexity_std,
            'complexity_min': self.complexity_min,
            'complexity_max': self.complexity_max,
            'verification_rate': self.verification_rate,
            'verification_methods': self.verification_methods,
            'generator_stats': self.generator_stats,
            'function_stats': self.function_stats,
            'total_generation_time': self.total_generation_time,
            'average_generation_time': self.average_generation_time
        }
    
    @classmethod
    def from_dataset(cls, dataset: List[ODEInstance]) -> 'DatasetStatistics':
        """Compute statistics from a dataset"""
        stats = cls()
        
        if not dataset:
            return stats
        
        stats.total_odes = len(dataset)
        stats.verified_odes = sum(1 for ode in dataset if ode.verified)
        stats.linear_odes = sum(1 for ode in dataset if ode.generator_type == GeneratorType.LINEAR)
        stats.nonlinear_odes = sum(1 for ode in dataset if ode.generator_type == GeneratorType.NONLINEAR)
        stats.pantograph_odes = sum(1 for ode in dataset if ode.has_pantograph)
        
        # Complexity statistics
        complexities = [ode.complexity_score for ode in dataset]
        stats.complexity_mean = np.mean(complexities)
        stats.complexity_std = np.std(complexities)
        stats.complexity_min = min(complexities)
        stats.complexity_max = max(complexities)
        
        # Verification statistics
        stats.verification_rate = stats.verified_odes / stats.total_odes if stats.total_odes > 0 else 0.0
        
        for ode in dataset:
            method = ode.verification_method.value
            stats.verification_methods[method] = stats.verification_methods.get(method, 0) + 1
        
        # Generator statistics
        for ode in dataset:
            gen_key = f"{ode.generator_type.value}_{ode.generator_name}"
            if gen_key not in stats.generator_stats:
                stats.generator_stats[gen_key] = {'total': 0, 'verified': 0}
            stats.generator_stats[gen_key]['total'] += 1
            if ode.verified:
                stats.generator_stats[gen_key]['verified'] += 1
        
        # Function statistics
        for ode in dataset:
            func = ode.function_name
            if func not in stats.function_stats:
                stats.function_stats[func] = {'total': 0, 'verified': 0}
            stats.function_stats[func]['total'] += 1
            if ode.verified:
                stats.function_stats[func]['verified'] += 1
        
        # Timing statistics
        generation_times = [ode.generation_time for ode in dataset]
        stats.total_generation_time = sum(generation_times)
        stats.average_generation_time = np.mean(generation_times)
        
        return stats

@dataclass
class HybridODE:
    """Special class for hybrid/delay ODEs"""
    id: int
    ode_type: str  # 'delay', 'fractional', 'stochastic', etc.
    base_generator: str
    
    # Symbolic representations
    ode_symbolic: str
    ode_latex: str
    solution_symbolic: Optional[str] = None
    solution_latex: Optional[str] = None
    
    # Special parameters
    delay_terms: List[str] = field(default_factory=list)
    fractional_orders: Dict[str, float] = field(default_factory=dict)
    stochastic_terms: List[str] = field(default_factory=list)
    
    # Metadata
    parameters: Dict[str, float] = field(default_factory=dict)
    initial_conditions: Dict[str, str] = field(default_factory=dict)
    boundary_conditions: Dict[str, str] = field(default_factory=dict)
    
    verified: bool = False
    complexity_score: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'ode_type': self.ode_type,
            'base_generator': self.base_generator,
            'ode_symbolic': self.ode_symbolic,
            'ode_latex': self.ode_latex,
            'solution_symbolic': self.solution_symbolic,
            'solution_latex': self.solution_latex,
            'delay_terms': self.delay_terms,
            'fractional_orders': self.fractional_orders,
            'stochastic_terms': self.stochastic_terms,
            'parameters': self.parameters,
            'initial_conditions': self.initial_conditions,
            'boundary_conditions': self.boundary_conditions,
            'verified': self.verified,
            'complexity_score': self.complexity_score,
            'metadata': self.metadata
        }

@dataclass
class VerificationResult:
    """Detailed verification result"""
    verified: bool
    method: VerificationMethod
    confidence: float
    
    # Detailed results by method
    substitution_result: Optional[bool] = None
    checkodesol_result: Optional[bool] = None
    numeric_result: Optional[bool] = None
    
    # Residual information
    symbolic_residual: Optional[str] = None
    numeric_residuals: List[float] = field(default_factory=list)
    max_residual: Optional[float] = None
    
    # Timing
    verification_time: float = 0.0
    
    # Error information
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'verified': self.verified,
            'method': self.method.value,
            'confidence': self.confidence,
            'substitution_result': self.substitution_result,
            'checkodesol_result': self.checkodesol_result,
            'numeric_result': self.numeric_result,
            'symbolic_residual': self.symbolic_residual,
            'numeric_residuals': self.numeric_residuals,
            'max_residual': self.max_residual,
            'verification_time': self.verification_time,
            'errors': self.errors
        }

# Type aliases for clarity
ODEDataset = List[ODEInstance]
GeneratorMap = Dict[str, Any]  # Maps generator name to generator instance
FunctionLibrary = Dict[str, Callable]  # Maps function name to function

# Constants
MAX_COMPLEXITY_SCORE = 1000
MAX_OPERATION_COUNT = 500
DEFAULT_NUMERIC_TOLERANCE = 1e-8
DEFAULT_VERIFICATION_TIMEOUT = 30.0

# Utility functions
def create_empty_ode_instance(ode_id: int = -1) -> ODEInstance:
    """Create an empty ODE instance for testing"""
    return ODEInstance(
        id=ode_id,
        generator_type=GeneratorType.LINEAR,
        generator_name="test",
        function_name="identity",
        ode_symbolic="",
        ode_latex="",
        solution_symbolic="",
        solution_latex="",
        initial_conditions={},
        parameters={},
        complexity_score=0,
        operation_count=0,
        atom_count=0,
        symbol_count=0,
        verified=False,
        verification_method=VerificationMethod.PENDING,
        verification_confidence=0.0,
        has_pantograph=False,
        generation_time=0.0
    )

def is_high_complexity_ode(ode: ODEInstance, threshold: int = 100) -> bool:
    """Check if an ODE has high complexity"""
    return ode.complexity_score > threshold

def filter_verified_odes(dataset: ODEDataset) -> ODEDataset:
    """Filter to get only verified ODEs"""
    return [ode for ode in dataset if ode.verified]

def filter_by_generator_type(dataset: ODEDataset, gen_type: GeneratorType) -> ODEDataset:
    """Filter ODEs by generator type"""
    return [ode for ode in dataset if ode.generator_type == gen_type]

def group_by_function(dataset: ODEDataset) -> Dict[str, ODEDataset]:
    """Group ODEs by function name"""
    groups = {}
    for ode in dataset:
        if ode.function_name not in groups:
            groups[ode.function_name] = []
        groups[ode.function_name].append(ode)
    return groups