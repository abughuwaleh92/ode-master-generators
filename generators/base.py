from abc import ABC, abstractmethod
import sympy as sp
from typing import Dict, Tuple, Optional
import time
import logging
from core.types import GeneratorResult, ODEInstance, GeneratorType, NonlinearityMetrics
from core.symbols import SYMBOLS
from utils.derivatives import DerivativeComputer

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """Abstract base class for all ODE generators"""
    
    def __init__(self, name: str, generator_type: GeneratorType):
        self.name = name
        self.generator_type = generator_type
        self.derivative_computer = DerivativeComputer()
        
    @abstractmethod
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        """
        Generate ODE, solution, and initial conditions
        Returns: (ode_equation, solution, initial_conditions)
        """
        pass
    
    def create_ode_instance(
        self, 
        f_key: str, 
        params: Dict[str, float],
        ode_id: int,
        verified: bool = False,
        verification_method: str = "pending",
        verification_confidence: float = 0.0
    ) -> GeneratorResult:
        """Create a complete ODE instance"""
        start_time = time.time()
        
        try:
            # Generate ODE
            ode, solution, ics = self.generate(f_key, params)
            
            if ode is None:
                return GeneratorResult(
                    success=False,
                    error=f"Failed to generate ODE with {self.name}"
                )
            
            # Create instance
            instance = ODEInstance(
                id=ode_id,
                generator_type=self.generator_type,
                generator_name=self.name,
                function_name=f_key,
                ode_symbolic=str(ode),
                ode_numeric=str(ode),
                ode_latex=sp.latex(ode),
                solution_symbolic=str(solution),
                solution_numeric=str(solution),
                solution_latex=sp.latex(solution),
                initial_conditions=ics,
                parameters=params,
                complexity_score=len(str(ode)),
                operation_count=sp.count_ops(ode),
                atom_count=len(ode.atoms()),
                symbol_count=len(ode.free_symbols),
                nonlinearity_metrics=self._compute_nonlinearity_metrics(params),
                has_pantograph=self._check_pantograph(ode),
                verified=verified,
                verification_method=verification_method,
                verification_confidence=verification_confidence,
                generation_time=time.time() - start_time
            )
            
            return GeneratorResult(
                success=True,
                ode_instance=instance,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name} generator: {e}")
            return GeneratorResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        """Compute nonlinearity metrics based on parameters"""
        return NonlinearityMetrics(
            pow_deriv_max=params.get('q', 1),
            pow_yprime=params.get('v', 1),
            has_pantograph='a' in params,
            total_nonlinear_degree=params.get('q', 1) + params.get('v', 1)
        )
    
    def _check_pantograph(self, ode: sp.Eq) -> bool:
        """Check if ODE contains pantograph terms"""
        y = SYMBOLS.y
        x = SYMBOLS.x
        
        # Look for y(x/a) or similar terms
        for atom in ode.atoms():
            if isinstance(atom, sp.Function) and atom.func == y:
                if len(atom.args) > 0 and atom.args[0] != x:
                    if 'a' in str(atom.args[0]) or '/' in str(atom.args[0]):
                        return True
        
        return False
def create_ode_instance(
    self, 
    f_key: str, 
    params: Dict[str, float],
    ode_id: int
) -> GeneratorResult:
    """Create a complete ODE instance"""
    start_time = time.time()
    
    try:
        # Generate ODE
        ode, solution, ics = self.generate(f_key, params)
        
        if ode is None or solution is None:
            return GeneratorResult(
                success=False,
                error=f"Failed to generate ODE for {f_key}",
                ode_instance=None
            )
        
        # Debug logging
        logger.debug(f"Generated ODE type: {type(ode)}")
        logger.debug(f"Generated solution type: {type(solution)}")
        
        # Ensure we have SymPy expressions
        if isinstance(ode, str):
            ode = sp.sympify(ode)
        if isinstance(solution, str):
            solution = sp.sympify(solution)
        
        # Create instance
        ode_instance = ODEInstance(
            id=ode_id,
            generator_type=self.generator_type,
            generator_name=self.name,
            function_name=f_key,
            ode_symbolic=str(ode),  # Store as string
            ode_latex=sp.latex(ode),
            solution_symbolic=str(solution),  # Store as string
            solution_latex=sp.latex(solution),
            initial_conditions=ics,
            parameters=params,
            complexity_score=sp.count_ops(ode),
            operation_count=sp.count_ops(ode),
            atom_count=len(ode.atoms()),
            symbol_count=len(ode.free_symbols),
            has_pantograph=self._check_pantograph(ode),
            verified=False,
            verification_method=VerificationMethod.PENDING,
            verification_confidence=0.0,
            generation_time=time.time() - start_time
        )
        
        # Add nonlinearity metrics if nonlinear
        if self.generator_type == GeneratorType.NONLINEAR:
            ode_instance.nonlinearity_metrics = self._compute_nonlinearity_metrics(params)
        
        return GeneratorResult(
            success=True,
            ode_instance=ode_instance,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Error creating ODE instance: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        
        return GeneratorResult(
            success=False,
            error=str(e),
            ode_instance=None
        )
