import sympy as sp
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
import signal
import multiprocessing as mp
from functools import wraps
import time

# Import from core types
from core.types import VerificationMethod
from core.symbols import SYMBOLS
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)

def timeout(seconds):
    """Timeout decorator using signals (Unix only)"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            
            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        return wrapper
    return decorator

class ODEVerifier:
    """Production-grade ODE verification system with timeouts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Use caller-supplied configuration if present; otherwise fall back to safe defaults.
        cfg = config or {
            "numeric_test_points": [0.1, 0.3, 0.7, 1.0],
            "residual_tolerance": 1e-8,
            "verification_timeout": 30,
        }
        self.test_points        = cfg["numeric_test_points"]
        self.residual_tolerance = cfg["residual_tolerance"]
        self.timeout_seconds    = cfg["verification_timeout"]
        
    def verify(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr
    ) -> Tuple[bool, VerificationMethod, float]:
        """
        Verify ODE solution with timeout protection
        Returns: (verified, method, confidence)
        """
        # For Windows compatibility, use threading-based timeout or skip timeout
        import platform
        
        if platform.system() == 'Windows':
            # Direct verification without multiprocessing on Windows
            return self._verify_with_methods(ode_expr, solution_expr)
        else:
            # Use multiprocessing for timeout on Unix-like systems
            with mp.Pool(processes=1) as pool:
                try:
                    result = pool.apply_async(
                        self._verify_with_methods,
                        args=(ode_expr, solution_expr)
                    )
                    
                    # Get result with timeout
                    verified, method, confidence = result.get(timeout=self.timeout_seconds)
                    return verified, method, confidence
                    
                except mp.TimeoutError:
                    logger.warning(f"Verification timed out after {self.timeout_seconds}s")
                    return False, VerificationMethod.FAILED, 0.0
                except Exception as e:
                    logger.error(f"Verification error: {e}")
                    return False, VerificationMethod.FAILED, 0.0
    
    def _verify_with_methods(
        self,
        ode_expr: sp.Expr,
        solution_expr: sp.Expr
    ) -> Tuple[bool, VerificationMethod, float]:
        """Try verification methods in order"""
        methods = [
            (self._verify_by_substitution, VerificationMethod.SUBSTITUTION),
            (self._verify_by_checkodesol, VerificationMethod.CHECKODESOL),
            (self._verify_numerically, VerificationMethod.NUMERIC)
        ]
        
        for verify_func, method in methods:
            try:
                verified, confidence = verify_func(ode_expr, solution_expr)
                if verified:
                    return True, method, confidence
            except Exception as e:
                logger.debug(f"Verification method {method.value} failed: {e}")
                continue
        
        return False, VerificationMethod.FAILED, 0.0
    
    def _verify_by_substitution(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr
    ) -> Tuple[bool, float]:
        """Enhanced substitution verification with confidence scoring"""
        try:
            # Ensure expressions are SymPy objects
            if isinstance(ode_expr, str):
                ode_expr = sp.sympify(ode_expr)
            if isinstance(solution_expr, str):
                solution_expr = sp.sympify(solution_expr)
            
            # Get maximum derivative order
            max_order = self._get_max_derivative_order(ode_expr)
            
            # Create substitutions
            y_func = SYMBOLS.y
            x = SYMBOLS.x
            
            substitutions = {y_func(x): solution_expr}
            
            # Add derivative substitutions
            for order in range(1, max_order + 1):
                substitutions[y_func(x).diff(x, order)] = sp.diff(solution_expr, x, order)
            
            # Handle pantograph terms
            substitutions = self._add_pantograph_substitutions(
                ode_expr, solution_expr, substitutions
            )
            
            # Substitute and compute residual
            lhs = ode_expr.lhs.subs(substitutions)
            rhs = ode_expr.rhs.subs(substitutions)
            residual = lhs - rhs
            
            # Try multiple simplification strategies
            confidence = 0.0
            
            # Method 1: Direct comparison
            if residual.equals(0) or residual == 0:
                return True, 1.0
            
            # Method 2: Simplification
            try:
                simplified = sp.simplify(residual)
                if simplified.equals(0) or simplified == 0:
                    return True, 0.95
            except:
                pass
            
            # Method 3: Expansion and simplification
            try:
                expanded = sp.expand(residual)
                if expanded.equals(0) or expanded == 0:
                    return True, 0.9
                
                simplified_expanded = sp.simplify(expanded)
                if simplified_expanded.equals(0) or simplified_expanded == 0:
                    return True, 0.85
            except:
                pass
            
            # Method 4: Numerical evaluation
            try:
                if self._check_numeric_residual(residual):
                    return True, 0.8
            except:
                pass
            
            # Method 5: Check if constant near zero
            try:
                if residual.is_constant():
                    val = abs(float(residual))
                    if val < self.residual_tolerance:
                        confidence = 1.0 - (val / self.residual_tolerance)
                        return True, confidence * 0.7
            except:
                pass
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Substitution verification error: {e}")
            return False, 0.0
    
    def _verify_by_checkodesol(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr
    ) -> Tuple[bool, float]:
        """Verify using SymPy's checkodesol"""
        try:
            from sympy.solvers.ode import checkodesol
            
            # Convert to standard form if needed
            y = SYMBOLS.y(SYMBOLS.x)
            
            # Try direct check
            result = checkodesol(ode_expr, solution_expr, y)
            
            if isinstance(result, tuple) and len(result) >= 2:
                is_valid = result[0]
                if is_valid:
                    return True, 0.9
            elif result:
                return True, 0.9
                
            return False, 0.0
            
        except Exception as e:
            logger.debug(f"checkodesol failed: {e}")
            return False, 0.0
    
    def _verify_numerically(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr
    ) -> Tuple[bool, float]:
        """Numerical verification at test points"""
        try:
            # Get substitutions
            y_func = SYMBOLS.y
            x = SYMBOLS.x
            
            max_order = self._get_max_derivative_order(ode_expr)
            
            # Create lambdified functions
            solution_func = sp.lambdify(x, solution_expr, 'numpy')
            
            # Test at multiple points
            residuals = []
            
            for test_x in self.test_points:
                try:
                    # Compute solution and derivatives at test point
                    substitutions = {x: test_x}
                    substitutions[y_func(x)] = solution_func(test_x)
                    
                    # Add derivatives
                    for order in range(1, max_order + 1):
                        deriv_expr = sp.diff(solution_expr, x, order)
                        deriv_func = sp.lambdify(x, deriv_expr, 'numpy')
                        substitutions[y_func(x).diff(x, order)] = deriv_func(test_x)
                    
                    # Compute residual
                    lhs_val = float(ode_expr.lhs.subs(substitutions))
                    rhs_val = float(ode_expr.rhs.subs(substitutions))
                    residual = abs(lhs_val - rhs_val)
                    residuals.append(residual)
                    
                except Exception as e:
                    logger.debug(f"Numeric evaluation failed at x={test_x}: {e}")
                    return False, 0.0
            
            # Check if all residuals are small
            max_residual = max(residuals)
            if max_residual < self.residual_tolerance:
                confidence = 1.0 - (max_residual / self.residual_tolerance)
                return True, confidence * 0.7
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Numerical verification error: {e}")
            return False, 0.0
    
    def _get_max_derivative_order(self, ode_expr: sp.Expr) -> int:
        """Dynamically determine maximum derivative order"""
        max_order = 0
        
        for atom in ode_expr.atoms(sp.Derivative):
            if hasattr(atom, 'args') and len(atom.args) > 1:
                for arg in atom.args[1:]:
                    if isinstance(arg, tuple) and len(arg) == 2:
                        max_order = max(max_order, arg[1])
                    elif arg == SYMBOLS.x:
                        max_order = max(max_order, 1)
        
        return max_order
    
    def _add_pantograph_substitutions(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr, 
        substitutions: Dict
    ) -> Dict:
        """Add substitutions for pantograph terms"""
        y_func = SYMBOLS.y
        x = SYMBOLS.x
        
        # Look for pantograph terms like y(x/a)
        for atom in ode_expr.atoms():
            if isinstance(atom, sp.Function) and str(atom.func) == 'y':
                if len(atom.args) > 0 and atom.args[0] != x:
                    scale_arg = atom.args[0]
                    try:
                        substitutions[atom] = solution_expr.subs(x, scale_arg)
                    except Exception:
                        # Try simplifying first
                        simplified = sp.simplify(solution_expr)
                        substitutions[atom] = simplified.subs(x, scale_arg)
        
        return substitutions
    
    def _check_numeric_residual(self, residual: sp.Expr) -> bool:
        """Check if residual is numerically zero at test points"""
        try:
            x = SYMBOLS.x
            
            for test_val in self.test_points:
                # Use sp.N for proper numeric evaluation
                try:
                    numeric_val = sp.N(residual.subs(x, test_val), 15)
                    
                    # Handle complex results properly
                    if numeric_val.is_real:
                        magnitude = abs(float(numeric_val))
                    else:
                        magnitude = abs(complex(numeric_val))
                    
                    if magnitude > self.residual_tolerance:
                        return False
                        
                except Exception as e:
                    logger.debug(f"Numeric evaluation failed at x={test_val}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Numeric residual check failed: {e}")
            return False
    
    def analyze_ode_properties(self, ode_expr: sp.Expr) -> Dict[str, Any]:
        """Analyze ODE properties for classification"""
        try:
            if isinstance(ode_expr, str):
                ode_expr = sp.sympify(ode_expr)
            
            atoms = ode_expr.atoms()
            
            properties = {
                'order': self._get_max_derivative_order(ode_expr),
                'operation_count': sp.count_ops(ode_expr),
                'atom_count': len(atoms),
                'symbol_count': len(ode_expr.free_symbols),
                'has_nonlinear_terms': self._check_nonlinearity(atoms),
                'has_exponential': any(atom.func is sp.exp for atom in atoms if hasattr(atom, 'func')),
                'has_trigonometric': any(
                    atom.func in [sp.sin, sp.cos, sp.tan] 
                    for atom in atoms if hasattr(atom, 'func')
                ),
                'has_hyperbolic': any(
                    atom.func in [sp.sinh, sp.cosh, sp.tanh] 
                    for atom in atoms if hasattr(atom, 'func')
                ),
                'has_logarithmic': any(
                    atom.func is sp.log 
                    for atom in atoms if hasattr(atom, 'func')
                ),
                'has_rational': len(ode_expr.atoms(sp.Rational)) > 0,
                'complexity_score': len(str(ode_expr)),
            }
            
            return properties
            
        except Exception as e:
            logger.error(f"Error analyzing ODE properties: {e}")
            return {
                'order': 0, 
                'operation_count': 0, 
                'complexity_score': 0,
                'atom_count': 0, 
                'symbol_count': 0
            }
    
    def _check_nonlinearity(self, atoms) -> bool:
        """Check if ODE has nonlinear terms"""
        for atom in atoms:
            # Check for powers > 1
            if hasattr(atom, 'is_Pow') and atom.is_Pow:
                if hasattr(atom, 'exp') and atom.exp > 1:
                    return True
            
            # Check for products of dependent variables
            if hasattr(atom, 'is_Mul') and atom.is_Mul:
                y_count = sum(
                    1 for arg in atom.args 
                    if hasattr(arg, 'has') and arg.has(SYMBOLS.y)
                )
                if y_count > 1:
                    return True
        
        return False
