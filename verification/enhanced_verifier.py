# verification/enhanced_verifier.py
"""
Enhanced verification system with full support for all ODE types
"""

import sympy as sp
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import logging
from functools import wraps
import signal
import time
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize

from core.types import VerificationMethod
from core.symbols import SYMBOLS
from verification.verifier import ODEVerifier

logger = logging.getLogger(__name__)


class EnhancedODEVerifier(ODEVerifier):
    """
    Enhanced verifier with advanced verification methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Extended configuration
        self.series_order = config.get('series_order', 10) if config else 10
        self.pantograph_points = config.get('pantograph_test_points', 
                                           [0.5, 1.0, 2.0, 4.0]) if config else [0.5, 1.0, 2.0, 4.0]
        self.complex_tolerance = config.get('complex_tolerance', 1e-10) if config else 1e-10
        
    def verify(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr
    ) -> Tuple[bool, VerificationMethod, float]:
        """
        Enhanced verification with multiple strategies
        """
        # Check for special ODE types
        if self._is_pantograph(ode_expr):
            return self._verify_pantograph(ode_expr, solution_expr)
        
        if self._is_higher_order(ode_expr):
            return self._verify_higher_order(ode_expr, solution_expr)
        
        if self._has_complex_terms(ode_expr):
            return self._verify_complex(ode_expr, solution_expr)
        
        # Fall back to standard verification
        return super().verify(ode_expr, solution_expr)
    
    def _is_pantograph(self, ode_expr: sp.Expr) -> bool:
        """Check if ODE contains pantograph terms"""
        y = SYMBOLS.y
        x = SYMBOLS.x
        
        for atom in ode_expr.atoms():
            if isinstance(atom, sp.Function) and atom.func == y:
                if atom.args and atom.args[0] != x:
                    # Check for x/a pattern
                    arg = atom.args[0]
                    if arg.has(x) and (arg/x).is_constant():
                        return True
        return False
    
    def _is_higher_order(self, ode_expr: sp.Expr) -> bool:
        """Check if ODE is higher than second order"""
        return self._get_max_derivative_order(ode_expr) > 2
    
    def _has_complex_terms(self, ode_expr: sp.Expr) -> bool:
        """Check if ODE has complex exponential terms"""
        return ode_expr.has(sp.I) or any(
            atom.has(sp.I) for atom in ode_expr.atoms() 
            if hasattr(atom, 'has')
        )
    
    def _verify_pantograph(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr
    ) -> Tuple[bool, VerificationMethod, float]:
        """
        Special verification for pantograph equations
        """
        try:
            y = SYMBOLS.y
            x = SYMBOLS.x
            
            # Detect delay factor
            delay_factor = self._detect_delay_factor(ode_expr)
            
            # Build comprehensive substitution dictionary
            substitutions = {y(x): solution_expr}
            
            # Add pantograph terms
            for atom in ode_expr.atoms():
                if isinstance(atom, sp.Function) and atom.func == y:
                    if atom.args and atom.args[0] != x:
                        arg = atom.args[0]
                        # Substitute in solution
                        substitutions[atom] = solution_expr.subs(x, arg)
            
            # Add derivatives
            max_order = self._get_max_derivative_order(ode_expr)
            for order in range(1, max_order + 1):
                substitutions[y(x).diff(x, order)] = sp.diff(solution_expr, x, order)
            
            # Apply substitutions
            lhs_sub = ode_expr.lhs.subs(substitutions)
            rhs_sub = ode_expr.rhs.subs(substitutions)
            residual = lhs_sub - rhs_sub
            
            # Try simplification strategies
            strategies = [
                lambda r: r,
                lambda r: sp.simplify(r),
                lambda r: sp.expand(r),
                lambda r: sp.trigsimp(r) if r.has(sp.sin, sp.cos) else r,
                lambda r: sp.radsimp(r),
                lambda r: sp.cancel(r)
            ]
            
            for strategy in strategies:
                try:
                    simplified = strategy(residual)
                    if simplified == 0 or simplified.equals(0):
                        return True, VerificationMethod.SUBSTITUTION, 1.0
                except:
                    continue
            
            # Numerical verification at pantograph-specific points
            residuals = []
            for test_x in self.pantograph_points:
                try:
                    numeric_residual = float(residual.subs(x, test_x).evalf())
                    residuals.append(abs(numeric_residual))
                except:
                    continue
            
            if residuals:
                max_residual = max(residuals)
                if max_residual < self.residual_tolerance:
                    confidence = 1.0 - (max_residual / self.residual_tolerance)
                    return True, VerificationMethod.NUMERIC, confidence * 0.8
            
            return False, VerificationMethod.FAILED, 0.0
            
        except Exception as e:
            logger.error(f"Pantograph verification error: {e}")
            return False, VerificationMethod.FAILED, 0.0
    
    def _verify_higher_order(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr
    ) -> Tuple[bool, VerificationMethod, float]:
        """
        Verification for higher-order ODEs (order > 2)
        """
        try:
            y = SYMBOLS.y
            x = SYMBOLS.x
            
            max_order = self._get_max_derivative_order(ode_expr)
            
            # Build substitutions for all derivatives
            substitutions = {y(x): solution_expr}
            
            for order in range(1, max_order + 1):
                substitutions[y(x).diff(x, order)] = sp.diff(solution_expr, x, order)
            
            # Apply substitutions
            lhs_sub = ode_expr.lhs.subs(substitutions)
            rhs_sub = ode_expr.rhs.subs(substitutions)
            residual = lhs_sub - rhs_sub
            
            # Series expansion verification
            try:
                # Expand around x=0
                series_residual = residual.series(x, 0, self.series_order)
                
                # Check if all coefficients are zero
                coeffs_zero = True
                for i in range(self.series_order):
                    coeff = series_residual.coeff(x, i)
                    if coeff and abs(complex(coeff.evalf())) > self.residual_tolerance:
                        coeffs_zero = False
                        break
                
                if coeffs_zero:
                    return True, VerificationMethod.SUBSTITUTION, 0.95
            except:
                pass
            
            # Numerical verification with adaptive points
            test_points = np.linspace(0.01, 2.0, 20)
            residuals = []
            
            for test_x in test_points:
                try:
                    numeric_residual = abs(complex(residual.subs(x, test_x).evalf()))
                    residuals.append(numeric_residual)
                except:
                    continue
            
            if residuals:
                max_residual = max(residuals)
                avg_residual = np.mean(residuals)
                
                if max_residual < self.residual_tolerance:
                    confidence = 1.0 - (avg_residual / self.residual_tolerance)
                    return True, VerificationMethod.NUMERIC, confidence * 0.7
            
            return False, VerificationMethod.FAILED, 0.0
            
        except Exception as e:
            logger.error(f"Higher-order verification error: {e}")
            return False, VerificationMethod.FAILED, 0.0
    
    def _verify_complex(
        self, 
        ode_expr: sp.Expr, 
        solution_expr: sp.Expr
    ) -> Tuple[bool, VerificationMethod, float]:
        """
        Verification for ODEs with complex terms
        """
        try:
            y = SYMBOLS.y
            x = SYMBOLS.x
            
            # Separate real and imaginary parts
            solution_real = sp.re(solution_expr)
            solution_imag = sp.im(solution_expr)
            
            # Build substitutions
            substitutions = {y(x): solution_expr}
            
            max_order = self._get_max_derivative_order(ode_expr)
            for order in range(1, max_order + 1):
                substitutions[y(x).diff(x, order)] = sp.diff(solution_expr, x, order)
            
            # Apply substitutions
            lhs_sub = ode_expr.lhs.subs(substitutions)
            rhs_sub = ode_expr.rhs.subs(substitutions)
            
            # Check real and imaginary parts separately
            residual_real = sp.re(lhs_sub - rhs_sub)
            residual_imag = sp.im(lhs_sub - rhs_sub)
            
            # Simplify
            try:
                residual_real = sp.simplify(residual_real)
                residual_imag = sp.simplify(residual_imag)
                
                if residual_real == 0 and residual_imag == 0:
                    return True, VerificationMethod.SUBSTITUTION, 1.0
            except:
                pass
            
            # Numerical check
            test_points = np.linspace(0.1, 2.0, 10)
            max_residual = 0
            
            for test_x in test_points:
                try:
                    real_val = abs(float(residual_real.subs(x, test_x).evalf()))
                    imag_val = abs(float(residual_imag.subs(x, test_x).evalf()))
                    
                    total_residual = np.sqrt(real_val**2 + imag_val**2)
                    max_residual = max(max_residual, total_residual)
                except:
                    continue
            
            if max_residual < self.complex_tolerance:
                confidence = 1.0 - (max_residual / self.complex_tolerance)
                return True, VerificationMethod.NUMERIC, confidence * 0.85
            
            return False, VerificationMethod.FAILED, 0.0
            
        except Exception as e:
            logger.error(f"Complex verification error: {e}")
            return False, VerificationMethod.FAILED, 0.0
    
    def _detect_delay_factor(self, ode_expr: sp.Expr) -> float:
        """Detect the delay factor in pantograph equations"""
        y = SYMBOLS.y
        x = SYMBOLS.x
        
        for atom in ode_expr.atoms():
            if isinstance(atom, sp.Function) and atom.func == y:
                if atom.args and atom.args[0] != x:
                    arg = atom.args[0]
                    # Check if it's of form x/a
                    if arg.is_Mul:
                        for factor in arg.as_coeff_Mul():
                            if factor.has(x):
                                ratio = arg / x
                                if ratio.is_constant():
                                    return float(1/ratio)
        return 2.0  # Default delay factor
    
    def verify_batch(
        self, 
        ode_solution_pairs: List[Tuple[sp.Expr, sp.Expr]]
    ) -> List[Tuple[bool, VerificationMethod, float]]:
        """
        Batch verification with parallel processing
        """
        results = []
        
        # Use multiprocessing for batch verification
        import multiprocessing as mp
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            async_results = []
            
            for ode, solution in ode_solution_pairs:
                async_result = pool.apply_async(
                    self.verify,
                    args=(ode, solution)
                )
                async_results.append(async_result)
            
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=self.timeout_seconds)
                    results.append(result)
                except mp.TimeoutError:
                    results.append((False, VerificationMethod.FAILED, 0.0))
                except Exception as e:
                    logger.error(f"Batch verification error: {e}")
                    results.append((False, VerificationMethod.FAILED, 0.0))
        
        return results
