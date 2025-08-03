import sympy as sp
from functools import lru_cache
from typing import Dict, Tuple, Optional
import logging
from core.symbols import SYMBOLS
from core.functions import AnalyticFunctionLibrary

logger = logging.getLogger(__name__)

class DerivativeComputer:
    """Optimized derivative computation with caching"""
    
    def __init__(self, max_order: int = 5, cache_size: int = 256):
        self.max_order = max_order
        self.cache_size = cache_size
        self.f_library = AnalyticFunctionLibrary.get_safe_library()
        
    @lru_cache(maxsize=256)
    def compute_all_derivatives(self, f_key: str) -> Dict[int, sp.Expr]:
        """Compute all derivatives up to max_order and cache them"""
        try:
            if f_key not in self.f_library:
                raise ValueError(f"Unknown function: {f_key}")
            
            f_func = self.f_library[f_key]
            
            # Create symbolic expressions
            f_at_alpha_beta = f_func(SYMBOLS.alpha + SYMBOLS.beta)
            f_at_alpha_beta_exp = f_func(SYMBOLS.alpha + SYMBOLS.beta * sp.exp(-SYMBOLS.x))
            
            # Compute all derivatives
            derivatives = {
                'f_at_alpha_beta': f_at_alpha_beta,
                'f_at_alpha_beta_exp': f_at_alpha_beta_exp,
                'derivatives': {}
            }
            
            derivatives['derivatives'][0] = f_at_alpha_beta_exp
            
            for order in range(1, self.max_order + 1):
                derivatives['derivatives'][order] = sp.diff(
                    f_at_alpha_beta_exp, SYMBOLS.alpha, order
                )
            
            return derivatives
            
        except Exception as e:
            logger.error(f"Error computing derivatives for {f_key}: {e}")
            return None
    
    def get_derivatives(self, f_key: str, max_order: int) -> Tuple[sp.Expr, sp.Expr, Dict[int, sp.Expr]]:
        """Get derivatives up to specified order"""
        all_derivs = self.compute_all_derivatives(f_key)
        
        if all_derivs is None:
            return None, None, {i: sp.S(0) for i in range(max_order + 1)}
        
        # Extract only needed derivatives
        needed_derivs = {
            i: all_derivs['derivatives'][i] 
            for i in range(min(max_order + 1, self.max_order + 1))
        }
        
        return (
            all_derivs['f_at_alpha_beta'],
            all_derivs['f_at_alpha_beta_exp'],
            needed_derivs
        )

    @staticmethod
    def substitute_parameters(expression: sp.Expr, params: Dict[str, float]) -> sp.Expr:
        """Substitute numerical parameter values into symbolic expressions"""
        try:
            substitutions = {
                SYMBOLS.alpha: float(params.get('alpha', 0)),
                SYMBOLS.beta: float(params.get('beta', 1)),
                SYMBOLS.M: float(params.get('M', 0))
            }
            
            # Add any additional parameters
            if 'a' in params:
                substitutions[SYMBOLS.a] = float(params['a'])
            if 'q' in params:
                substitutions[SYMBOLS.q] = float(params['q'])
            if 'v' in params:
                substitutions[SYMBOLS.v] = float(params['v'])
            
            return expression.subs(substitutions)
            
        except Exception as e:
            logger.error(f"Error in parameter substitution: {e}")
            return expression