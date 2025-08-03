import sympy as sp
from typing import Dict, Callable, Any, List
import logging

logger = logging.getLogger(__name__)

class AnalyticFunctionLibrary:
    """Domain-safe analytic functions with error handling"""
    
    @staticmethod
    def get_safe_library() -> Dict[str, Callable]:
        """Return domain-safe function library with enhanced coverage"""
        # Note: We use lambda functions to ensure proper symbolic handling
        return {
            # Basic polynomial functions (always safe)
            "identity": lambda z: z,
            "quadratic": lambda z: z**2,
            "cubic": lambda z: z**3,
            "quartic": lambda z: z**4 - 3*z**2 + 2,
            "quintic": lambda z: z**5 - 5*z**3 + 4*z,
            
            # Exponential family (always safe)
            "exponential": lambda z: sp.exp(z),
            "exp_scaled": lambda z: sp.exp(z/2),
            "exp_quadratic": lambda z: sp.exp(z**2/4),
            "exp_negative": lambda z: sp.exp(-z),
            
            # Trigonometric family (always safe)
            "sine": lambda z: sp.sin(z),
            "cosine": lambda z: sp.cos(z),
            "tangent_safe": lambda z: sp.sin(z)/(sp.cos(z) + sp.S(1)/10),
            "sine_scaled": lambda z: sp.sin(z/2),
            "cosine_scaled": lambda z: sp.cos(z/3),
            "sine_squared": lambda z: sp.sin(z)**2,
            "cosine_squared": lambda z: sp.cos(z)**2,
            
            # Hyperbolic family (always safe)
            "sinh": lambda z: sp.sinh(z),
            "cosh": lambda z: sp.cosh(z),
            "tanh": lambda z: sp.tanh(z/2),
            "sinh_scaled": lambda z: sp.sinh(z/3),
            "cosh_scaled": lambda z: sp.cosh(z/4),
            
            # Domain-safe logarithmic functions
            "log_safe": lambda z: sp.log(sp.Abs(z) + sp.S(1)/10),
            "log_shifted": lambda z: sp.log(z**2 + 1),
            "log_scaled": lambda z: sp.log(sp.Abs(z/2) + 1),
            
            # Safe rational functions
            "rational_simple": lambda z: z/(z**2 + 1),
            "rational_stable": lambda z: (z**2 + 1)/(z**4 + z**2 + 1),
            "rational_cubic": lambda z: z**3/(z**4 + 1),
            
            # Composite functions for advanced testing
            "exp_sin": lambda z: sp.exp(sp.sin(z)/2),
            "sin_exp": lambda z: sp.sin(sp.exp(z/4)),
            "gaussian": lambda z: sp.exp(-z**2/4),
            "bessel_like": lambda z: sp.sin(z)/sp.sqrt(sp.Abs(z) + 1),
            
            # Special functions
            "erf_approx": lambda z: 2/sp.sqrt(sp.pi) * z * sp.exp(-z**2/4),
            "sigmoid": lambda z: 1/(1 + sp.exp(-z)),
            "softplus": lambda z: sp.log(1 + sp.exp(z/2))
        }
    
    @staticmethod
    def validate_function(func_name: str, test_value: float = 1.0) -> bool:
        """Validate that a function can be evaluated safely"""
        try:
            library = AnalyticFunctionLibrary.get_safe_library()
            if func_name not in library:
                return False
            
            func = library[func_name]
            result = func(test_value)
            
            # Handle complex results properly
            numeric_result = result.evalf()
            
            if numeric_result.is_real:
                value = float(numeric_result)
            else:
                value = complex(numeric_result)
                
            # Check if finite
            import numpy as np
            if isinstance(value, complex):
                return np.isfinite(value.real) and np.isfinite(value.imag)
            else:
                return np.isfinite(value)
                
        except Exception as e:
            logger.debug(f"Function validation failed for {func_name}: {e}")
            return False
    
    @staticmethod
    def get_function_info(func_name: str) -> Dict[str, Any]:
        """Get information about a specific function"""
        library = AnalyticFunctionLibrary.get_safe_library()
        
        if func_name not in library:
            return {
                'exists': False,
                'name': func_name,
                'error': 'Function not found in library'
            }
        
        func = library[func_name]
        
        # Categorize the function
        category = 'unknown'
        if any(x in func_name for x in ['identity', 'quadratic', 'cubic', 'quartic', 'quintic']):
            category = 'polynomial'
        elif 'exp' in func_name:
            category = 'exponential'
        elif any(x in func_name for x in ['sin', 'cos', 'tan']):
            category = 'trigonometric'
        elif any(x in func_name for x in ['sinh', 'cosh', 'tanh']):
            category = 'hyperbolic'
        elif 'log' in func_name:
            category = 'logarithmic'
        elif 'rational' in func_name:
            category = 'rational'
        else:
            category = 'composite'
        
        # Test domain safety
        test_values = [-10, -1, -0.1, 0, 0.1, 1, 10]
        domain_safe = True
        
        for val in test_values:
            if not AnalyticFunctionLibrary.validate_function(func_name, val):
                domain_safe = False
                break
        
        return {
            'exists': True,
            'name': func_name,
            'category': category,
            'domain_safe': domain_safe,
            'has_parameters': '/' in str(func(sp.Symbol('z')))
        }
    
    @staticmethod
    def list_functions_by_category() -> Dict[str, List[str]]:
        """List all functions organized by category"""
        library = AnalyticFunctionLibrary.get_safe_library()
        
        categories = {
            'polynomial': [],
            'exponential': [],
            'trigonometric': [],
            'hyperbolic': [],
            'logarithmic': [],
            'rational': [],
            'composite': [],
            'special': []
        }
        
        for func_name in library.keys():
            info = AnalyticFunctionLibrary.get_function_info(func_name)
            category = info.get('category', 'special')
            if category in categories:
                categories[category].append(func_name)
            else:
                categories['special'].append(func_name)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    @staticmethod
    def evaluate_at_point(func_name: str, point: complex) -> complex:
        """Safely evaluate a function at a given point"""
        library = AnalyticFunctionLibrary.get_safe_library()
        
        if func_name not in library:
            raise ValueError(f"Function {func_name} not found in library")
        
        func = library[func_name]
        
        try:
            result = func(point)
            numeric_result = complex(result.evalf())
            return numeric_result
        except Exception as e:
            logger.error(f"Error evaluating {func_name} at {point}: {e}")
            raise
    
    @staticmethod
    def get_derivative(func_name: str, order: int = 1):
        """Get the symbolic derivative of a function"""
        library = AnalyticFunctionLibrary.get_safe_library()
        
        if func_name not in library:
            raise ValueError(f"Function {func_name} not found in library")
        
        func = library[func_name]
        z = sp.Symbol('z')
        
        # Get the symbolic expression
        expr = func(z)
        
        # Compute derivative
        for _ in range(order):
            expr = sp.diff(expr, z)
        
        # Return as a lambda function
        return lambda x: expr.subs(z, x)


# Convenience function for backward compatibility
def get_safe_analytic_functions() -> Dict[str, Callable]:
    """Get the library of safe analytic functions"""
    return AnalyticFunctionLibrary.get_safe_library()