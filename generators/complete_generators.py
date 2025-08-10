# generators/complete_generators.py
"""
Complete implementation of all generators from the paper
"""

import sympy as sp
from typing import Dict, Tuple, Optional
import numpy as np
from generators.base import BaseGenerator
from core.types import GeneratorType
from core.symbols import SYMBOLS
from core.master_theorems import MasterTheorems

class LinearGeneratorL5(BaseGenerator):
    """
    Generator L5: y(x/a) + y'(x) = RHS
    Pantograph equation with first derivative
    """
    
    def __init__(self):
        super().__init__("L5", GeneratorType.LINEAR)
        self.master_theorems = MasterTheorems()
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            a_val = params.get('a', 2)
            
            # Get function and derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, 1)
            
            if f_at_alpha_beta is None:
                return None, None, None
            
            # Get scaled version
            f_func = self.derivative_computer.f_library[f_key]
            f_at_alpha_beta_exp_scaled = f_func(
                SYMBOLS.alpha + SYMBOLS.beta * sp.exp(-SYMBOLS.x/a_val)
            )
            
            # Build RHS
            rhs_symbolic = sp.pi * (
                f_at_alpha_beta - f_at_alpha_beta_exp_scaled + SYMBOLS.M +
                SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1]
            )
            
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x/a_val) + y(x).diff(x)
            ode = sp.Eq(lhs, rhs)
            
            # Solution
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp + SYMBOLS.M)
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Initial conditions
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            
            ics = {'y(0)': str(ic_y0)}
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in L5 generator: {e}")
            return None, None, None


class LinearGeneratorL6(BaseGenerator):
    """
    Generator L6: y'''(x) + y(x) = RHS
    Third-order linear ODE using Corollary 4.2.1
    """
    
    def __init__(self):
        super().__init__("L6", GeneratorType.LINEAR)
        self.master_theorems = MasterTheorems()
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            n = 2  # For third-order, use n=2
            
            # Get function
            f_func = self.derivative_computer.f_library[f_key]
            
            # Build solution using Theorem 4.1
            solution_base, derivatives = self.master_theorems.theorem_4_1_solution(
                f_func, SYMBOLS.alpha, SYMBOLS.beta, SYMBOLS.x, n
            )
            
            # Add constant term
            solution_symbolic = solution_base + sp.pi * SYMBOLS.M
            
            # Get third derivative using Corollary 4.2.1
            third_deriv = derivatives.get(3, sp.S(0))
            
            # Build RHS
            rhs_symbolic = third_deriv + solution_symbolic
            
            # Substitute parameters
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x).diff(x, 3) + y(x)
            ode = sp.Eq(lhs, rhs)
            
            # Initial conditions
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            ic_y1 = self.derivative_computer.substitute_parameters(
                sp.pi * SYMBOLS.beta * sp.cos(sp.pi/(2*n)) / sp.sqrt(2), 
                params
            )
            ic_y2 = sp.S(0)
            
            ics = {
                'y(0)': str(ic_y0),
                "y'(0)": str(ic_y1),
                "y''(0)": str(ic_y2)
            }
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in L6 generator: {e}")
            return None, None, None


class LinearGeneratorL7(BaseGenerator):
    """
    Generator L7: y'''(x) + y'(x) = RHS
    Mixed third and first order derivatives
    """
    
    def __init__(self):
        super().__init__("L7", GeneratorType.LINEAR)
        self.master_theorems = MasterTheorems()
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            n = 2
            
            # Get function and build solution
            f_func = self.derivative_computer.f_library[f_key]
            solution_base, derivatives = self.master_theorems.theorem_4_1_solution(
                f_func, SYMBOLS.alpha, SYMBOLS.beta, SYMBOLS.x, n
            )
            
            solution_symbolic = solution_base + sp.pi * SYMBOLS.M
            
            # Build RHS combining third and first derivatives
            third_deriv = derivatives.get(3, sp.S(0))
            first_deriv = derivatives.get(1, sp.S(0))
            
            rhs_symbolic = third_deriv + first_deriv
            
            # Substitute parameters
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x).diff(x, 3) + y(x).diff(x)
            ode = sp.Eq(lhs, rhs)
            
            # Initial conditions
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            ic_y1 = self.derivative_computer.substitute_parameters(
                sp.pi * SYMBOLS.beta * sp.cos(sp.pi/(2*n)) / sp.sqrt(2), 
                params
            )
            ic_y2 = sp.S(0)
            
            ics = {
                'y(0)': str(ic_y0),
                "y'(0)": str(ic_y1),
                "y''(0)": str(ic_y2)
            }
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in L7 generator: {e}")
            return None, None, None


class LinearGeneratorL8(BaseGenerator):
    """
    Generator L8: y'''(x) + y''(x) = RHS
    Mixed third and second order derivatives
    """
    
    def __init__(self):
        super().__init__("L8", GeneratorType.LINEAR)
        self.master_theorems = MasterTheorems()
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            n = 2
            
            # Get function and build solution
            f_func = self.derivative_computer.f_library[f_key]
            solution_base, derivatives = self.master_theorems.theorem_4_1_solution(
                f_func, SYMBOLS.alpha, SYMBOLS.beta, SYMBOLS.x, n
            )
            
            solution_symbolic = solution_base + sp.pi * SYMBOLS.M
            
            # Build RHS combining third and second derivatives
            third_deriv = derivatives.get(3, sp.S(0))
            second_deriv = derivatives.get(2, sp.S(0))
            
            rhs_symbolic = third_deriv + second_deriv
            
            # Substitute parameters
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x).diff(x, 3) + y(x).diff(x, 2)
            ode = sp.Eq(lhs, rhs)
            
            # Initial conditions
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            ic_y1 = self.derivative_computer.substitute_parameters(
                sp.pi * SYMBOLS.beta * sp.cos(sp.pi/(2*n)) / sp.sqrt(2), 
                params
            )
            ic_y2 = sp.S(0)
            
            ics = {
                'y(0)': str(ic_y0),
                "y'(0)": str(ic_y1),
                "y''(0)": str(ic_y2)
            }
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in L8 generator: {e}")
            return None, None, None


class NonlinearGeneratorN8(NonlinearGenerator):
    """
    Generator N8: sin(y''(x)) + y(x) = RHS
    Trigonometric nonlinearity in second derivative
    """
    
    def __init__(self):
        super().__init__("N8")
    
    def _get_required_derivatives(self) -> int:
        return 2
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            params = self._validate_parameters(params)
            
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, self._get_required_derivatives())
            
            if f_at_alpha_beta is None:
                return None, None, None
            
            # Build the derivative term
            deriv_term = (
                -SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1] -
                SYMBOLS.beta**2 * sp.exp(-2*SYMBOLS.x) * derivatives[2]
            )
            
            # Build RHS
            base_term = f_at_alpha_beta - f_at_alpha_beta_exp
            rhs_symbolic = sp.pi * base_term + SYMBOLS.M + sp.sin(sp.pi * deriv_term)
            
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = sp.sin(y(x).diff(x, 2)) + y(x)
            ode = sp.Eq(lhs, rhs)
            
            # Solution
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp) + SYMBOLS.M
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Initial conditions
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            ic_y1_symbolic = sp.pi * SYMBOLS.beta * sp.diff(f_at_alpha_beta, SYMBOLS.alpha)
            ic_y1 = self.derivative_computer.substitute_parameters(ic_y1_symbolic, params)
            
            ics = {
                'y(0)': str(ic_y0),
                "y'(0)": str(ic_y1)
            }
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in N8 generator: {e}")
            return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        from core.types import NonlinearityMetrics
        return NonlinearityMetrics(
            pow_deriv_max=1,
            pow_yprime=1,
            has_pantograph=False,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=1
        )


class NonlinearGeneratorN9(NonlinearGenerator):
    """
    Generator N9: e^(y''(x)) + e^(y'(x)) = RHS
    Exponential nonlinearity in derivatives
    """
    
    def __init__(self):
        super().__init__("N9")
    
    def _get_required_derivatives(self) -> int:
        return 2
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            params = self._validate_parameters(params)
            
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, self._get_required_derivatives())
            
            if f_at_alpha_beta is None:
                return None, None, None
            
            # Build derivative terms
            second_deriv_term = (
                -SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1] -
                SYMBOLS.beta**2 * sp.exp(-2*SYMBOLS.x) * derivatives[2]
            )
            first_deriv_term = SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1]
            
            # Build RHS
            rhs_symbolic = (
                sp.exp(sp.pi * second_deriv_term) + 
                sp.exp(sp.pi * first_deriv_term)
            )
            
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = sp.exp(y(x).diff(x, 2)) + sp.exp(y(x).diff(x))
            ode = sp.Eq(lhs, rhs)
            
            # Solution
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp) + SYMBOLS.M
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Initial conditions
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            ic_y1_symbolic = sp.pi * SYMBOLS.beta * sp.diff(f_at_alpha_beta, SYMBOLS.alpha)
            ic_y1 = self.derivative_computer.substitute_parameters(ic_y1_symbolic, params)
            
            ics = {
                'y(0)': str(ic_y0),
                "y'(0)": str(ic_y1)
            }
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in N9 generator: {e}")
            return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        from core.types import NonlinearityMetrics
        return NonlinearityMetrics(
            pow_deriv_max=1,
            pow_yprime=1,
            has_pantograph=False,
            is_exponential_nonlinear=True,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=np.inf
        )


class NonlinearGeneratorN10(NonlinearGenerator):
    """
    Generator N10: ln(y'(x)) + y(x/a) = RHS
    Logarithmic nonlinearity with pantograph term
    """
    
    def __init__(self):
        super().__init__("N10")
    
    def _get_required_derivatives(self) -> int:
        return 1
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            params = self._validate_parameters(params)
            a_val = params.get('a', 2)
            
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, self._get_required_derivatives())
            
            if f_at_alpha_beta is None:
                return None, None, None
            
            # Get scaled version for pantograph
            f_func = self.derivative_computer.f_library[f_key]
            f_at_alpha_beta_exp_scaled = f_func(
                SYMBOLS.alpha + SYMBOLS.beta * sp.exp(-SYMBOLS.x/a_val)
            )
            
            # Build derivative term
            first_deriv_term = SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1]
            
            # Build RHS
            rhs_symbolic = (
                sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp_scaled) + 
                SYMBOLS.M + 
                sp.log(sp.pi * first_deriv_term)
            )
            
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = sp.log(y(x).diff(x)) + y(x/a_val)
            ode = sp.Eq(lhs, rhs)
            
            # Solution
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp) + SYMBOLS.M
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Initial conditions
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            ic_y1_symbolic = sp.pi * SYMBOLS.beta * sp.diff(f_at_alpha_beta, SYMBOLS.alpha)
            ic_y1 = self.derivative_computer.substitute_parameters(ic_y1_symbolic, params)
            
            ics = {
                'y(0)': str(ic_y0),
                "y'(0)": str(ic_y1)
            }
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in N10 generator: {e}")
            return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        from core.types import NonlinearityMetrics
        return NonlinearityMetrics(
            pow_deriv_max=1,
            pow_yprime=1,
            has_pantograph=True,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=True,
            total_nonlinear_degree=np.inf
        )
