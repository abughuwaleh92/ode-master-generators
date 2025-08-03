import sympy as sp
from typing import Dict, Tuple
import logging
from generators.base import BaseGenerator
from core.types import GeneratorType
from core.symbols import SYMBOLS

logger = logging.getLogger(__name__)

class LinearGeneratorL1(BaseGenerator):
    """Linear Generator L1: y''(x) + y(x) = RHS"""
    
    def __init__(self):
        super().__init__("L1", GeneratorType.LINEAR)
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, 2)
            
            if f_at_alpha_beta is None:
                return None, None, None
            
            # Build RHS symbolically
            term1 = f_at_alpha_beta
            term2 = -f_at_alpha_beta_exp
            term3 = SYMBOLS.M
            term4 = -SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1]
            term5 = -SYMBOLS.beta**2 * sp.exp(-2*SYMBOLS.x) * derivatives[2]
            
            rhs_symbolic = sp.pi * (term1 + term2 + term3 + term4 + term5)
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x).diff(x, 2) + y(x)
            ode = sp.Eq(lhs, rhs)
            
            # Create solution
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp + SYMBOLS.M)
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
            logger.error(f"Error in L1 generator: {e}")
            return None, None, None


class LinearGeneratorL2(BaseGenerator):
    """Linear Generator L2: y''(x) + y'(x) = RHS"""
    
    def __init__(self):
        super().__init__("L2", GeneratorType.LINEAR)
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, 2)
            
            if f_at_alpha_beta is None:
                return None, None, None
            
            # Build RHS
            rhs_symbolic = -sp.pi * SYMBOLS.beta**2 * sp.exp(-2*SYMBOLS.x) * derivatives[2]
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x).diff(x, 2) + y(x).diff(x)
            ode = sp.Eq(lhs, rhs)
            
            # Solution (same as L1)
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp + SYMBOLS.M)
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
            logger.error(f"Error in L2 generator: {e}")
            return None, None, None


class LinearGeneratorL3(BaseGenerator):
    """Linear Generator L3: y(x) + y'(x) = RHS"""
    
    def __init__(self):
        super().__init__("L3", GeneratorType.LINEAR)
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, 1)
            
            if f_at_alpha_beta is None:
                return None, None, None
            
            # Build RHS
            rhs_symbolic = sp.pi * (
                f_at_alpha_beta - f_at_alpha_beta_exp + SYMBOLS.M +
                SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1]
            )
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x) + y(x).diff(x)
            ode = sp.Eq(lhs, rhs)
            
            # Solution
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp + SYMBOLS.M)
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Initial conditions
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            ics = {'y(0)': str(ic_y0)}
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in L3 generator: {e}")
            return None, None, None


class LinearGeneratorL4Pantograph(BaseGenerator):
    """Linear Generator L4: Pantograph equation y''(x) + y(x/a) - y(x) = RHS"""
    
    def __init__(self):
        super().__init__("L4", GeneratorType.LINEAR)
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            # Get pantograph parameter
            a_val = params.get('a', 2)
            
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, 2)
            
            if f_at_alpha_beta is None:
                return None, None, None
            
            # Get the function from library for scaled version
            f_func = self.derivative_computer.f_library[f_key]
            f_at_alpha_beta_exp_scaled = f_func(
                SYMBOLS.alpha + SYMBOLS.beta * sp.exp(-SYMBOLS.x/a_val)
            )
            
            # Build RHS
            rhs_symbolic = sp.pi * (
                f_at_alpha_beta_exp - f_at_alpha_beta_exp_scaled -
                SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1] -
                SYMBOLS.beta**2 * sp.exp(-2*SYMBOLS.x) * derivatives[2]
            )
            
            # Substitute parameters
            substitutions = {
                SYMBOLS.alpha: params['alpha'],
                SYMBOLS.beta: params['beta'],
                SYMBOLS.M: params['M']
            }
            rhs = rhs_symbolic.subs(substitutions)
            
            # Create ODE with pantograph term
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x).diff(x, 2) + y(x/a_val) - y(x)
            ode = sp.Eq(lhs, rhs)
            
            # Solution
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp + SYMBOLS.M)
            solution = solution_symbolic.subs(substitutions)
            
            # Initial conditions
            ic_y0 = (sp.pi * SYMBOLS.M).subs(substitutions)
            ic_y1_symbolic = sp.pi * SYMBOLS.beta * sp.diff(f_at_alpha_beta, SYMBOLS.alpha)
            ic_y1 = ic_y1_symbolic.subs(substitutions)
            
            ics = {
                'y(0)': str(ic_y0),
                "y'(0)": str(ic_y1)
            }
            
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in L4 pantograph generator: {e}")
            return None, None, None


# Registry of linear generators
LINEAR_GENERATORS = {
    'L1': LinearGeneratorL1,
    'L2': LinearGeneratorL2,
    'L3': LinearGeneratorL3,
    'L4': LinearGeneratorL4Pantograph
}