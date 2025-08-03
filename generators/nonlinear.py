# generators/nonlinear.py
"""
Nonlinear ODE Generators

This module implements all nonlinear ODE generators based on the master theorems.
Each generator creates ODEs with specific nonlinear structures and known exact solutions.
"""

import sympy as sp
from typing import Dict, Tuple, Optional, List, Any
import logging
from abc import abstractmethod
import numpy as np

from generators.base import BaseGenerator
from core.types import GeneratorType, NonlinearityMetrics
from core.symbols import SYMBOLS
from utils.derivatives import DerivativeComputer

logger = logging.getLogger(__name__)


class NonlinearGenerator(BaseGenerator):
    """Base class for all nonlinear generators with enhanced functionality"""
    
    def __init__(self, name: str):
        super().__init__(name, GeneratorType.NONLINEAR)
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate generator configuration"""
        pass
    
    @abstractmethod
    def _get_required_derivatives(self) -> int:
        """Return the maximum derivative order needed"""
        pass
    
    def _validate_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """Validate and sanitize parameters for nonlinear generators"""
        validated = params.copy()
        
        # Ensure nonlinear parameters are within safe ranges
        if 'q' in validated:
            q_val = validated['q']
            if not (1 <= q_val <= 10):
                logger.warning(f"Parameter q={q_val} outside safe range [1,10], clamping")
                validated['q'] = max(1, min(10, q_val))
        
        if 'v' in validated:
            v_val = validated['v']
            if not (1 <= v_val <= 10):
                logger.warning(f"Parameter v={v_val} outside safe range [1,10], clamping")
                validated['v'] = max(1, min(10, v_val))
        
        return validated
    
    def _build_nonlinear_term(self, base_expr: sp.Expr, power: int) -> sp.Expr:
        """Safely build nonlinear terms with overflow protection"""
        try:
            if power == 1:
                return base_expr
            
            # Check if base expression might cause numerical issues
            if base_expr.has(sp.exp):
                # For exponential terms, limit the power to prevent overflow
                max_safe_power = 5
                if power > max_safe_power:
                    logger.warning(f"Reducing power from {power} to {max_safe_power} for stability")
                    power = max_safe_power
            
            return base_expr**power
            
        except Exception as e:
            logger.error(f"Error building nonlinear term: {e}")
            return base_expr
    
    @abstractmethod
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        """Compute metrics specific to this nonlinear generator"""
        pass


class NonlinearGeneratorN1(NonlinearGenerator):
    """
    Nonlinear Generator N1: (y''(x))^q + y(x) = RHS
    
    This generator creates ODEs where the second derivative is raised to power q.
    Particularly useful for modeling nonlinear wave equations and reaction-diffusion systems.
    """
    
    def __init__(self):
        super().__init__("N1")
        
    def _get_required_derivatives(self) -> int:
        return 2
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            params = self._validate_parameters(params)
            q_val = params.get('q', 2)
            
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, self._get_required_derivatives())
            
            if f_at_alpha_beta is None:
                logger.error(f"Failed to compute derivatives for function {f_key}")
                return None, None, None
            
            # Build RHS components
            base_term = f_at_alpha_beta - f_at_alpha_beta_exp
            
            # Build the nonlinear derivative term
            deriv_term = (
                -SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1] -
                SYMBOLS.beta**2 * sp.exp(-2*SYMBOLS.x) * derivatives[2]
            )
            
            nonlinear_term = self._build_nonlinear_term(deriv_term, q_val)
            
            # Complete RHS
            rhs_symbolic = sp.pi * base_term + SYMBOLS.M + sp.pi**q_val * nonlinear_term
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = self._build_nonlinear_term(y(x).diff(x, 2), q_val) + y(x)
            ode = sp.Eq(lhs, rhs)
            
            # Solution (same for all N generators)
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
            
            logger.debug(f"N1 generated successfully with q={q_val}")
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in N1 generator: {e}")
            return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        """N1-specific nonlinearity metrics"""
        return NonlinearityMetrics(
            pow_deriv_max=params.get('q', 2),
            pow_yprime=1,
            has_pantograph=False,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=params.get('q', 2)
        )


class NonlinearGeneratorN2(NonlinearGenerator):
    """
    Nonlinear Generator N2: (y''(x))^q + (y'(x))^v = RHS
    
    This generator creates ODEs with powers of both first and second derivatives.
    Useful for modeling systems with nonlinear damping and nonlinear restoring forces.
    """
    
    def __init__(self):
        super().__init__("N2")
        
    def _get_required_derivatives(self) -> int:
        return 2
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            params = self._validate_parameters(params)
            q_val = params.get('q', 2)
            v_val = params.get('v', 3)
            
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, self._get_required_derivatives())
            
            if f_at_alpha_beta is None:
                logger.error(f"Failed to compute derivatives for function {f_key}")
                return None, None, None
            
            # Build RHS terms
            second_deriv_term = (
                -SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1] -
                SYMBOLS.beta**2 * sp.exp(-2*SYMBOLS.x) * derivatives[2]
            )
            first_deriv_term = SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1]
            
            # Build nonlinear terms
            term1 = self._build_nonlinear_term(second_deriv_term, q_val)
            term2 = self._build_nonlinear_term(first_deriv_term, v_val)
            
            rhs_symbolic = sp.pi**q_val * term1 + sp.pi**v_val * term2
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = (
                self._build_nonlinear_term(y(x).diff(x, 2), q_val) + 
                self._build_nonlinear_term(y(x).diff(x), v_val)
            )
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
            
            logger.debug(f"N2 generated successfully with q={q_val}, v={v_val}")
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in N2 generator: {e}")
            return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        """N2-specific nonlinearity metrics"""
        return NonlinearityMetrics(
            pow_deriv_max=params.get('q', 2),
            pow_yprime=params.get('v', 3),
            has_pantograph=False,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=params.get('q', 2) + params.get('v', 3)
        )


class NonlinearGeneratorN3(NonlinearGenerator):
    """
    Nonlinear Generator N3: y(x) + (y'(x))^v = RHS
    
    This generator creates first-order nonlinear ODEs.
    Useful for population dynamics, chemical kinetics, and other growth models.
    """
    
    def __init__(self):
        super().__init__("N3")
        
    def _get_required_derivatives(self) -> int:
        return 1
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            params = self._validate_parameters(params)
            v_val = params.get('v', 3)
            
            # Get derivatives (only need up to first order)
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, self._get_required_derivatives())
            
            if f_at_alpha_beta is None:
                logger.error(f"Failed to compute derivatives for function {f_key}")
                return None, None, None
            
            # Build RHS
            base_term = f_at_alpha_beta - f_at_alpha_beta_exp
            deriv_term = SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1]
            nonlinear_term = self._build_nonlinear_term(deriv_term, v_val)
            
            rhs_symbolic = sp.pi * base_term + SYMBOLS.M + sp.pi**v_val * nonlinear_term
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            lhs = y(x) + self._build_nonlinear_term(y(x).diff(x), v_val)
            ode = sp.Eq(lhs, rhs)
            
            # Solution
            solution_symbolic = sp.pi * (f_at_alpha_beta - f_at_alpha_beta_exp) + SYMBOLS.M
            solution = self.derivative_computer.substitute_parameters(solution_symbolic, params)
            
            # Initial conditions (only y(0) for first-order ODE)
            ic_y0 = self.derivative_computer.substitute_parameters(sp.pi * SYMBOLS.M, params)
            ics = {'y(0)': str(ic_y0)}
            
            logger.debug(f"N3 generated successfully with v={v_val}")
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in N3 generator: {e}")
            return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        """N3-specific nonlinearity metrics"""
        return NonlinearityMetrics(
            pow_deriv_max=1,
            pow_yprime=params.get('v', 3),
            has_pantograph=False,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=params.get('v', 3)
        )


# ---------------------------------------------------------------------------
# Fully‑implemented nonlinear generators N4, N5, N6
# (leave the imports that already exist at the top of nonlinear.py)
# ---------------------------------------------------------------------------

class NonlinearGeneratorN4(NonlinearGenerator):
    """
    N4 ─ Exponential nonlinearity in the second derivative:
          exp(y''(x)) + y(x) = RHS
    """

    def __init__(self) -> None:
        super().__init__("N4")

    # ------------------------------------------------------------------ #
    #  Mandatory abstract‑method implementations                          #
    # ------------------------------------------------------------------ #
    def _get_required_derivatives(self) -> int:
        return 2                                                     # need up to y''

    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        return NonlinearityMetrics(
            pow_deriv_max=1,                    # exponential, not a power
            pow_yprime=1,
            has_pantograph=False,
            is_exponential_nonlinear=True,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=np.inf       # exponential ⇒ “infinite” degree
        )

    # ------------------------------------------------------------------ #
    #  Generator proper                                                  #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        f_key : str,
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:

        params = self._validate_parameters(params)
        # ------------------------------------------------------------------
        # 1) obtain f(α+βe^{-x}) and its first two derivatives
        # ------------------------------------------------------------------
        f_at, f_at_exp, derivs = self.derivative_computer.get_derivatives(
            f_key, self._get_required_derivatives()
        )
        if f_at is None:
            logger.error(f"[N4] derivative generation failed for f='{f_key}'")
            return None, None, None

        # ------------------------------------------------------------------
        # 2) build RHS
        # ------------------------------------------------------------------
        base        = f_at - f_at_exp
        deriv_term  = (
            -SYMBOLS.beta * sp.exp(-SYMBOLS.x)  * derivs[1]
            -SYMBOLS.beta**2* sp.exp(-2*SYMBOLS.x)*derivs[2]
        )
        rhs_sym = sp.pi * base + SYMBOLS.M + sp.exp(sp.pi * deriv_term) - 1
        rhs     = self.derivative_computer.substitute_parameters(rhs_sym, params)

        # ------------------------------------------------------------------
        # 3) build ODE + closed‑form solution + ICs
        # ------------------------------------------------------------------
        y, x = SYMBOLS.y, SYMBOLS.x
        lhs  = sp.exp( y(x).diff(x, 2) ) + y(x)
        ode  = sp.Eq(lhs, rhs)

        sol_sym = sp.pi * (f_at - f_at_exp) + SYMBOLS.M
        sol     = self.derivative_computer.substitute_parameters(sol_sym, params)

        ic_y0   = self.derivative_computer.substitute_parameters(sp.pi*SYMBOLS.M, params)
        ic_y1   = self.derivative_computer.substitute_parameters(
                      sp.pi * SYMBOLS.beta * sp.diff(f_at, SYMBOLS.alpha), params
                  )
        ics = {'y(0)': str(ic_y0), "y'(0)": str(ic_y1)}

        logger.debug("[N4] instance created successfully")
        return ode, sol, ics


# ============================================================================

class NonlinearGeneratorN5(NonlinearGenerator):
    """
    N5 ─ Trigonometric nonlinearity:
          sin(y''(x)) + cos(y'(x)) + y(x) = RHS
    """

    def __init__(self) -> None:
        super().__init__("N5")

    # ------------------------------------------------------------------ #
    def _get_required_derivatives(self) -> int:
        return 2

    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        return NonlinearityMetrics(
            pow_deriv_max=1,
            pow_yprime   =1,
            has_pantograph=False,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=1         # bounded trig
        )

    def generate(
        self,
        f_key : str,
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:

        params = self._validate_parameters(params)
        f_at, f_at_exp, derivs = self.derivative_computer.get_derivatives(
            f_key, self._get_required_derivatives()
        )
        if f_at is None:
            logger.error(f"[N5] derivative generation failed for f='{f_key}'")
            return None, None, None

        base          = f_at - f_at_exp
        second_term   = (
            -SYMBOLS.beta * sp.exp(-SYMBOLS.x)  * derivs[1]
            -SYMBOLS.beta**2* sp.exp(-2*SYMBOLS.x)*derivs[2]
        )
        first_term    = SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivs[1]

        rhs_sym = (
            sp.pi * base + SYMBOLS.M
            + sp.sin(sp.pi * second_term)
            + sp.cos(sp.pi * first_term) - 1
        )
        rhs     = self.derivative_computer.substitute_parameters(rhs_sym, params)

        y, x = SYMBOLS.y, SYMBOLS.x
        lhs  = sp.sin( y(x).diff(x, 2) ) + sp.cos( y(x).diff(x) ) + y(x)
        ode  = sp.Eq(lhs, rhs)

        sol_sym = sp.pi * (f_at - f_at_exp) + SYMBOLS.M
        sol     = self.derivative_computer.substitute_parameters(sol_sym, params)

        ic_y0 = self.derivative_computer.substitute_parameters(sp.pi*SYMBOLS.M, params)
        ic_y1 = self.derivative_computer.substitute_parameters(
                    sp.pi * SYMBOLS.beta * sp.diff(f_at, SYMBOLS.alpha), params
                )
        ics = {'y(0)': str(ic_y0), "y'(0)": str(ic_y1)}

        logger.debug("[N5] instance created successfully")
        return ode, sol, ics


# ============================================================================

class NonlinearGeneratorN6Pantograph(NonlinearGenerator):
    """
    N6 ─ Non‑linear pantograph:
          (y''(x))^q + y(x/a)^v − y(x) = RHS
    """

    def __init__(self) -> None:
        super().__init__("N6")

    # ------------------------------------------------------------------ #
    def _get_required_derivatives(self) -> int:
        return 2

    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        return NonlinearityMetrics(
            pow_deriv_max=params.get('q', 2),
            pow_yprime   =params.get('v', 2),
            has_pantograph=True,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=params.get('q', 2) + params.get('v', 2)
        )

    def generate(
        self,
        f_key : str,
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:

        params = self._validate_parameters(params)
        q = params.get('q', 2)
        v = params.get('v', 2)
        a = params.get('a', 2)            # delay factor (>1)

        if a <= 1:
            logger.error("[N6] pantograph parameter a must be > 1")
            return None, None, None

        f_at, f_at_exp, derivs = self.derivative_computer.get_derivatives(
            f_key, self._get_required_derivatives()
        )
        if f_at is None:
            logger.error(f"[N6] derivative generation failed for f='{f_key}'")
            return None, None, None

        # f evaluated at shifted argument for RHS / pantograph term
        f_func   = self.derivative_computer.f_library[f_key]
        f_scaled = f_func( SYMBOLS.alpha + SYMBOLS.beta * sp.exp(-SYMBOLS.x/a) )

        base       = f_at_exp - f_scaled
        deriv_term = (
            -SYMBOLS.beta * sp.exp(-SYMBOLS.x)  * derivs[1]
            -SYMBOLS.beta**2* sp.exp(-2*SYMBOLS.x)*derivs[2]
        )

        rhs_sym = sp.pi * base + sp.pi**q * self._build_nonlinear_term(deriv_term, q)
        rhs     = self.derivative_computer.substitute_parameters(rhs_sym, params)

        # ------------------------------------------------------------------
        y, x = SYMBOLS.y, SYMBOLS.x
        lhs  = (
            self._build_nonlinear_term( y(x).diff(x, 2), q )
            + self._build_nonlinear_term( y(x/a), v )
            - y(x)
        )
        ode = sp.Eq(lhs, rhs)

        sol_sym = sp.pi * (f_at - f_at_exp) + SYMBOLS.M
        sol     = self.derivative_computer.substitute_parameters(sol_sym, params)

        ic_y0 = self.derivative_computer.substitute_parameters(sp.pi*SYMBOLS.M, params)
        ic_y1 = self.derivative_computer.substitute_parameters(
                    sp.pi * SYMBOLS.beta * sp.diff(f_at, SYMBOLS.alpha), params
                )
        ics = {'y(0)': str(ic_y0), "y'(0)": str(ic_y1)}

        logger.debug(f"[N6] instance created with q={q}, v={v}, a={a}")
        return ode, sol, ics

class NonlinearGeneratorN7Composite(NonlinearGenerator):
    """
    Nonlinear Generator N7: y''(x) + f(y'(x)) + g(y(x)) = RHS
    
    Composite nonlinearity with user-selectable nonlinear functions.
    Allows for highly customizable nonlinear ODEs.
    """
    
    def __init__(self):
        super().__init__("N7")
        self.nonlinear_functions = {
            'power': lambda u, p: u**p,
            'exp': lambda u, p: sp.exp(p*u) - 1,
            'log': lambda u, p: sp.log(sp.Abs(u) + 1)**p,
            'sin': lambda u, p: sp.sin(p*u),
            'tanh': lambda u, p: sp.tanh(p*u)
        }
        
    def _get_required_derivatives(self) -> int:
        return 2
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        try:
            params = self._validate_parameters(params)
            
            # Get nonlinear function choices
            f_type = params.get('f_type', 'power')
            g_type = params.get('g_type', 'tanh')
            f_param = params.get('f_param', 2)
            g_param = params.get('g_param', 1)
            
            # Validate function types
            if f_type not in self.nonlinear_functions:
                f_type = 'power'
            if g_type not in self.nonlinear_functions:
                g_type = 'tanh'
            
            # Get derivatives
            f_at_alpha_beta, f_at_alpha_beta_exp, derivatives = \
                self.derivative_computer.get_derivatives(f_key, self._get_required_derivatives())
            
            if f_at_alpha_beta is None:
                logger.error(f"Failed to compute derivatives for function {f_key}")
                return None, None, None
            
            # Build RHS
            base_term = f_at_alpha_beta - f_at_alpha_beta_exp
            
            # Derivative terms
            second_deriv_term = (
                -SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1] -
                SYMBOLS.beta**2 * sp.exp(-2*SYMBOLS.x) * derivatives[2]
            )
            first_deriv_term = SYMBOLS.beta * sp.exp(-SYMBOLS.x) * derivatives[1]
            
            # Apply selected nonlinear functions
            f_nonlinear = self.nonlinear_functions[f_type]
            g_nonlinear = self.nonlinear_functions[g_type]
            
            f_term = f_nonlinear(sp.pi * first_deriv_term, f_param)
            g_term = g_nonlinear(sp.pi * base_term + SYMBOLS.M, g_param)
            
            rhs_symbolic = sp.pi * second_deriv_term + f_term + g_term
            rhs = self.derivative_computer.substitute_parameters(rhs_symbolic, params)
            
            # Create ODE
            y = SYMBOLS.y
            x = SYMBOLS.x
            
            f_lhs = f_nonlinear(y(x).diff(x), f_param)
            g_lhs = g_nonlinear(y(x), g_param)
            
            lhs = y(x).diff(x, 2) + f_lhs + g_lhs
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
            
            logger.debug(f"N7 composite generated with f={f_type}, g={g_type}")
            return ode, solution, ics
            
        except Exception as e:
            logger.error(f"Error in N7 composite generator: {e}")
            return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        """N7-specific nonlinearity metrics"""
        f_type = params.get('f_type', 'power')
        g_type = params.get('g_type', 'tanh')
        
        # Determine metrics based on function types
        is_exp = f_type == 'exp' or g_type == 'exp'
        is_log = f_type == 'log' or g_type == 'log'
        
        return NonlinearityMetrics(
            pow_deriv_max=params.get('f_param', 2) if f_type == 'power' else 1,
            pow_yprime=params.get('g_param', 1) if g_type == 'power' else 1,
            has_pantograph=False,
            is_exponential_nonlinear=is_exp,
            is_logarithmic_nonlinear=is_log,
            total_nonlinear_degree=2  # Composite
        )


# Additional advanced generators for future extensions

class NonlinearGeneratorN8Fractional(NonlinearGenerator):
    """
    Nonlinear Generator N8: D^α y(x) + (y(x))^q = RHS
    
    Fractional derivative nonlinear ODE.
    Models anomalous diffusion and memory effects.
    """
    
    def __init__(self):
        super().__init__("N8")
        
    def _get_required_derivatives(self) -> int:
        return 2  # For approximation
    
    def generate(
        self, 
        f_key: str, 
        params: Dict[str, float]
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, str]]:
        # This is a placeholder for fractional calculus implementation
        # In practice, would use special functions for fractional derivatives
        logger.warning("N8 Fractional generator not fully implemented")
        return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        return NonlinearityMetrics(
            pow_deriv_max=1,
            pow_yprime=params.get('q', 2),
            has_pantograph=False,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=params.get('q', 2)
        )


class NonlinearGeneratorN9Implicit(NonlinearGenerator):
    """
    Nonlinear Generator N9: F(y''(x), y'(x), y(x)) = RHS
    
    Implicit nonlinear ODE with general function F.
    Most general form of nonlinear ODE.
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
        # This is a placeholder for implicit ODE implementation
        logger.warning("N9 Implicit generator not fully implemented")
        return None, None, None
    
    def _compute_nonlinearity_metrics(self, params: Dict[str, float]) -> NonlinearityMetrics:
        return NonlinearityMetrics(
            pow_deriv_max=1,
            pow_yprime=1,
            has_pantograph=False,
            is_exponential_nonlinear=False,
            is_logarithmic_nonlinear=False,
            total_nonlinear_degree=1
        )


# Registry of all nonlinear generators
NONLINEAR_GENERATORS = {
    'N1': NonlinearGeneratorN1,
    'N2': NonlinearGeneratorN2,
    'N3': NonlinearGeneratorN3,
    'N4': NonlinearGeneratorN4,
    'N5': NonlinearGeneratorN5,
    'N6': NonlinearGeneratorN6Pantograph,
    'N7': NonlinearGeneratorN7Composite,
    # 'N8': NonlinearGeneratorN8Fractional,  # Future extension
    # 'N9': NonlinearGeneratorN9Implicit,    # Future extension
}


def get_nonlinear_generator(name: str) -> Optional[NonlinearGenerator]:
    """Factory function to get nonlinear generator by name"""
    generator_class = NONLINEAR_GENERATORS.get(name)
    if generator_class:
        return generator_class()
    else:
        logger.error(f"Unknown nonlinear generator: {name}")
        return None


def list_available_generators() -> List[Dict[str, Any]]:
    """List all available nonlinear generators with descriptions"""
    generators = []
    
    descriptions = {
        'N1': "Power of second derivative: (y'')^q + y = RHS",
        'N2': "Mixed powers: (y'')^q + (y')^v = RHS",
        'N3': "Power of first derivative: y + (y')^v = RHS",
        'N4': "Exponential nonlinearity: exp(y'') + y = RHS",
        'N5': "Trigonometric: sin(y'') + cos(y') + y = RHS",
        'N6': "Nonlinear pantograph: (y'')^q + y(x/a)^v - y = RHS",
        'N7': "Composite: y'' + f(y') + g(y) = RHS",
        'N8': "Fractional: D^α y + y^q = RHS (future)",
        'N9': "Implicit: F(y'', y', y) = RHS (future)"
    }
    
    complexity = {
        'N1': 'Medium',
        'N2': 'High',
        'N3': 'Low',
        'N4': 'Very High',
        'N5': 'High',
        'N6': 'Very High',
        'N7': 'Variable',
        'N8': 'Very High',
        'N9': 'Extreme'
    }
    
    for name, gen_class in NONLINEAR_GENERATORS.items():
        generators.append({
            'name': name,
            'class': gen_class.__name__,
            'description': descriptions.get(name, ''),
            'complexity': complexity.get(name, 'Unknown'),
            'has_pantograph': 'pantograph' in gen_class.__name__.lower(),
            'implemented': name not in ['N8', 'N9']
        })
    
    return generators


def validate_nonlinear_params(params: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate parameters for nonlinear generators
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check power parameters
    if 'q' in params:
        q = params['q']
        if not isinstance(q, (int, float)) or q < 1 or q > 10:
            errors.append(f"Parameter q={q} must be between 1 and 10")
    
    if 'v' in params:
        v = params['v']
        if not isinstance(v, (int, float)) or v < 1 or v > 10:
            errors.append(f"Parameter v={v} must be between 1 and 10")
    
    # Check pantograph parameter
    if 'a' in params:
        a = params['a']
        if not isinstance(a, (int, float)) or a <= 1:
            errors.append(f"Pantograph parameter a={a} must be > 1")
    
    # Check composite parameters
    if 'f_type' in params:
        if params['f_type'] not in ['power', 'exp', 'log', 'sin', 'tanh']:
            errors.append(f"Unknown f_type: {params['f_type']}")
    
    if 'g_type' in params:
        if params['g_type'] not in ['power', 'exp', 'log', 'sin', 'tanh']:
            errors.append(f"Unknown g_type: {params['g_type']}")
    
    return len(errors) == 0, errors