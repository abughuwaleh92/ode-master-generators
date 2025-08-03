#!/usr/bin/env python
"""Test script to debug ODE generation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from generators.linear import LinearGeneratorL1
from verification.verifier import ODEVerifier
import sympy as sp

def test_simple_generation():
    """Test a simple ODE generation"""
    print("Testing simple ODE generation...\n")
    
    # Create generator
    generator = LinearGeneratorL1()
    
    # Simple parameters
    params = {
        'alpha': 1.0,
        'beta': 1.0,
        'M': 0.0,
        'q': 2,
        'v': 3,
        'a': 2
    }
    
    # Generate ODE
    print("Generating ODE...")
    ode, solution, ics = generator.generate('identity', params)
    
    if ode is None:
        print("Generation failed!")
        return
    
    print(f"ODE type: {type(ode)}")
    print(f"ODE: {ode}")
    print(f"\nSolution type: {type(solution)}")
    print(f"Solution: {solution}")
    print(f"\nInitial conditions: {ics}")
    
    # Test verification
    print("\nTesting verification...")
    verifier_config = {
        'numeric_test_points': [0.1, 0.5, 1.0],
        'residual_tolerance': 1e-8
    }
    verifier = ODEVerifier(verifier_config)
    
    # Verify
    verified, method, confidence = verifier.verify(ode, solution)
    
    print(f"\nVerified: {verified}")
    print(f"Method: {method}")
    print(f"Confidence: {confidence}")
    
    # Test substitution manually
    print("\nManual substitution test:")
    from core.symbols import SYMBOLS
    
    y = SYMBOLS.y
    x = SYMBOLS.x
    
    # Substitute solution into ODE
    subs = {y(x): solution}
    subs[y(x).diff(x)] = sp.diff(solution, x)
    subs[y(x).diff(x, 2)] = sp.diff(solution, x, 2)
    
    lhs_sub = ode.lhs.subs(subs)
    rhs_sub = ode.rhs.subs(subs)
    
    print(f"LHS after substitution: {lhs_sub}")
    print(f"RHS after substitution: {rhs_sub}")
    
    residual = sp.simplify(lhs_sub - rhs_sub)
    print(f"Residual: {residual}")
    
    if residual == 0:
        print("Manual verification: SUCCESS")
    else:
        print("Manual verification: FAILED")

if __name__ == "__main__":
    test_simple_generation()