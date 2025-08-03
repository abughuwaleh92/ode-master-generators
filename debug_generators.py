# debug_generators.py
#!/usr/bin/env python
"""Debug script for failed generators"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from generators.linear import LinearGeneratorL4Pantograph
from generators.nonlinear import NonlinearGeneratorN4, NonlinearGeneratorN5, NonlinearGeneratorN6Pantograph
from verification.verifier import ODEVerifier
import sympy as sp
from core.symbols import SYMBOLS

def debug_generator(generator, gen_name, function='identity'):
    """Debug a specific generator"""
    print(f"\n{'='*60}")
    print(f"Debugging {gen_name}")
    print('='*60)
    
    # Simple parameters
    params = {
        'alpha': 1.0,
        'beta': 1.0,
        'M': 0.0,
        'q': 2,
        'v': 3,
        'a': 2
    }
    
    print(f"Parameters: {params}")
    
    try:
        # Generate ODE
        print(f"\nGenerating ODE...")
        ode, solution, ics = generator.generate(function, params)
        
        if ode is None:
            print("Generation failed!")
            return
        
        print(f"ODE: {ode}")
        print(f"Solution: {solution}")
        print(f"Initial conditions: {ics}")
        
        # Test verification
        print("\nTesting verification...")
        verifier_config = {
            'numeric_test_points': [0.1, 0.5, 1.0],
            'residual_tolerance': 1e-6  # More tolerant
        }
        verifier = ODEVerifier(verifier_config)
        
        # Try substitution manually
        print("\nManual substitution test:")
        y = SYMBOLS.y
        x = SYMBOLS.x
        
        # Build substitutions
        subs = {y(x): solution}
        
        # Get max derivative order
        max_order = 0
        for atom in ode.atoms(sp.Derivative):
            if hasattr(atom, 'args') and len(atom.args) > 1:
                for arg in atom.args[1:]:
                    if isinstance(arg, tuple) and len(arg) == 2:
                        max_order = max(max_order, arg[1])
        
        print(f"Max derivative order: {max_order}")
        
        # Add derivatives
        for order in range(1, max_order + 1):
            deriv = sp.diff(solution, x, order)
            subs[y(x).diff(x, order)] = deriv
            print(f"y{'^' + str(order)}(x) = {deriv}")
        
        # Handle pantograph terms if present
        if hasattr(generator, 'name') and 'Pantograph' in generator.name:
            a_val = params.get('a', 2)
            # Look for y(x/a) terms
            for atom in ode.atoms():
                if isinstance(atom, sp.Function) and atom.func == y:
                    arg = atom.args[0] if atom.args else None
                    if arg and arg != x:
                        print(f"Found pantograph term: {atom}")
                        subs[atom] = solution.subs(x, arg)
        
        print(f"\nSubstitutions: {subs}")
        
        # Substitute
        lhs_sub = ode.lhs.subs(subs)
        rhs_sub = ode.rhs.subs(subs)
        
        print(f"\nLHS after substitution: {lhs_sub}")
        print(f"RHS after substitution: {rhs_sub}")
        
        # Try different simplification methods
        print("\nTrying simplification methods:")
        
        residual = lhs_sub - rhs_sub
        print(f"Raw residual: {residual}")
        
        # Method 1: Direct simplify
        try:
            simplified = sp.simplify(residual)
            print(f"After simplify: {simplified}")
            if simplified == 0:
                print("SUCCESS with simplify!")
        except Exception as e:
            print(f"Simplify failed: {e}")
        
        # Method 2: Expand then simplify
        try:
            expanded = sp.expand(residual)
            print(f"After expand: {expanded}")
            simplified_expanded = sp.simplify(expanded)
            print(f"After expand+simplify: {simplified_expanded}")
            if simplified_expanded == 0:
                print("SUCCESS with expand+simplify!")
        except Exception as e:
            print(f"Expand failed: {e}")
        
        # Method 3: Numerical check
        try:
            print("\nNumerical verification:")
            test_points = [0.1, 0.5, 1.0]
            max_residual = 0
            
            for test_x in test_points:
                residual_val = residual.subs(x, test_x).evalf()
                print(f"Residual at x={test_x}: {residual_val}")
                max_residual = max(max_residual, abs(complex(residual_val)))
            
            print(f"Max residual: {max_residual}")
            if max_residual < 1e-6:
                print("SUCCESS with numerical verification!")
        except Exception as e:
            print(f"Numerical check failed: {e}")
        
        # Try the actual verifier
        print("\nUsing ODEVerifier:")
        verified, method, confidence = verifier.verify(ode, solution)
        print(f"Verified: {verified}")
        print(f"Method: {method}")
        print(f"Confidence: {confidence}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Debug each failed generator
if __name__ == "__main__":
    generators_to_debug = [
        (LinearGeneratorL4Pantograph(), "L4 Pantograph"),
        (NonlinearGeneratorN4(), "N4 Exponential"),
        (NonlinearGeneratorN5(), "N5 Trigonometric"),
        (NonlinearGeneratorN6Pantograph(), "N6 Nonlinear Pantograph")
    ]
    
    for generator, name in generators_to_debug:
        debug_generator(generator, name)