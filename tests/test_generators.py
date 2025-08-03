# tests/test_generators.py
import pytest
import sympy as sp
from core.symbols import SYMBOLS
from generators.linear import LinearGeneratorL1, LinearGeneratorL2
from generators.nonlinear import NonlinearGeneratorN1

class TestLinearGenerators:
    """Test suite for linear generators"""
    
    @pytest.fixture
    def standard_params(self):
        return {
            'alpha': 1.0,
            'beta': 1.0,
            'M': 0.0,
            'q': 2,
            'v': 3,
            'a': 2
        }
    
    @pytest.mark.parametrize("function", ["identity", "exponential", "sine"])
    def test_generator_l1(self, function, standard_params):
        """Test L1 generator with various functions"""
        generator = LinearGeneratorL1()
        ode, solution, ics = generator.generate(function, standard_params)
        
        assert ode is not None
        assert solution is not None
        assert 'y(0)' in ics
        assert "y'(0)" in ics
        
        # Verify it's a second-order linear ODE
        assert SYMBOLS.y(SYMBOLS.x).diff(SYMBOLS.x, 2) in ode.lhs.atoms()
    
    @pytest.mark.slow
    def test_generator_l4_pantograph(self, standard_params):
        """Test L4 pantograph generator"""
        from generators.linear import LinearGeneratorL4Pantograph
        
        generator = LinearGeneratorL4Pantograph()
        ode, solution, ics = generator.generate("identity", standard_params)
        
        assert ode is not None
        # Check for pantograph term y(x/a)
        assert any('/' in str(arg) for arg in ode.atoms() if hasattr(arg, 'args'))

class TestVerification:
    """Test verification system"""
    
    @pytest.fixture
    def verifier(self):
        from verification.verifier import ODEVerifier
        config = {
            'numeric_test_points': [0.1, 0.5, 1.0],
            'residual_tolerance': 1e-8,
            'verification_timeout': 5
        }
        return ODEVerifier(config)
    
    def test_timeout_handling(self, verifier):
        """Test verification timeout"""
        # Create an ODE that will timeout
        y = SYMBOLS.y
        x = SYMBOLS.x
        
        # Very complex expression
        complex_expr = sum(sp.sin(i*x)**i for i in range(100))
        ode = sp.Eq(y(x).diff(x), complex_expr)
        solution = y(x)  # Wrong solution
        
        verified, method, confidence = verifier.verify(ode, solution)
        
        assert not verified
        assert confidence == 0.0

# tests/conftest.py
import pytest
import logging

def pytest_configure(config):
    """Configure pytest"""
    logging.basicConfig(level=logging.DEBUG)

def pytest_collection_modifyitems(config, items):
    """Add markers to tests"""
    for item in items:
        # Mark all tests that take > 1s as slow
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
