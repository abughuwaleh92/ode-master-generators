#!/usr/bin/env python
"""Test script to verify all imports work correctly"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    # Test core imports
    from core.types import ODEInstance
    from core.symbols import SYMBOLS
    from core.functions import AnalyticFunctionLibrary
    print("✓ Core imports successful")
    
    # Test utils imports
    from utils.config import ConfigManager
    from utils.derivatives import DerivativeComputer
    from utils.features import FeatureExtractor
    print("✓ Utils imports successful")
    
    # Test generators imports
    from generators.base import BaseGenerator
    from generators.linear import LinearGeneratorL1
    from generators.nonlinear import NonlinearGeneratorN1
    print("✓ Generators imports successful")
    
    # Test verification imports
    from verification.verifier import ODEVerifier
    print("✓ Verification imports successful")
    
    # Test pipeline imports
    from pipeline.generator import ODEDatasetGenerator
    from pipeline.parallel import ParallelODEGenerator
    print("✓ Pipeline imports successful")
    
    print("\n✅ All imports successful!")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    config = ConfigManager()
    print(f"✓ ConfigManager initialized")
    print(f"✓ Default samples per combo: {config.get('generation.samples_per_combo')}")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print(f"Make sure you're running from the project root directory")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)