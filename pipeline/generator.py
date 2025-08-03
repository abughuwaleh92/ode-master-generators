"""
ODE Dataset Generator Pipeline

This module handles the complete pipeline for generating ODE datasets,
including parallel processing, streaming, and checkpointing.
"""

import random
import time
import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import sympy as sp
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from core.types import (
    ODEInstance, GeneratorResult, GeneratorType, 
    VerificationMethod, DatasetStatistics
)
from core.functions import AnalyticFunctionLibrary
from core.symbols import SYMBOLS
from generators.linear import LINEAR_GENERATORS
from generators.nonlinear import NONLINEAR_GENERATORS
from verification.verifier import ODEVerifier
from utils.config import ConfigManager

logger = logging.getLogger(__name__)


class ODEDatasetGenerator:
    """
    Main dataset generator with streaming support and enhanced verification
    """
    
    def __init__(self, config: ConfigManager = None, seed: int = 42):
        """
        Initialize the ODE dataset generator
        
        Args:
            config: ConfigManager instance (if None, creates default)
            seed: Random seed for reproducibility
        """
        self.config = config if config is not None else ConfigManager()
        self.seed = seed
        self._set_random_seeds()
        
        # Initialize components
        self.f_library = AnalyticFunctionLibrary.get_safe_library()
        
        # Create verifier with config
        verifier_config = self.config.config.get('verification', {})
        self.verifier = ODEVerifier(verifier_config)
        
        # Initialize generators
        self._initialize_generators()
        
        # Dataset storage
        self.dataset: List[ODEInstance] = []
        self.failed_generations: List[Dict] = []
        self.statistics = {
            'start_time': None,
            'end_time': None,
            'generator_stats': {},
            'function_stats': {},
            'verification_stats': Counter(),
            'generation_times': []
        }
        
        # Streaming file handle
        self._stream_file = None
        
        # Cache for working generators
        self._working_generators = None
        
    def _set_random_seeds(self):
        """Set all random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        logger.debug(f"Random seeds set to {self.seed}")
        
    def _initialize_generators(self):
        """Initialize all available generators"""
        self.linear_generators = {}
        for name, gen_class in LINEAR_GENERATORS.items():
            try:
                self.linear_generators[name] = gen_class()
                logger.debug(f"Initialized linear generator {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize linear generator {name}: {e}")
        
        self.nonlinear_generators = {}
        for name, gen_class in NONLINEAR_GENERATORS.items():
            try:
                self.nonlinear_generators[name] = gen_class()
                logger.debug(f"Initialized nonlinear generator {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize nonlinear generator {name}: {e}")
    
    def sample_parameters(self) -> Dict[str, float]:
        """Sample parameters from configured ranges"""
        param_ranges = self.config.get('generation.parameter_ranges', {})
        
        # Default ranges if not in config
        default_ranges = {
            'alpha': [0, 0.5, 1, 1.5, 2],
            'beta': [0.5, 1, 1.5, 2],
            'M': [0, 0.5, 1],
            'q': [2, 3],
            'v': [2, 3, 4],
            'a': [2, 3, 4]
        }
        
        sampled = {}
        for param, default_values in default_ranges.items():
            values = param_ranges.get(param, default_values)
            if isinstance(values, list) and values:
                sampled[param] = random.choice(values)
            else:
                sampled[param] = values if isinstance(values, (int, float)) else default_values[0]
        
        return sampled
    
    def _enhanced_verify(self, ode: sp.Expr, solution: sp.Expr) -> Tuple[bool, VerificationMethod, float]:
        """
        Enhanced verification that handles pantograph and derivative substitutions properly
        """
        try:
            y = SYMBOLS.y
            x = SYMBOLS.x
            
            # Build complete substitution dictionary
            subs = {y(x): solution}
            
            # Get maximum derivative order
            max_order = 0
            for atom in ode.atoms(sp.Derivative):
                if isinstance(atom, sp.Derivative):
                    # Get the derivative order
                    if len(atom.args) > 1:
                        var_count = atom.args[1:]
                        for var_tuple in var_count:
                            if isinstance(var_tuple, tuple) and len(var_tuple) == 2:
                                max_order = max(max_order, var_tuple[1])
                            else:
                                max_order = max(max_order, 1)
            
            logger.debug(f"Maximum derivative order: {max_order}")
            
            # Add all derivatives to substitution
            for order in range(1, max_order + 1):
                deriv = sp.diff(solution, x, order)
                subs[y(x).diff(x, order)] = deriv
                logger.debug(f"Added derivative order {order}: {deriv}")
            
            # Handle pantograph terms - look for y(expr) where expr != x
            pantograph_terms = []
            for atom in ode.atoms(sp.Function):
                if isinstance(atom, sp.Function) and atom.func == y:
                    if atom.args and atom.args[0] != x:
                        pantograph_terms.append(atom)
            
            # Substitute pantograph terms
            for pterm in pantograph_terms:
                arg = pterm.args[0]
                # Substitute x with the argument in the solution
                subs[pterm] = solution.subs(x, arg)
                logger.debug(f"Added pantograph substitution: {pterm} -> {solution.subs(x, arg)}")
            
            # Apply substitutions
            lhs_sub = ode.lhs.subs(subs)
            rhs_sub = ode.rhs.subs(subs)
            residual = lhs_sub - rhs_sub
            
            logger.debug(f"LHS after substitution: {lhs_sub}")
            logger.debug(f"RHS after substitution: {rhs_sub}")
            logger.debug(f"Residual: {residual}")
            
            # Try to simplify the residual
            try:
                # First expand derivatives
                residual = residual.doit()
                
                # Then try different simplification strategies
                simplified = sp.simplify(residual)
                if simplified == 0 or abs(complex(simplified)) < 1e-10:
                    return True, VerificationMethod.SUBSTITUTION, 1.0
                
                # Try expand and simplify
                expanded = sp.expand(residual)
                simplified_expanded = sp.simplify(expanded)
                if simplified_expanded == 0 or abs(complex(simplified_expanded)) < 1e-10:
                    return True, VerificationMethod.SUBSTITUTION, 0.95
                
                # Try trigsimp for trigonometric expressions
                if residual.has(sp.sin) or residual.has(sp.cos):
                    trig_simplified = sp.trigsimp(residual)
                    if trig_simplified == 0 or abs(complex(trig_simplified)) < 1e-10:
                        return True, VerificationMethod.SUBSTITUTION, 0.9
                
            except Exception as e:
                logger.debug(f"Simplification failed: {e}")
            
            # If symbolic verification fails, try numeric
            return self.verifier._verify_numerically(ode, solution)
            
        except Exception as e:
            logger.error(f"Enhanced verification failed: {e}")
            return False, VerificationMethod.FAILED, 0.0
    
    def test_generators(self) -> Dict[str, Dict[str, Any]]:
        """Test all generators and identify working ones"""
        if self._working_generators is not None:
            return self._working_generators
            
        logger.info("Testing generator reliability...")
        
        working_generators = {
            'linear': {},
            'nonlinear': {}
        }
        
        # Test linear generators
        for name, generator in self.linear_generators.items():
            try:
                params = self.sample_parameters()
                logger.debug(f"Testing {name} with params: {params}")
                
                # Generate ODE directly
                ode, solution, ics = generator.generate('identity', params)
                
                if ode is not None and solution is not None:
                    # Use enhanced verification
                    verified, method, confidence = self._enhanced_verify(ode, solution)
                    
                    if verified:
                        working_generators['linear'][name] = generator
                        logger.info(f"✓ LINEAR {name}: Operational (method: {method.value}, confidence: {confidence:.2f})")
                    else:
                        logger.warning(f"✗ LINEAR {name}: Verification failed")
                        # Additional debug info
                        logger.debug(f"  ODE: {ode}")
                        logger.debug(f"  Solution: {solution}")
                else:
                    logger.warning(f"✗ LINEAR {name}: Generation failed")
                    
            except Exception as e:
                logger.error(f"✗ LINEAR {name}: Error - {str(e)[:100]}...")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Test nonlinear generators
        for name, generator in self.nonlinear_generators.items():
            try:
                params = self.sample_parameters()
                logger.debug(f"Testing {name} with params: {params}")
                
                # Generate ODE directly
                ode, solution, ics = generator.generate('identity', params)
                
                if ode is not None and solution is not None:
                    # Use enhanced verification
                    verified, method, confidence = self._enhanced_verify(ode, solution)
                    
                    if verified:
                        working_generators['nonlinear'][name] = generator
                        logger.info(f"✓ NONLINEAR {name}: Operational (method: {method.value}, confidence: {confidence:.2f})")
                    else:
                        logger.warning(f"✗ NONLINEAR {name}: Verification failed")
                        # Additional debug info
                        logger.debug(f"  ODE: {ode}")
                        logger.debug(f"  Solution: {solution}")
                else:
                    logger.warning(f"✗ NONLINEAR {name}: Generation failed")
                    
            except Exception as e:
                logger.error(f"✗ NONLINEAR {name}: Error - {str(e)[:100]}...")
                import traceback
                logger.debug(traceback.format_exc())
        
        self._working_generators = working_generators
        return working_generators
    
    def generate_and_verify_single(
        self,
        generator: Any,
        gen_type: str,
        gen_name: str,
        f_key: str,
        params: Dict[str, float]
    ) -> Tuple[Optional[sp.Expr], Optional[sp.Expr], Optional[Dict], bool, Any, float]:
        """Generate and verify a single ODE, returning all components"""
        try:
            # Generate ODE
            ode, solution, ics = generator.generate(f_key, params)
            
            if ode is None or solution is None:
                logger.debug(f"Generation returned None for {gen_name} + {f_key}")
                return None, None, None, False, None, 0.0
            
            # Use enhanced verification
            verified, method, confidence = self._enhanced_verify(ode, solution)
            
            return ode, solution, ics, verified, method, confidence
            
        except Exception as e:
            logger.debug(f"Generation/verification error for {gen_name} + {f_key}: {e}")
            return None, None, None, False, None, 0.0
    
    def generate_single_ode(
        self, 
        generator: Any,
        gen_type: str,
        gen_name: str,
        f_key: str,
        ode_id: int
    ) -> Optional[ODEInstance]:
        """Generate and verify a single ODE"""
        start_time = time.time()
        
        logger.debug(f"[ODE #{ode_id}] Generating: {gen_name} + {f_key}")
        
        try:
            # Sample parameters
            params = self.sample_parameters()
            
            # Generate and verify
            ode, solution, ics, verified, method, confidence = self.generate_and_verify_single(
                generator, gen_type, gen_name, f_key, params
            )
            
            if ode is None or solution is None:
                self.failed_generations.append({
                    'generator': f"{gen_type}_{gen_name}",
                    'function': f_key,
                    'error': "Generation failed",
                    'params': params
                })
                return None
            
            # Analyze ODE properties
            properties = self.verifier.analyze_ode_properties(ode)
            
            # Create lambdified versions for numeric evaluation
            x = SYMBOLS.x
            
            try:
                # For the ODE, we need the residual function
                # Build substitutions for lambdification
                y = SYMBOLS.y
                
                # Create a residual expression
                residual_expr = ode.lhs - ode.rhs
                
                # Replace y(x) and derivatives with functions of x
                # This is a simplified approach - in practice, you'd solve the ODE numerically
                ode_numeric = None  # We'll leave this as None for now
                solution_numeric = sp.lambdify(x, solution, 'numpy')
                
            except Exception as e:
                logger.debug(f"Lambdification failed: {e}")
                ode_numeric = None
                solution_numeric = None
            
            # Initialize nonlinearity metrics
            nonlinearity_metrics = None
            if gen_type.lower() == 'nonlinear' and hasattr(generator, '_compute_nonlinearity_metrics'):
                try:
                    nonlinearity_metrics = generator._compute_nonlinearity_metrics(params)
                except Exception as e:
                    logger.debug(f"Failed to compute nonlinearity metrics: {e}")
            
            # Create ODE instance
            generation_time = time.time() - start_time
            
            ode_instance = ODEInstance(
                id=ode_id,
                generator_type=GeneratorType(gen_type.lower()),
                generator_name=gen_name,
                function_name=f_key,
                ode_symbolic=str(ode),
                ode_latex=sp.latex(ode),
                ode_numeric=ode_numeric,
                solution_symbolic=str(solution),
                solution_latex=sp.latex(solution),
                solution_numeric=solution_numeric,
                initial_conditions=ics,
                parameters=params,
                complexity_score=properties.get('complexity_score', 0),
                operation_count=properties.get('operation_count', 0),
                atom_count=properties.get('atom_count', 0),
                symbol_count=properties.get('symbol_count', 0),
                has_pantograph=self._check_pantograph(ode),
                verified=verified,
                verification_method=method if verified else VerificationMethod.FAILED,
                verification_confidence=confidence,
                generation_time=generation_time,
                nonlinearity_metrics=nonlinearity_metrics
            )
            
            # Update statistics
            self._update_statistics(ode_instance, gen_type, gen_name, f_key)
            
            logger.debug(f"[ODE #{ode_id}] Success: verified={verified}, time={generation_time:.3f}s")
            
            return ode_instance
            
        except Exception as e:
            logger.error(f"Error generating ODE #{ode_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            self.failed_generations.append({
                'generator': f"{gen_type}_{gen_name}",
                'function': f_key,
                'error': str(e)
            })
            return None
    
    def _check_pantograph(self, ode: sp.Expr) -> bool:
        """Check if ODE contains pantograph terms"""
        y = SYMBOLS.y
        x = SYMBOLS.x
        
        # Look for y(f(x)) where f(x) != x
        for atom in ode.atoms(sp.Function):
            if isinstance(atom, sp.Function) and atom.func == y:
                if atom.args and atom.args[0] != x:
                    return True
        
        return False
    
    def _update_statistics(
        self, 
        ode: ODEInstance,
        gen_type: str,
        gen_name: str,
        f_key: str
    ):
        """Update generation statistics"""
        # Generator statistics
        gen_key = f"{gen_type}_{gen_name}"
        if gen_key not in self.statistics['generator_stats']:
            self.statistics['generator_stats'][gen_key] = {
                'total': 0,
                'successful': 0,
                'verified': 0
            }
        
        stats = self.statistics['generator_stats'][gen_key]
        stats['total'] += 1
        stats['successful'] += 1
        if ode.verified:
            stats['verified'] += 1
        
        # Function statistics
        if f_key not in self.statistics['function_stats']:
            self.statistics['function_stats'][f_key] = {
                'total': 0,
                'verified': 0
            }
        
        self.statistics['function_stats'][f_key]['total'] += 1
        if ode.verified:
            self.statistics['function_stats'][f_key]['verified'] += 1
        
        # Verification statistics
        self.statistics['verification_stats'][ode.verification_method.value] += 1
        
        # Generation time
        self.statistics['generation_times'].append(ode.generation_time)
    
    def _open_stream_file(self):
        """Open streaming file for writing with UTF-8 encoding"""
        if self.config.get('performance.streaming_enabled', True):
            filepath = self.config.get('output.streaming_file', 'ode_dataset.jsonl')
            self._stream_file = open(filepath, 'w', encoding='utf-8')
            logger.info(f"Streaming to {filepath}")
    
    def _close_stream_file(self):
        """Close streaming file"""
        if self._stream_file:
            self._stream_file.close()
            self._stream_file = None
            logger.debug("Closed streaming file")
    
    def _write_to_stream(self, ode: ODEInstance):
        """Write ODE to streaming file"""
        if self._stream_file:
            try:
                json_line = json.dumps(ode.to_dict(), default=str)
                self._stream_file.write(json_line + '\n')
                self._stream_file.flush()
            except Exception as e:
                logger.error(f"Failed to write to stream: {e}")
    
    def _save_checkpoint(self, checkpoint_id: int):
        """Save compressed checkpoint of current dataset"""
        if self.config.get('output.save_intermediate', True):
            try:
                import gzip
                
                checkpoint_path = f"checkpoint_{checkpoint_id:06d}.jsonl.gz"
                
                with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as f:
                    for ode in self.dataset:
                        f.write(json.dumps(ode.to_dict(), default=str) + '\n')
                
                logger.info(f"Checkpoint saved: {checkpoint_path} ({len(self.dataset)} ODEs)")
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
    
    def generate_dataset(
        self, 
        samples_per_combo: Optional[int] = None
    ) -> List[ODEInstance]:
        """Generate complete ODE dataset"""
        if samples_per_combo is None:
            samples_per_combo = self.config.get('generation.samples_per_combo', 5)
        
        logger.info("="*60)
        logger.info("Starting ODE Dataset Generation")
        logger.info("="*60)
        logger.info(f"Samples per combination: {samples_per_combo}")
        logger.info(f"Random seed: {self.seed}")
        
        # Test generators
        working_generators = self.test_generators()
        
        total_working = (
            len(working_generators['linear']) + 
            len(working_generators['nonlinear'])
        )
        
        if total_working == 0:
            logger.error("No working generators found!")
            return []
        
        logger.info(f"Working generators: {total_working}")
        logger.info(f"Functions in library: {len(self.f_library)}")
        
        # Calculate total expected ODEs
        total_expected = total_working * len(self.f_library) * samples_per_combo
        logger.info(f"Expected total ODEs: {total_expected}")
        
        # Open streaming file
        self._open_stream_file()
        
        # Start generation
        self.statistics['start_time'] = datetime.now()
        ode_id = 0
        checkpoint_interval = self.config.get('output.checkpoint_interval', 500)
        
        # Progress tracking
        progress_interval = 10
        last_log_time = time.time()
        generation_start = time.time()
        
        try:
            # Generate from linear generators
            for gen_name, generator in working_generators['linear'].items():
                logger.info(f"\nProcessing LINEAR generator: {gen_name}")
                gen_start_time = time.time()
                gen_count = 0
                
                for f_idx, f_key in enumerate(self.f_library.keys()):
                    for sample in range(samples_per_combo):
                        ode = self.generate_single_ode(
                            generator, 'linear', gen_name, f_key, ode_id
                        )
                        
                        if ode:
                            self.dataset.append(ode)
                            self._write_to_stream(ode)
                            ode_id += 1
                            gen_count += 1
                            
                            # Checkpoint
                            if ode_id % checkpoint_interval == 0:
                                self._save_checkpoint(ode_id)
                        
                        # Progress updates
                        if ode_id > 0 and ode_id % progress_interval == 0:
                            current_time = time.time()
                            elapsed = current_time - generation_start
                            rate = ode_id / elapsed
                            eta = (total_expected - ode_id) / rate if rate > 0 else 0
                            
                            logger.info(
                                f"Progress: {ode_id}/{total_expected} "
                                f"({100*ode_id/total_expected:.1f}%) | "
                                f"Rate: {rate:.1f} ODEs/s | "
                                f"ETA: {eta/60:.1f} min"
                            )
                        
                        # Detailed progress every 50 ODEs
                        if ode_id % 50 == 0:
                            self._log_progress(ode_id, total_expected)
                
                gen_elapsed = time.time() - gen_start_time
                logger.info(
                    f"Completed {gen_name}: {gen_count} ODEs in {gen_elapsed:.1f}s "
                    f"({gen_count/gen_elapsed:.1f} ODEs/s)"
                )
            
            # Generate from nonlinear generators
            for gen_name, generator in working_generators['nonlinear'].items():
                logger.info(f"\nProcessing NONLINEAR generator: {gen_name}")
                gen_start_time = time.time()
                gen_count = 0
                
                for f_idx, f_key in enumerate(self.f_library.keys()):
                    for sample in range(samples_per_combo):
                        ode = self.generate_single_ode(
                            generator, 'nonlinear', gen_name, f_key, ode_id
                        )
                        
                        if ode:
                            self.dataset.append(ode)
                            self._write_to_stream(ode)
                            ode_id += 1
                            gen_count += 1
                            
                            # Checkpoint
                            if ode_id % checkpoint_interval == 0:
                                self._save_checkpoint(ode_id)
                        
                        # Progress updates
                        if ode_id > 0 and ode_id % progress_interval == 0:
                            current_time = time.time()
                            elapsed = current_time - generation_start
                            rate = ode_id / elapsed
                            eta = (total_expected - ode_id) / rate if rate > 0 else 0
                            
                            logger.info(
                                f"Progress: {ode_id}/{total_expected} "
                                f"({100*ode_id/total_expected:.1f}%) | "
                                f"Rate: {rate:.1f} ODEs/s | "
                                f"ETA: {eta/60:.1f} min"
                            )
                        
                        # Detailed progress every 50 ODEs
                        if ode_id % 50 == 0:
                            self._log_progress(ode_id, total_expected)
                
                gen_elapsed = time.time() - gen_start_time
                logger.info(
                    f"Completed {gen_name}: {gen_count} ODEs in {gen_elapsed:.1f}s "
                    f"({gen_count/gen_elapsed:.1f} ODEs/s)"
                )
            
        except KeyboardInterrupt:
            logger.warning("\nGeneration interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error during generation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        finally:
            self._close_stream_file()
            self.statistics['end_time'] = datetime.now()
        
        # Final statistics
        self._print_final_statistics()
        
        # Save final checkpoint
        if self.dataset:
            self._save_checkpoint(len(self.dataset))
        
        return self.dataset
    
    def _log_progress(self, current: int, total: int):
        """Log detailed generation progress"""
        if current == 0:
            return
            
        success_rate = 100 * len(self.dataset) / current
        elapsed = (datetime.now() - self.statistics['start_time']).total_seconds()
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        
        # Get recent verification stats
        recent_verified = sum(1 for ode in self.dataset[-50:] if ode.verified) if len(self.dataset) >= 50 else sum(1 for ode in self.dataset if ode.verified)
        recent_total = min(50, len(self.dataset))
        recent_rate = 100 * recent_verified / recent_total if recent_total > 0 else 0
        
        logger.info(
            f"[{current:4d}/{total:4d}] "
            f"Success: {success_rate:.1f}% | "
            f"Recent verification: {recent_rate:.1f}% | "
            f"Rate: {rate:.1f} ODEs/s | "
            f"ETA: {eta:.0f}s"
        )
    
    def _print_final_statistics(self):
        """Print comprehensive final generation statistics"""
        if not self.statistics['start_time'] or not self.statistics['end_time']:
            return
            
        total_time = (
            self.statistics['end_time'] - self.statistics['start_time']
        ).total_seconds()
        
        total_generated = len(self.dataset)
        total_attempted = sum(
            stats['total'] 
            for stats in self.statistics['generator_stats'].values()
        )
        
        verification_rate = 100 * sum(
            1 for ode in self.dataset if ode.verified
        ) / total_generated if total_generated > 0 else 0
        
        logger.info("\n" + "="*60)
        logger.info("Generation Complete!")
        logger.info("="*60)
        
        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"  ODEs generated: {total_generated}")
        logger.info(f"  ODEs attempted: {total_attempted}")
        logger.info(f"  Success rate: {100*total_generated/total_attempted:.1f}%" if total_attempted > 0 else "N/A")
        logger.info(f"  Overall rate: {total_generated/total_time:.1f} ODEs/second")
        logger.info(f"  Verification rate: {verification_rate:.1f}%")
        logger.info(f"  Failed generations: {len(self.failed_generations)}")
        
        if self.statistics['generation_times']:
            avg_time = np.mean(self.statistics['generation_times'])
            logger.info(f"  Average generation time: {avg_time*1000:.1f}ms per ODE")
        
        # Generator statistics
        logger.info("\nGenerator Performance:")
        for gen_key, stats in sorted(self.statistics['generator_stats'].items()):
            success_rate = 100 * stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            verification_rate = 100 * stats['verified'] / stats['successful'] if stats['successful'] > 0 else 0
            
            logger.info(
                f"  {gen_key:15s}: "
                f"{stats['successful']:4d}/{stats['total']:4d} generated ({success_rate:5.1f}%), "
                f"{stats['verified']:4d} verified ({verification_rate:5.1f}%)"
            )
        
        # Verification method distribution
        logger.info("\nVerification Methods Used:")
        total_verifications = sum(self.statistics['verification_stats'].values())
        for method, count in sorted(self.statistics['verification_stats'].items()):
            percentage = 100 * count / total_verifications if total_verifications > 0 else 0
            logger.info(f"  {method:15s}: {count:5d} ({percentage:5.1f}%)")
        
        # Top functions by verification rate
        logger.info("\nTop Functions by Verification Rate:")
        func_rates = []
        for func, stats in self.statistics['function_stats'].items():
            if stats['total'] > 0:
                rate = 100 * stats['verified'] / stats['total']
                func_rates.append((func, rate, stats['total']))
        
        func_rates.sort(key=lambda x: x[1], reverse=True)
        for func, rate, total in func_rates[:5]:
            logger.info(f"  {func:20s}: {rate:5.1f}% ({total} ODEs)")
        
        # Save detailed statistics
        self.save_report()
    
    def save_report(self, filepath: Optional[str] = None):
        """Save comprehensive generation report"""
        if filepath is None:
            filepath = self.config.get('output.report_file', 'generation_report.json')
        
        # Compute dataset statistics
        stats = DatasetStatistics.from_dataset(self.dataset)
        
        report = {
            'generation_info': {
                'start_time': str(self.statistics['start_time']),
                'end_time': str(self.statistics['end_time']),
                'duration_seconds': (
                    self.statistics['end_time'] - self.statistics['start_time']
                ).total_seconds() if self.statistics['start_time'] and self.statistics['end_time'] else 0,
                'total_generated': len(self.dataset),
                'total_failed': len(self.failed_generations),
                'seed': self.seed,
                'config': self.config.config
            },
            'dataset_statistics': stats.to_dict(),
            'generator_performance': self.statistics['generator_stats'],
            'function_performance': self.statistics['function_stats'],
            'verification_methods': dict(self.statistics['verification_stats']),
            'timing_analysis': {
                'generation_times': {
                    'mean': np.mean(self.statistics['generation_times']),
                    'std': np.std(self.statistics['generation_times']),
                    'min': np.min(self.statistics['generation_times']),
                    'max': np.max(self.statistics['generation_times']),
                    'percentiles': {
                        '25': np.percentile(self.statistics['generation_times'], 25),
                        '50': np.percentile(self.statistics['generation_times'], 50),
                        '75': np.percentile(self.statistics['generation_times'], 75),
                        '95': np.percentile(self.statistics['generation_times'], 95)
                    }
                } if self.statistics['generation_times'] else {}
            },
            'failed_generations': {
                'total': len(self.failed_generations),
                'by_generator': Counter(f['generator'] for f in self.failed_generations),
                'by_function': Counter(f['function'] for f in self.failed_generations),
                'sample_errors': self.failed_generations[:20]  # First 20 errors
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"\nDetailed report saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Resume generation from a checkpoint file"""
        try:
            import gzip
            
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            
            # Clear current dataset
            self.dataset = []
            
            # Load checkpoint
            open_func = gzip.open if checkpoint_path.endswith('.gz') else open
            
            with open_func(checkpoint_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        ode_dict = json.loads(line)
                        ode = ODEInstance.from_dict(ode_dict)
                        self.dataset.append(ode)
            
            logger.info(f"Loaded {len(self.dataset)} ODEs from checkpoint")
            
            # Rebuild statistics
            self._rebuild_statistics_from_dataset()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return False
    
    def _rebuild_statistics_from_dataset(self):
        """Rebuild statistics from loaded dataset"""
        self.statistics = {
            'start_time': datetime.now(),
            'end_time': None,
            'generator_stats': {},
            'function_stats': {},
            'verification_stats': Counter(),
            'generation_times': []
        }
        
        for ode in self.dataset:
            gen_key = f"{ode.generator_type.value}_{ode.generator_name}"
            
            # Generator stats
            if gen_key not in self.statistics['generator_stats']:
                self.statistics['generator_stats'][gen_key] = {
                    'total': 0,
                    'successful': 0,
                    'verified': 0
                }
            
            stats = self.statistics['generator_stats'][gen_key]
            stats['total'] += 1
            stats['successful'] += 1
            if ode.verified:
                stats['verified'] += 1
            
            # Function stats
            if ode.function_name not in self.statistics['function_stats']:
                self.statistics['function_stats'][ode.function_name] = {
                    'total': 0,
                    'verified': 0
                }
            
            self.statistics['function_stats'][ode.function_name]['total'] += 1
            if ode.verified:
                self.statistics['function_stats'][ode.function_name]['verified'] += 1
            
            # Verification stats
            self.statistics['verification_stats'][ode.verification_method.value] += 1
            
            # Generation times
            self.statistics['generation_times'].append(ode.generation_time)


# Utility function for parallel generation
def generate_ode_batch(args):
    """Generate a batch of ODEs (for parallel processing)"""
    generator_class, gen_type, gen_name, function_keys, params_list, start_id = args
    
    try:
        # Create generator instance
        generator = generator_class()
        
        # Create verifier
        from verification.verifier import ODEVerifier
        verifier = ODEVerifier({
            'numeric_test_points': [0.1, 0.5, 1.0],
            'residual_tolerance': 1e-8
        })
        
        results = []
        
        for i, (f_key, params) in enumerate(zip(function_keys, params_list)):
            ode_id = start_id + i
            
            try:
                # Generate ODE
                ode, solution, ics = generator.generate(f_key, params)
                
                if ode is None or solution is None:
                    continue
                
                # Verify
                verified, method, confidence = verifier.verify(ode, solution)
                
                # Create ODE instance
                properties = verifier.analyze_ode_properties(ode)
                
                ode_instance = ODEInstance(
                    id=ode_id,
                    generator_type=GeneratorType(gen_type.lower()),
                    generator_name=gen_name,
                    function_name=f_key,
                    ode_symbolic=str(ode),
                    ode_latex=sp.latex(ode),
                    solution_symbolic=str(solution),
                    solution_latex=sp.latex(solution),
                    initial_conditions=ics,
                    parameters=params,
                    complexity_score=properties.get('complexity_score', 0),
                    operation_count=properties.get('operation_count', 0),
                    atom_count=properties.get('atom_count', 0),
                    symbol_count=properties.get('symbol_count', 0),
                    has_pantograph=any('/' in str(arg) for arg in ode.atoms() if hasattr(arg, 'args')),
                    verified=verified,
                    verification_method=method if verified else VerificationMethod.FAILED,
                    verification_confidence=confidence,
                    generation_time=0.0  # Not tracked in parallel mode
                )
                
                results.append(ode_instance)
                
            except Exception as e:
                logger.debug(f"Failed to generate {gen_name} + {f_key}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return []