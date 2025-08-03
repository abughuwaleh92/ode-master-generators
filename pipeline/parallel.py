import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
import logging
import pickle
import dill  # Better serialization for complex objects
from functools import partial

from core.types import ODEInstance, GeneratorResult
from utils.config import ConfigManager

logger = logging.getLogger(__name__)

class ParallelODEGenerator:
    """Parallel ODE generation for large-scale datasets"""
    
    def __init__(self, config: ConfigManager, n_workers: Optional[int] = None):
        self.config = config
        self.n_workers = n_workers or config.get('performance.n_workers') or mp.cpu_count()
        
    def generate_batch_parallel(
        self, 
        generation_tasks: List[Dict[str, Any]]
    ) -> List[ODEInstance]:
        """
        Generate ODEs in parallel
        
        Args:
            generation_tasks: List of task dictionaries with keys:
                - generator_type: 'linear' or 'nonlinear'
                - generator_name: e.g., 'L1', 'N1'
                - function_name: e.g., 'identity', 'sine'
                - ode_id: unique identifier
                - params: parameter dictionary
        """
        logger.info(f"Starting parallel generation with {self.n_workers} workers")
        
        # Split tasks into chunks for workers
        chunk_size = max(1, len(generation_tasks) // self.n_workers)
        task_chunks = [
            generation_tasks[i:i + chunk_size]
            for i in range(0, len(generation_tasks), chunk_size)
        ]
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit tasks
            futures = []
            for chunk in task_chunks:
                future = executor.submit(self._process_chunk, chunk, self.config.config)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Parallel chunk processing error: {e}")
        
        logger.info(f"Parallel generation completed: {len(results)} ODEs generated")
        return results
    
    @staticmethod
    def _process_chunk(
        tasks: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> List[ODEInstance]:
        """Process a chunk of generation tasks (runs in separate process)"""
        # Import here to avoid pickling issues
        from generators.linear import LINEAR_GENERATORS
        from generators.nonlinear import NONLINEAR_GENERATORS
        from verification.verifier import ODEVerifier
        
        results = []
        verifier = ODEVerifier(config['verification'])
        
        # Initialize generators in the worker process
        linear_generators = {
            name: gen_class() for name, gen_class in LINEAR_GENERATORS.items()
        }
        nonlinear_generators = {
            name: gen_class() for name, gen_class in NONLINEAR_GENERATORS.items()
        }
        
        for task in tasks:
            try:
                # Get appropriate generator
                if task['generator_type'] == 'linear':
                    generator = linear_generators.get(task['generator_name'])
                else:
                    generator = nonlinear_generators.get(task['generator_name'])
                
                if not generator:
                    continue
                
                # Generate ODE
                result = generator.create_ode_instance(
                    task['function_name'],
                    task['params'],
                    task['ode_id']
                )
                
                if result.success and result.ode_instance:
                    # Verify
                    ode = result.ode_instance
                    verified, method, confidence = verifier.verify(
                        ode.ode_symbolic,
                        ode.solution_symbolic
                    )
                    
                    # Update verification info
                    ode.verified = verified
                    ode.verification_method = method
                    ode.verification_confidence = confidence
                    
                    # Analyze properties
                    properties = verifier.analyze_ode_properties(ode.ode_symbolic)
                    ode.complexity_score = properties['complexity_score']
                    ode.operation_count = properties['operation_count']
                    ode.atom_count = properties['atom_count']
                    ode.symbol_count = properties['symbol_count']
                    
                    results.append(ode)
                    
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing task: {e}")
                continue
        
        return results
    
    def create_generation_tasks(
        self,
        working_generators: Dict[str, Dict],
        function_names: List[str],
        samples_per_combo: int,
        starting_id: int = 0
    ) -> List[Dict[str, Any]]:
        """Create list of generation tasks for parallel processing"""
        tasks = []
        ode_id = starting_id
        
        # Parameter ranges from config
        param_ranges = self.config.get('generation.parameter_ranges')
        
        # Create tasks for linear generators
        for gen_name in working_generators.get('linear', {}):
            for f_name in function_names:
                for sample in range(samples_per_combo):
                    # Sample parameters
                    params = {
                        'alpha': np.random.choice(param_ranges['alpha']),
                        'beta': np.random.choice(param_ranges['beta']),
                        'M': np.random.choice(param_ranges['M']),
                        'q': np.random.choice(param_ranges['q']),
                        'v': np.random.choice(param_ranges['v']),
                        'a': np.random.choice(param_ranges['a'])
                    }
                    
                    tasks.append({
                        'generator_type': 'linear',
                        'generator_name': gen_name,
                        'function_name': f_name,
                        'ode_id': ode_id,
                        'params': params
                    })
                    ode_id += 1
        
        # Create tasks for nonlinear generators
        for gen_name in working_generators.get('nonlinear', {}):
            for f_name in function_names:
                for sample in range(samples_per_combo):
                    # Sample parameters
                    params = {
                        'alpha': np.random.choice(param_ranges['alpha']),
                        'beta': np.random.choice(param_ranges['beta']),
                        'M': np.random.choice(param_ranges['M']),
                        'q': np.random.choice(param_ranges['q']),
                        'v': np.random.choice(param_ranges['v']),
                        'a': np.random.choice(param_ranges['a'])
                    }
                    
                    tasks.append({
                        'generator_type': 'nonlinear',
                        'generator_name': gen_name,
                        'function_name': f_name,
                        'ode_id': ode_id,
                        'params': params
                    })
                    ode_id += 1
        
        return tasks