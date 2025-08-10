# utils/performance_optimizer.py
"""
Performance optimization utilities
"""

import numpy as np
import sympy as sp
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import multiprocessing as mp
import asyncio
import pickle
import hashlib
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Advanced caching system for ODE operations
    """
    
    def __init__(self, cache_dir: str = "cache", max_memory_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_mb = max_memory_mb
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = pickle.dumps((args, kwargs))
        return hashlib.sha256(key_data).hexdigest()
    
    def disk_cache(self, prefix: str = ""):
        """Decorator for disk caching"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = self.cache_key(*args, **kwargs)
                cache_file = self.cache_dir / f"{prefix}_{key}.pkl"
                
                # Check disk cache
                if cache_file.exists():
                    self.cache_stats['hits'] += 1
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                
                # Compute and cache
                self.cache_stats['misses'] += 1
                result = func(*args, **kwargs)
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                
                return result
            return wrapper
        return decorator
    
    def memory_cache_with_ttl(self, ttl_seconds: int = 3600):
        """Memory cache with time-to-live"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = self.cache_key(*args, **kwargs)
                
                # Check memory cache
                if key in self.memory_cache:
                    cached_time, cached_result = self.memory_cache[key]
                    if time.time() - cached_time < ttl_seconds:
                        self.cache_stats['hits'] += 1
                        return cached_result
                
                # Compute and cache
                self.cache_stats['misses'] += 1
                result = func(*args, **kwargs)
                self.memory_cache[key] = (time.time(), result)
                
                # Check memory limit
                self._check_memory_limit()
                
                return result
            return wrapper
        return decorator
    
    def _check_memory_limit(self):
        """Evict old entries if memory limit exceeded"""
        import sys
        
        # Estimate memory usage
        memory_usage_mb = sys.getsizeof(self.memory_cache) / (1024 * 1024)
        
        if memory_usage_mb > self.max_memory_mb:
            # Evict oldest 20% of entries
            items = list(self.memory_cache.items())
            items.sort(key=lambda x: x[1][0])  # Sort by timestamp
            
            evict_count = len(items) // 5
            for key, _ in items[:evict_count]:
                del self.memory_cache[key]
                self.cache_stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total if total > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'cache_size_mb': len(self.memory_cache) * 0.001  # Rough estimate
        }


class ParallelProcessor:
    """
    Optimized parallel processing for ODE generation
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or mp.cpu_count()
        self.executor = None
    
    def __enter__(self):
        self.executor = mp.Pool(processes=self.n_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.close()
            self.executor.join()
    
    def map_async(
        self,
        func: Callable,
        tasks: List[Any],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        Asynchronous parallel map with progress tracking
        """
        if not chunk_size:
            chunk_size = max(1, len(tasks) // (self.n_workers * 4))
        
        results = []
        async_results = []
        
        # Submit tasks in chunks
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            async_result = self.executor.apply_async(
                self._process_chunk,
                args=(func, chunk)
            )
            async_results.append(async_result)
        
        # Collect results with progress
        from tqdm import tqdm
        
        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for async_result in async_results:
                chunk_results = async_result.get()
                results.extend(chunk_results)
                pbar.update(len(chunk_results))
        
        return results
    
    @staticmethod
    def _process_chunk(func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of tasks"""
        return [func(task) for task in chunk]
    
    def parallel_generate(
        self,
        generator_func: Callable,
        parameter_sets: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Parallel ODE generation with load balancing
        """
        # Group by expected complexity
        simple_params = []
        complex_params = []
        
        for params in parameter_sets:
            if params.get('q', 2) > 3 or params.get('v', 2) > 3:
                complex_params.append(params)
            else:
                simple_params.append(params)
        
        # Process complex ones with smaller chunks
        results = []
        
        if complex_params:
            complex_results = self.map_async(
                generator_func,
                complex_params,
                chunk_size=max(1, len(complex_params) // (self.n_workers * 8))
            )
            results.extend(complex_results)
        
        if simple_params:
            simple_results = self.map_async(
                generator_func,
                simple_params,
                chunk_size=max(1, len(simple_params) // (self.n_workers * 2))
            )
            results.extend(simple_results)
        
        return results


class GPUAccelerator:
    """
    GPU acceleration for numerical operations (requires CuPy)
    """
    
    def __init__(self):
        self.gpu_available = self._check_gpu()
        
        if self.gpu_available:
            import cupy as cp
            self.cp = cp
            logger.info("GPU acceleration enabled")
        else:
            logger.info("GPU not available, using CPU")
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return True
        except:
            return False
    
    def accelerate_numerical_verification(
        self,
        ode_func: Callable,
        solution_func: Callable,
        test_points: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated numerical verification
        """
        if not self.gpu_available:
            # CPU fallback
            residuals = []
            for x in test_points:
                try:
                    ode_val = ode_func(x)
                    sol_val = solution_func(x)
                    residuals.append(abs(ode_val - sol_val))
                except:
                    residuals.append(np.inf)
            return np.array(residuals)
        
        # GPU computation
        try:
            # Transfer to GPU
            x_gpu = self.cp.asarray(test_points)
            
            # Vectorized computation on GPU
            ode_vals = self.cp.vectorize(ode_func)(x_gpu)
            sol_vals = self.cp.vectorize(solution_func)(x_gpu)
            
            residuals_gpu = self.cp.abs(ode_vals - sol_vals)
            
            # Transfer back to CPU
            return self.cp.asnumpy(residuals_gpu)
            
        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            return self.accelerate_numerical_verification(
                ode_func, solution_func, test_points
            )


class SymbolicOptimizer:
    """
    Optimizations for symbolic computations
    """
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def cached_derivative(expr_str: str, var_str: str, order: int) -> str:
        """Cached symbolic differentiation"""
        expr = sp.sympify(expr_str)
        var = sp.Symbol(var_str)
        result = sp.diff(expr, var, order)
        return str(result)
    
    @staticmethod
    @lru_cache(maxsize=512)
    def cached_simplify(expr_str: str) -> str:
        """Cached simplification"""
        expr = sp.sympify(expr_str)
        # Try multiple simplification strategies
        strategies = [
            lambda e: e,
            sp.simplify,
            sp.expand,
            sp.factor,
            sp.cancel,
            sp.trigsimp,
            sp.radsimp
        ]
        
        best_expr = expr
        best_ops = sp.count_ops(expr)
        
        for strategy in strategies:
            try:
                simplified = strategy(expr)
                ops = sp.count_ops(simplified)
                if ops < best_ops:
                    best_expr = simplified
                    best_ops = ops
            except:
                continue
        
        return str(best_expr)
    
    @staticmethod
    def parallel_simplify(expressions: List[sp.Expr], n_workers: int = 4) -> List[sp.Expr]:
        """Parallel simplification of multiple expressions"""
        with mp.Pool(processes=n_workers) as pool:
            str_exprs = [str(expr) for expr in expressions]
            simplified_strs = pool.map(SymbolicOptimizer.cached_simplify, str_exprs)
            return [sp.sympify(s) for s in simplified_strs]


# Global cache manager instance
cache_manager = CacheManager()

# Performance monitoring decorator
def monitor_performance(func: Callable) -> Callable:
    """Monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0
        
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except:
            pass
        
        result = func(*args, **kwargs)
        
        elapsed_time = time.time() - start_time
        
        try:
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory
            
            logger.debug(
                f"{func.__name__} - Time: {elapsed_time:.3f}s, "
                f"Memory: {memory_used:.1f}MB"
            )
        except:
            logger.debug(f"{func.__name__} - Time: {elapsed_time:.3f}s")
        
        return result
    return wrapper
