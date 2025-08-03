# scripts/parallel_generator.py
"""
Parallel ODE generation with load balancing

Benefits:
- Maximizes CPU utilization
- Dynamic load balancing
- Fault tolerance
- Real-time progress monitoring
"""

import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import signal
import sys
from tqdm import tqdm

from pipeline.generator import ODEGenerator

class ParallelODEGenerator:
    def __init__(self, 
                 n_workers: Optional[int] = None,
                 checkpoint_interval: int = 100):
        self.n_workers = n_workers or mp.cpu_count()
        self.checkpoint_interval = checkpoint_interval
        self.manager = Manager()
        self.progress_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.error_queue = self.manager.Queue()
        
    def worker_process(self, 
                      worker_id: int,
                      task_queue: Queue,
                      result_queue: Queue,
                      progress_queue: Queue,
                      error_queue: Queue):
        """Worker process for ODE generation"""
        
        # Set up worker-specific logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Worker-{worker_id}] %(asctime)s - %(message)s'
        )
        
        # Initialize generator for this worker
        generator = ODEGenerator(samples_per_combination=1)
        
        while True:
            try:
                # Get task from queue
                task = task_queue.get(timeout=1)
                
                if task is None:  # Poison pill
                    break
                    
                gen_name, func_name, sample_idx = task
                
                # Generate ODE
                start_time = time.time()
                ode_data = generator.generate_single(gen_name, func_name)
                generation_time = time.time() - start_time
                
                if ode_data:
                    ode_data['worker_id'] = worker_id
                    ode_data['generation_time'] = generation_time
                    result_queue.put(ode_data)
                    progress_queue.put(('success', worker_id, gen_name, func_name))
                else:
                    error_queue.put({
                        'worker_id': worker_id,
                        'generator': gen_name,
                        'function': func_name,
                        'error': 'Generation failed'
                    })
                    progress_queue.put(('failure', worker_id, gen_name, func_name))
                    
            except mp.TimeoutError:
                continue
            except Exception as e:
                error_queue.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'task': task if 'task' in locals() else None
                })
    
    def progress_monitor(self, total_tasks: int, pbar: tqdm):
        """Monitor and display progress"""
        completed = 0
        worker_stats = {}
        
        while completed < total_tasks:
            try:
                status, worker_id, gen_name, func_name = self.progress_queue.get(timeout=1)
                
                if worker_id not in worker_stats:
                    worker_stats[worker_id] = {'success': 0, 'failure': 0}
                
                worker_stats[worker_id][status] += 1
                completed += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Workers': len(worker_stats),
                    'Success': sum(s['success'] for s in worker_stats.values()),
                    'Failed': sum(s['failure'] for s in worker_stats.values())
                })
                
            except:
                continue
    
    def generate_parallel(self, 
                         generators: List[str],
                         functions: List[str],
                         samples_per_combination: int,
                         output_file: str) -> Dict:
        """
        Generate ODEs in parallel
        
        Returns:
            Statistics about the generation process
        """
        
        # Create task queue
        task_queue = self.manager.Queue()
        total_tasks = 0
        
        # Populate task queue
        for gen_name in generators:
            for func_name in functions:
                for i in range(samples_per_combination):
                    task_queue.put((gen_name, func_name, i))
                    total_tasks += 1
        
        # Add poison pills for workers
        for _ in range(self.n_workers):
            task_queue.put(None)
        
        print(f"Starting parallel generation with {self.n_workers} workers")
        print(f"Total tasks: {total_tasks}")
        
        # Start workers
        workers = []
        for i in range(self.n_workers):
            p = mp.Process(
                target=self.worker_process,
                args=(i, task_queue, self.result_queue, 
                     self.progress_queue, self.error_queue)
            )
            p.start()
            workers.append(p)
        
        # Start progress monitor
        pbar = tqdm(total=total_tasks, desc="Generating ODEs")
        monitor = mp.Process(
            target=self.progress_monitor,
            args=(total_tasks, pbar)
        )
        monitor.start()
        
        # Collect results
        results = []
        errors = []
        
        with open(output_file, 'w') as f:
            while len(results) < total_tasks:
                # Check for results
                try:
                    result = self.result_queue.get(timeout=0.1)
                    results.append(result)
                    f.write(json.dumps(result) + '\n')
                    
                    # Checkpoint
                    if len(results) % self.checkpoint_interval == 0:
                        f.flush()
                        
                except:
                    pass
                
                # Check for errors
                try:
                    error = self.error_queue.get(timeout=0.1)
                    errors.append(error)
                except:
                    pass
                
                # Check if workers are alive
                alive_workers = sum(1 for w in workers if w.is_alive())
                if alive_workers == 0 and len(results) < total_tasks:
                    print(f"\nWarning: All workers died! Collected {len(results)}/{total_tasks}")
                    break
        
        # Wait for workers to finish
        for w in workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
        
        monitor.terminate()
        pbar.close()
        
        # Compute statistics
        stats = {
            'total_tasks': total_tasks,
            'completed': len(results),
            'errors': len(errors),
            'success_rate': len(results) / total_tasks if total_tasks > 0 else 0,
            'workers_used': self.n_workers,
            'output_file': output_file
        }
        
        # Save error log
        if errors:
            error_file = output_file.replace('.jsonl', '_errors.json')
            with open(error_file, 'w') as f:
                json.dump(errors, f, indent=2)
            stats['error_file'] = error_file
        
        return stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel ODE generation')
    parser.add_argument('--generators', nargs='+', 
                       default=['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3', 'N7'])
    parser.add_argument('--functions', nargs='+',
                       default=['identity', 'quadratic', 'sine', 'cosine', 'exponential'])
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--output', default='parallel_odes.jsonl')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ParallelODEGenerator(n_workers=args.workers)
    
    # Run generation
    stats = generator.generate_parallel(
        generators=args.generators,
        functions=args.functions,
        samples_per_combination=args.samples,
        output_file=args.output
    )
    
    print(f"\nGeneration complete!")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()