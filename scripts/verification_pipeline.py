# scripts/verification_pipeline.py
"""
Advanced verification pipeline with multiple strategies

Benefits:
- Multiple verification methods
- Confidence scoring
- Automatic fallback strategies
- Detailed error analysis
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sympy as sp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm

from verification.symbolic_verifier import SymbolicVerifier
from verification.numerical_verifier import NumericalVerifier

class AdvancedVerificationPipeline:
    def __init__(self, 
                 methods: List[str] = ['substitution', 'numerical', 'series'],
                 confidence_threshold: float = 0.95):
        
        self.methods = methods
        self.confidence_threshold = confidence_threshold
        self.verifiers = {
            'substitution': SymbolicVerifier(method='substitution'),
            'numerical': NumericalVerifier(method='rk45'),
            'series': SymbolicVerifier(method='series')
        }
        
        self.stats = {
            'total': 0,
            'verified': 0,
            'failed': 0,
            'by_method': {m: 0 for m in methods},
            'confidence_distribution': []
        }
    
    def verify_ode(self, ode_data: Dict) -> Dict:
        """
        Verify single ODE with multiple methods
        
        Returns:
            Updated ODE data with verification results
        """
        self.stats['total'] += 1
        
        ode_str = ode_data.get('ode_symbolic')
        solution_str = ode_data.get('solution_symbolic')
        
        if not ode_str or not solution_str:
            ode_data['verification_status'] = 'missing_data'
            self.stats['failed'] += 1
            return ode_data
        
        # Try each verification method
        verification_results = []
        
        for method in self.methods:
            try:
                verifier = self.verifiers[method]
                result = verifier.verify(ode_str, solution_str)
                
                verification_results.append({
                    'method': method,
                    'verified': result.get('verified', False),
                    'confidence': result.get('confidence', 0.0),
                    'details': result.get('details', {})
                })
                
                # If high confidence verification, we can stop
                if result.get('verified') and result.get('confidence', 0) >= self.confidence_threshold:
                    break
                    
            except Exception as e:
                verification_results.append({
                    'method': method,
                    'verified': False,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # Aggregate results
        verified_count = sum(1 for r in verification_results if r['verified'])
        max_confidence = max(r['confidence'] for r in verification_results)
        
        # Decision logic
        if verified_count > len(self.methods) / 2:  # Majority vote
            ode_data['verified'] = True
            ode_data['verification_confidence'] = max_confidence
            self.stats['verified'] += 1
        else:
            ode_data['verified'] = False
            ode_data['verification_confidence'] = max_confidence
            self.stats['failed'] += 1
        
        # Record which method succeeded
        for result in verification_results:
            if result['verified']:
                self.stats['by_method'][result['method']] += 1
                break
        
        self.stats['confidence_distribution'].append(max_confidence)
        
        # Add detailed results
        ode_data['verification_results'] = verification_results
        ode_data['verification_method'] = 'multi_method_ensemble'
        
        return ode_data
    
    def verify_dataset(self, 
                      input_file: str,
                      output_file: str,
                      max_workers: int = 4,
                      batch_size: int = 100):
        """Verify entire dataset with parallel processing"""
        
        # Load dataset
        print(f"Loading dataset from {input_file}")
        odes = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    odes.append(json.loads(line))
        
        print(f"Loaded {len(odes)} ODEs for verification")
        
        # Process in batches
        verified_odes = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches
            futures = []
            for i in range(0, len(odes), batch_size):
                batch = odes[i:i+batch_size]
                future = executor.submit(self._verify_batch, batch)
                futures.append(future)
            
            # Collect results with progress bar
            with tqdm(total=len(odes), desc="Verifying ODEs") as pbar:
                for future in as_completed(futures):
                    batch_results = future.result()
                    verified_odes.extend(batch_results)
                    pbar.update(len(batch_results))
        
        # Save results
        print(f"Saving verified dataset to {output_file}")
        with open(output_file, 'w') as f:
            for ode in verified_odes:
                f.write(json.dumps(ode) + '\n')
        
        # Print statistics
        self._print_stats()
        
        return self.stats
    
    def _verify_batch(self, batch: List[Dict]) -> List[Dict]:
        """Verify a batch of ODEs"""
        return [self.verify_ode(ode) for ode in batch]
    
    def _print_stats(self):
        """Print verification statistics"""
        print("\nVerification Statistics:")
        print(f"Total ODEs: {self.stats['total']}")
        print(f"Verified: {self.stats['verified']} ({100*self.stats['verified']/self.stats['total']:.1f}%)")
        print(f"Failed: {self.stats['failed']} ({100*self.stats['failed']/self.stats['total']:.1f}%)")
        
        print("\nSuccess by method:")
        for method, count in self.stats['by_method'].items():
            print(f"  {method}: {count}")
        
        if self.stats['confidence_distribution']:
            confidences = self.stats['confidence_distribution']
            print(f"\nConfidence statistics:")
            print(f"  Mean: {np.mean(confidences):.3f}")
            print(f"  Std: {np.std(confidences):.3f}")
            print(f"  Min: {np.min(confidences):.3f}")
            print(f"  Max: {np.max(confidences):.3f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced ODE verification')
    parser.add_argument('input', help='Input dataset')
    parser.add_argument('--output', default=None, help='Output file')
    parser.add_argument('--methods', nargs='+', 
                       default=['substitution', 'numerical', 'series'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--confidence', type=float, default=0.95)
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.replace('.jsonl', '_verified.jsonl')
    
    # Create pipeline
    pipeline = AdvancedVerificationPipeline(
        methods=args.methods,
        confidence_threshold=args.confidence
    )
    
    # Run verification
    stats = pipeline.verify_dataset(
        args.input,
        args.output,
        max_workers=args.workers
    )
    
    # Save stats
    stats_file = args.output.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nVerification complete! Stats saved to {stats_file}")

if __name__ == "__main__":
    main()