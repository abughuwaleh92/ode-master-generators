# scripts/dataset_enhancer.py
"""
Enhance dataset quality through various techniques

Benefits:
- Automatic error correction
- Missing data imputation
- Quality scoring
- Dataset augmentation
"""

import json
import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
from tqdm import tqdm

class ODEDatasetEnhancer:
    def __init__(self):
        self.enhancement_stats = {
            'total_processed': 0,
            'errors_fixed': 0,
            'missing_data_filled': 0,
            'augmented': 0,
            'quality_improved': 0
        }
    
    def enhance_dataset(self, 
                       input_file: str,
                       output_file: str,
                       augment: bool = True):
        """
        Enhance entire dataset
        
        Steps:
        1. Fix syntax errors
        2. Fill missing data
        3. Augment with variations
        4. Improve quality scores
        """
        
        enhanced_odes = []
        
        print(f"Loading dataset from {input_file}")
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc="Enhancing ODEs"):
                if line.strip():
                    ode_data = json.loads(line)
                    
                    # Enhance ODE
                    enhanced = self.enhance_ode(ode_data)
                    enhanced_odes.append(enhanced)
                    
                    # Augment if requested
                    if augment and enhanced.get('verified', False):
                        augmented = self.augment_ode(enhanced)
                        enhanced_odes.extend(augmented)
                    
                    self.enhancement_stats['total_processed'] += 1
        
        # Save enhanced dataset
        print(f"Saving enhanced dataset to {output_file}")
        with open(output_file, 'w') as f:
            for ode in enhanced_odes:
                f.write(json.dumps(ode) + '\n')
        
        # Print statistics
        self._print_stats()
        
        return self.enhancement_stats
    
    def enhance_ode(self, ode_data: Dict) -> Dict:
        """Enhance single ODE"""
        
        # Fix syntax errors
        ode_data = self._fix_syntax_errors(ode_data)
        
        # Fill missing fields
        ode_data = self._fill_missing_data(ode_data)
        
        # Improve quality
        ode_data = self._improve_quality(ode_data)
        
        # Recompute complexity if needed
        if 'ode_symbolic' in ode_data:
            ode_data['complexity_score'] = self._compute_complexity(ode_data['ode_symbolic'])
        
        return ode_data
    
    def _fix_syntax_errors(self, ode_data: Dict) -> Dict:
        """Fix common syntax errors in ODEs"""
        
        ode_str = ode_data.get('ode_symbolic', '')
        if not ode_str:
            return ode_data
        
        original = ode_str
        
        # Common fixes
        # 1. Balance parentheses
        open_count = ode_str.count('(')
        close_count = ode_str.count(')')
        if open_count > close_count:
            ode_str += ')' * (open_count - close_count)
        elif close_count > open_count:
            ode_str = '(' * (close_count - open_count) + ode_str
        
        # 2. Fix derivative notation
        ode_str = re.sub(r"y'''\s*\(x\)", "Derivative(y(x), x, 3)", ode_str)
        ode_str = re.sub(r"y''\s*\(x\)", "Derivative(y(x), x, 2)", ode_str)
        ode_str = re.sub(r"y'\s*\(x\)", "Derivative(y(x), x)", ode_str)
        
        # 3. Fix common typos
        ode_str = ode_str.replace('**', '**')  # Fix double asterisks
        ode_str = re.sub(r'(\d)\s+(\d)', r'\1*\2', ode_str)  # Add missing multiplication
        
        # 4. Ensure equation format
        if '=' not in ode_str:
            ode_str += ' = 0'
        
        # Validate with SymPy
        try:
            sp.sympify(ode_str)
            ode_data['ode_symbolic'] = ode_str
            
            if ode_str != original:
                self.enhancement_stats['errors_fixed'] += 1
                ode_data['syntax_fixed'] = True
                
        except:
            # If still invalid, keep original
            pass
        
        return ode_data
    
    def _fill_missing_data(self, ode_data: Dict) -> Dict:
        """Fill missing fields with reasonable defaults"""
        
        filled_any = False
        
        # Essential fields
        if 'id' not in ode_data:
            ode_data['id'] = f"enhanced_{np.random.randint(1000000)}"
            filled_any = True
        
        if 'generator_name' not in ode_data:
            ode_data['generator_name'] = 'unknown'
            filled_any = True
        
        if 'function_name' not in ode_data:
            ode_data['function_name'] = 'unknown'
            filled_any = True
        
        if 'verified' not in ode_data:
            ode_data['verified'] = False
            filled_any = True
        
        # Complexity metrics
        if 'complexity_score' not in ode_data and 'ode_symbolic' in ode_data:
            ode_data['complexity_score'] = self._compute_complexity(ode_data['ode_symbolic'])
            filled_any = True
        
        # LaTeX representations
        if 'ode_latex' not in ode_data and 'ode_symbolic' in ode_data:
            try:
                ode = sp.sympify(ode_data['ode_symbolic'])
                ode_data['ode_latex'] = sp.latex(ode)
                filled_any = True
            except:
                pass
        
        if filled_any:
            self.enhancement_stats['missing_data_filled'] += 1
            ode_data['data_filled'] = True
        
        return ode_data
    
    def _improve_quality(self, ode_data: Dict) -> Dict:
        """Improve ODE quality through various techniques"""
        
        improved = False
        
        # 1. Simplify ODE if possible
        if 'ode_symbolic' in ode_data:
            try:
                ode = sp.sympify(ode_data['ode_symbolic'])
                simplified = sp.simplify(ode)
                
                if simplified != ode:
                    ode_data['ode_simplified'] = str(simplified)
                    improved = True
            except:
                pass
        
        # 2. Add mathematical properties
        if 'ode_symbolic' in ode_data:
            properties = self._analyze_ode_properties(ode_data['ode_symbolic'])
            if properties:
                ode_data['mathematical_properties'] = properties
                improved = True
        
        # 3. Quality score
        quality_score = self._compute_quality_score(ode_data)
        ode_data['quality_score'] = quality_score
        
        if quality_score > 0.8:
            ode_data['high_quality'] = True
            improved = True
        
        if improved:
            self.enhancement_stats['quality_improved'] += 1
        
        return ode_data
    
    def augment_ode(self, ode_data: Dict) -> List[Dict]:
        """Generate variations of the ODE"""
        
        augmented = []
        
        # 1. Parameter variations
        if 'parameters' in ode_data:
            params = ode_data['parameters']
            
            # Small perturbations
            for i in range(3):
                new_params = params.copy()
                for key in ['alpha', 'beta', 'M']:
                    if key in new_params:
                        # Add small noise
                        new_params[key] *= (1 + np.random.normal(0, 0.1))
                
                new_ode = ode_data.copy()
                new_ode['parameters'] = new_params
                new_ode['id'] = f"{ode_data['id']}_aug_{i}"
                new_ode['augmented'] = True
                new_ode['augmentation_type'] = 'parameter_perturbation'
                
                augmented.append(new_ode)
        
        # 2. Complexity variations
        if 'ode_symbolic' in ode_data:
            try:# Add complexity by introducing terms
               ode = sp.sympify(ode_data['ode_symbolic'])
               
               # Add small nonlinear term
               y = sp.Function('y')
               x = sp.Symbol('x')
               
               if isinstance(ode, sp.Eq):
                   # Add small perturbation term
                   epsilon = 0.01
                   perturbation = epsilon * sp.sin(x) * y(x)
                   
                   new_lhs = ode.lhs + perturbation
                   new_ode = sp.Eq(new_lhs, ode.rhs)
                   
                   augmented_ode = ode_data.copy()
                   augmented_ode['id'] = f"{ode_data['id']}_aug_complex"
                   augmented_ode['ode_symbolic'] = str(new_ode)
                   augmented_ode['augmented'] = True
                   augmented_ode['augmentation_type'] = 'complexity_variation'
                   augmented_ode['complexity_score'] = self._compute_complexity(str(new_ode))
                   
                   augmented.append(augmented_ode)
           except:
               pass
       
       self.enhancement_stats['augmented'] += len(augmented)
       return augmented
   
   def _compute_complexity(self, ode_str: str) -> int:
       """Compute complexity score for ODE"""
       
       complexity = len(ode_str)  # Base complexity
       
       # Add weights for different elements
       complexity += ode_str.count('sin') * 5
       complexity += ode_str.count('cos') * 5
       complexity += ode_str.count('exp') * 7
       complexity += ode_str.count('log') * 6
       complexity += ode_str.count('**') * 3
       complexity += ode_str.count('Derivative') * 10
       
       return complexity
   
   def _analyze_ode_properties(self, ode_str: str) -> Dict:
       """Analyze mathematical properties of ODE"""
       
       properties = {}
       
       try:
           ode = sp.sympify(ode_str)
           
           # Check linearity
           y = sp.Function('y')
           x = sp.Symbol('x')
           
           # Simple linearity check
           properties['appears_linear'] = '**' not in ode_str and all(
               func not in ode_str for func in ['sin', 'cos', 'exp', 'log']
           )
           
           # Order detection
           if 'Derivative' in str(ode):
               max_order = 0
               for i in range(1, 5):
                   if f'Derivative(y(x), x, {i})' in str(ode) or f'Derivative(y(x), (x, {i}))' in str(ode):
                       max_order = i
               properties['order'] = max_order
           
           # Autonomous check
           properties['autonomous'] = 'x' not in str(ode).replace('y(x)', 'y')
           
           # Homogeneous check (simplified)
           properties['appears_homogeneous'] = '= 0' in ode_str
           
       except:
           pass
       
       return properties
   
   def _compute_quality_score(self, ode_data: Dict) -> float:
       """Compute quality score for ODE"""
       
       score = 0.0
       max_score = 0.0
       
       # Verification status (40%)
       max_score += 0.4
       if ode_data.get('verified', False):
           score += 0.4
       
       # Has solution (20%)
       max_score += 0.2
       if 'solution_symbolic' in ode_data and ode_data['solution_symbolic']:
           score += 0.2
       
       # Has LaTeX (10%)
       max_score += 0.1
       if 'ode_latex' in ode_data and ode_data['ode_latex']:
           score += 0.1
       
       # Has initial conditions (10%)
       max_score += 0.1
       if 'initial_conditions' in ode_data and ode_data['initial_conditions']:
           score += 0.1
       
       # Reasonable complexity (10%)
       max_score += 0.1
       complexity = ode_data.get('complexity_score', 0)
       if 50 <= complexity <= 500:  # Reasonable range
           score += 0.1
       elif 20 <= complexity <= 1000:  # Acceptable range
           score += 0.05
       
       # Has parameters (10%)
       max_score += 0.1
       if 'parameters' in ode_data and len(ode_data['parameters']) > 0:
           score += 0.1
       
       return score / max_score if max_score > 0 else 0.0
   
   def _print_stats(self):
       """Print enhancement statistics"""
       
       print("\nEnhancement Statistics:")
       print(f"Total processed: {self.enhancement_stats['total_processed']}")
       print(f"Syntax errors fixed: {self.enhancement_stats['errors_fixed']}")
       print(f"Missing data filled: {self.enhancement_stats['missing_data_filled']}")
       print(f"ODEs augmented: {self.enhancement_stats['augmented']}")
       print(f"Quality improved: {self.enhancement_stats['quality_improved']}")


# scripts/dataset_cleaner.py
"""
Clean and validate ODE dataset

Benefits:
- Remove duplicates
- Filter invalid ODEs
- Standardize format
- Ensure consistency
"""

class ODEDatasetCleaner:
   def __init__(self):
       self.cleaning_stats = {
           'total_input': 0,
           'duplicates_removed': 0,
           'invalid_removed': 0,
           'format_standardized': 0,
           'total_output': 0
       }
   
   def clean_dataset(self, 
                    input_file: str,
                    output_file: str,
                    remove_duplicates: bool = True,
                    validate_odes: bool = True,
                    min_quality_score: float = 0.3):
       """Clean entire dataset"""
       
       seen_odes = set()
       cleaned_odes = []
       
       print(f"Cleaning dataset from {input_file}")
       
       with open(input_file, 'r') as f:
           for line in tqdm(f, desc="Cleaning ODEs"):
               if not line.strip():
                   continue
               
               self.cleaning_stats['total_input'] += 1
               
               try:
                   ode_data = json.loads(line)
                   
                   # Remove duplicates
                   if remove_duplicates:
                       ode_key = self._get_ode_key(ode_data)
                       if ode_key in seen_odes:
                           self.cleaning_stats['duplicates_removed'] += 1
                           continue
                       seen_odes.add(ode_key)
                   
                   # Validate ODE
                   if validate_odes and not self._validate_ode(ode_data):
                       self.cleaning_stats['invalid_removed'] += 1
                       continue
                   
                   # Check quality threshold
                   if 'quality_score' in ode_data:
                       if ode_data['quality_score'] < min_quality_score:
                           self.cleaning_stats['invalid_removed'] += 1
                           continue
                   
                   # Standardize format
                   ode_data = self._standardize_format(ode_data)
                   
                   cleaned_odes.append(ode_data)
                   
               except Exception as e:
                   print(f"Error processing ODE: {e}")
                   self.cleaning_stats['invalid_removed'] += 1
       
       # Sort by ID for consistency
       cleaned_odes.sort(key=lambda x: x.get('id', ''))
       
       # Save cleaned dataset
       print(f"Saving cleaned dataset to {output_file}")
       with open(output_file, 'w') as f:
           for ode in cleaned_odes:
               f.write(json.dumps(ode) + '\n')
       
       self.cleaning_stats['total_output'] = len(cleaned_odes)
       
       # Print statistics
       self._print_stats()
       
       return self.cleaning_stats
   
   def _get_ode_key(self, ode_data: Dict) -> str:
       """Generate unique key for ODE"""
       
       # Use ODE string and solution as key
       ode_str = ode_data.get('ode_symbolic', '')
       sol_str = ode_data.get('solution_symbolic', '')
       
       # Normalize by removing spaces
       ode_str = ode_str.replace(' ', '')
       sol_str = sol_str.replace(' ', '')
       
       return f"{ode_str}|{sol_str}"
   
   def _validate_ode(self, ode_data: Dict) -> bool:
       """Validate ODE data"""
       
       # Check required fields
       required_fields = ['id', 'generator_name', 'ode_symbolic']
       for field in required_fields:
           if field not in ode_data:
               return False
       
       # Validate ODE syntax
       try:
           ode_str = ode_data['ode_symbolic']
           if not ode_str or not isinstance(ode_str, str):
               return False
           
           # Try to parse with SymPy
           ode = sp.sympify(ode_str)
           
           # Check if it's an equation
           if not isinstance(ode, sp.Eq):
               return False
           
           return True
           
       except:
           return False
   
   def _standardize_format(self, ode_data: Dict) -> Dict:
       """Standardize ODE data format"""
       
       # Ensure consistent field types
       if 'verified' in ode_data:
           ode_data['verified'] = bool(ode_data['verified'])
       
       if 'complexity_score' in ode_data:
           ode_data['complexity_score'] = int(ode_data['complexity_score'])
       
       # Standardize parameter format
       if 'parameters' in ode_data:
           params = ode_data['parameters']
           if isinstance(params, dict):
               # Convert all values to float
               for key, value in params.items():
                   if isinstance(value, (int, float)):
                       params[key] = float(value)
       
       # Add timestamp if missing
       if 'timestamp' not in ode_data:
           ode_data['timestamp'] = str(pd.Timestamp.now())
       
       self.cleaning_stats['format_standardized'] += 1
       
       return ode_data
   
   def _print_stats(self):
       """Print cleaning statistics"""
       
       print("\nCleaning Statistics:")
       print(f"Total input ODEs: {self.cleaning_stats['total_input']}")
       print(f"Duplicates removed: {self.cleaning_stats['duplicates_removed']}")
       print(f"Invalid ODEs removed: {self.cleaning_stats['invalid_removed']}")
       print(f"Format standardized: {self.cleaning_stats['format_standardized']}")
       print(f"Total output ODEs: {self.cleaning_stats['total_output']}")
       print(f"Reduction: {100*(1-self.cleaning_stats['total_output']/self.cleaning_stats['total_input']):.1f}%")


def main():
   import argparse
   
   parser = argparse.ArgumentParser(description='Enhance and clean ODE dataset')
   parser.add_argument('input', help='Input dataset')
   parser.add_argument('--output', help='Output dataset')
   parser.add_argument('--enhance', action='store_true', help='Enhance dataset')
   parser.add_argument('--clean', action='store_true', help='Clean dataset')
   parser.add_argument('--augment', action='store_true', help='Augment dataset')
   parser.add_argument('--min-quality', type=float, default=0.3, help='Minimum quality score')
   
   args = parser.parse_args()
   
   if not args.output:
       base_name = Path(args.input).stem
       suffix = []
       if args.enhance:
           suffix.append('enhanced')
       if args.clean:
           suffix.append('cleaned')
       if args.augment:
           suffix.append('augmented')
       
       args.output = f"{base_name}_{'_'.join(suffix)}.jsonl"
   
   # Run enhancement
   if args.enhance:
       enhancer = ODEDatasetEnhancer()
       temp_output = args.input.replace('.jsonl', '_enhanced_temp.jsonl')
       enhancer.enhance_dataset(args.input, temp_output, augment=args.augment)
       args.input = temp_output
   
   # Run cleaning
   if args.clean:
       cleaner = ODEDatasetCleaner()
       cleaner.clean_dataset(args.input, args.output, min_quality_score=args.min_quality)
   else:
       # Just rename if only enhancing
       import shutil
       shutil.move(args.input, args.output)
   
   # Cleanup temp file
   if args.enhance and args.clean:
       Path(temp_output).unlink()
   
   print(f"\nDataset processing complete! Output: {args.output}")

if __name__ == "__main__":
   main()