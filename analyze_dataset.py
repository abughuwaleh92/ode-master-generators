# analyze_dataset.py
#!/usr/bin/env python
"""Analyze generated ODE dataset"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Any

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load ODE dataset from JSONL file"""
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset

def analyze_dataset(dataset: List[Dict[str, Any]]):
    """Comprehensive dataset analysis"""
    print(f"\n{'='*60}")
    print("ODE Dataset Analysis")
    print('='*60)
    
    # Basic statistics
    print(f"\nTotal ODEs: {len(dataset)}")
    
    if not dataset:
        print("No ODEs in dataset!")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(dataset)
    
    # Verification statistics
    verified_count = df['verified'].sum()
    verification_rate = 100 * verified_count / len(df)
    print(f"Verified ODEs: {verified_count} ({verification_rate:.1f}%)")
    
    # Generator type distribution
    print("\nGenerator Type Distribution:")
    gen_type_counts = df['generator_type'].value_counts()
    for gen_type, count in gen_type_counts.items():
        print(f"  {gen_type}: {count} ({100*count/len(df):.1f}%)")
    
    # Generator performance
    print("\nGenerator Performance:")
    gen_stats = df.groupby('generator_name').agg({
        'verified': ['count', 'sum', 'mean']
    }).round(3)
    gen_stats.columns = ['Total', 'Verified', 'Rate']
    print(gen_stats)
    
    # Function distribution
    print("\nTop 10 Functions by Usage:")
    func_counts = df['function_name'].value_counts().head(10)
    for func, count in func_counts.items():
        print(f"  {func}: {count}")
    
    # Complexity analysis
    print("\nComplexity Statistics:")
    print(f"  Mean: {df['complexity_score'].mean():.1f}")
    print(f"  Std: {df['complexity_score'].std():.1f}")
    print(f"  Min: {df['complexity_score'].min()}")
    print(f"  Max: {df['complexity_score'].max()}")
    print(f"  Median: {df['complexity_score'].median()}")
    
    # High complexity ODEs
    high_complexity = df[df['complexity_score'] > 100]
    print(f"\nHigh complexity ODEs (>100): {len(high_complexity)}")
    
    # Pantograph equations
    pantograph_count = df['has_pantograph'].sum() if 'has_pantograph' in df else 0
    print(f"Pantograph equations: {pantograph_count}")
    
    # Verification methods
    print("\nVerification Methods:")
    method_counts = df['verification_method'].value_counts()
    for method, count in method_counts.items():
        print(f"  {method}: {count} ({100*count/len(df):.1f}%)")
    
    # Generation time analysis
    if 'generation_time' in df:
        print("\nGeneration Time Statistics:")
        print(f"  Mean: {df['generation_time'].mean()*1000:.1f} ms")
        print(f"  Std: {df['generation_time'].std()*1000:.1f} ms")
        print(f"  Min: {df['generation_time'].min()*1000:.1f} ms")
        print(f"  Max: {df['generation_time'].max()*1000:.1f} ms")
    
    # Nonlinearity analysis
    nonlinear_df = df[df['generator_type'] == 'nonlinear']
    if len(nonlinear_df) > 0 and 'nonlinearity_metrics' in nonlinear_df.columns:
        print("\nNonlinearity Metrics:")
        # Extract nonlinearity metrics
        metrics = []
        for idx, row in nonlinear_df.iterrows():
            if row['nonlinearity_metrics']:
                metrics.append(row['nonlinearity_metrics'])
        
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            print(f"  Exponential nonlinear: {metrics_df['is_exponential_nonlinear'].sum()}")
            print(f"  Logarithmic nonlinear: {metrics_df['is_logarithmic_nonlinear'].sum()}")
            print(f"  With pantograph: {metrics_df['has_pantograph'].sum()}")
    
    return df

def create_visualizations(df: pd.DataFrame, output_dir: str = "analysis_plots"):
    """Create visualization plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Verification rate by generator
    plt.figure(figsize=(10, 6))
    gen_verification = df.groupby('generator_name')['verified'].mean() * 100
    gen_verification.plot(kind='bar')
    plt.title('Verification Rate by Generator')
    plt.xlabel('Generator')
    plt.ylabel('Verification Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/verification_by_generator.png")
    plt.close()
    
    # 2. Complexity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['complexity_score'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('ODE Complexity Distribution')
    plt.xlabel('Complexity Score')
    plt.ylabel('Count')
    plt.axvline(df['complexity_score'].mean(), color='red', 
                linestyle='dashed', linewidth=2, label=f'Mean: {df["complexity_score"].mean():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/complexity_distribution.png")
    plt.close()
    
    # 3. Generator type pie chart
    plt.figure(figsize=(8, 8))
    gen_type_counts = df['generator_type'].value_counts()
    plt.pie(gen_type_counts.values, labels=gen_type_counts.index, autopct='%1.1f%%')
    plt.title('Generator Type Distribution')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/generator_type_pie.png")
    plt.close()
    
    # 4. Generation time boxplot (if available)
    if 'generation_time' in df:
        plt.figure(figsize=(10, 6))
        df_time = df.copy()
        df_time['generation_time_ms'] = df_time['generation_time'] * 1000
        df_time.boxplot(column='generation_time_ms', by='generator_name', rot=45)
        plt.title('Generation Time by Generator')
        plt.suptitle('')  # Remove default title
        plt.ylabel('Generation Time (ms)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/generation_time_boxplot.png")
    plt.close()
    
    print(f"\nVisualization saved to {output_dir}/")

def export_statistics(df: pd.DataFrame, output_file: str = "dataset_statistics.json"):
    """Export detailed statistics to JSON"""
    stats = {
        'total_odes': len(df),
        'verified_odes': int(df['verified'].sum()),
        'verification_rate': float(df['verified'].mean()),
        'generator_types': df['generator_type'].value_counts().to_dict(),
        'generator_performance': df.groupby('generator_name').agg({
            'verified': ['count', 'sum', 'mean']
        }).to_dict(),
        'complexity_stats': {
            'mean': float(df['complexity_score'].mean()),
            'std': float(df['complexity_score'].std()),
            'min': int(df['complexity_score'].min()),
            'max': int(df['complexity_score'].max()),
            'quartiles': df['complexity_score'].quantile([0.25, 0.5, 0.75]).tolist()
        },
        'function_distribution': df['function_name'].value_counts().to_dict(),
        'verification_methods': df['verification_method'].value_counts().to_dict()
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics exported to {output_file}")

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ODE dataset')
    parser.add_argument('dataset', nargs='?', default='ode_dataset.jsonl',
                        help='Path to dataset file')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--export', action='store_true',
                        help='Export statistics to JSON')
    
    args = parser.parse_args()
    
    # Load dataset
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file '{args.dataset}' not found!")
        return
    
    dataset = load_dataset(args.dataset)
    
    # Analyze
    df = analyze_dataset(dataset)
    
    if df is not None and not df.empty:
        # Create visualizations if requested
        if args.visualize:
            create_visualizations(df)
        
        # Export statistics if requested
        if args.export:
            export_statistics(df)

if __name__ == "__main__":
    main()