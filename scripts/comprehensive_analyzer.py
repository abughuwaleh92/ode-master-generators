# scripts/comprehensive_analyzer.py
"""
Comprehensive dataset analysis with visualizations

Benefits:
- Deep statistical analysis
- Pattern discovery
- Quality metrics
- Publication-ready visualizations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sympy as sp
from collections import defaultdict, Counter
from scipy import stats
import networkx as nx
from wordcloud import WordCloud

class ComprehensiveODEAnalyzer:
    def __init__(self, dataset_path: str, output_dir: str = 'analysis_output'):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        self.data = self._load_dataset()
        self.df = pd.DataFrame(self.data)
        
        # Configure plotting
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _load_dataset(self) -> List[Dict]:
        """Load JSONL dataset"""
        data = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def analyze_all(self):
        """Run all analyses"""
        print("Running comprehensive ODE dataset analysis...")
        
        # Basic statistics
        self.basic_statistics()
        
        # Generator performance
        self.analyze_generators()
        
        # Function distribution
        self.analyze_functions()
        
        # Complexity analysis
        self.analyze_complexity()
        
        # Verification analysis
        self.analyze_verification()
        
        # Pattern analysis
        self.analyze_patterns()
        
        # Network analysis
        self.analyze_relationships()
        
        # Generate report
        self.generate_report()
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")
    
    def basic_statistics(self):
        """Compute basic dataset statistics"""
        stats = {
            'total_odes': len(self.df),
            'unique_generators': self.df['generator_name'].nunique(),
            'unique_functions': self.df['function_name'].nunique(),
            'verified_odes': self.df['verified'].sum(),
            'verification_rate': self.df['verified'].mean(),
            'avg_complexity': self.df['complexity_score'].mean(),
            'std_complexity': self.df['complexity_score'].std(),
            'has_pantograph': self.df['has_pantograph'].sum() if 'has_pantograph' in self.df else 0
        }
        
        # Save stats
        with open(self.output_dir / 'basic_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def analyze_generators(self):
        """Analyze generator performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ODE count by generator
        gen_counts = self.df['generator_name'].value_counts()
        gen_counts.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('ODE Count by Generator')
        axes[0, 0].set_xlabel('Generator')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Verification rate by generator
        ver_rate = self.df.groupby('generator_name')['verified'].mean()
        ver_rate.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Verification Rate by Generator')
        axes[0, 1].set_xlabel('Generator')
        axes[0, 1].set_ylabel('Verification Rate')
        axes[0, 1].axhline(y=self.df['verified'].mean(), color='r', linestyle='--', label='Average')
        axes[0, 1].legend()
        
        # 3. Complexity distribution by generator
        self.df.boxplot(column='complexity_score', by='generator_name', ax=axes[1, 0])
        axes[1, 0].set_title('Complexity Distribution by Generator')
        axes[1, 0].set_xlabel('Generator')
        axes[1, 0].set_ylabel('Complexity Score')
        
        # 4. Generator type distribution
        gen_types = self.df['generator_type'].value_counts()
        gen_types.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
        axes[1, 1].set_title('Generator Type Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'generator_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_functions(self):
        """Analyze function distribution and performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top 20 functions
        top_funcs = self.df['function_name'].value_counts().head(20)
        top_funcs.plot(kind='barh', ax=axes[0, 0])
        axes[0, 0].set_title('Top 20 Functions Used')
        axes[0, 0].set_xlabel('Count')
        
        # 2. Function verification rates
        func_ver = self.df.groupby('function_name')['verified'].agg(['mean', 'count'])
        func_ver_filtered = func_ver[func_ver['count'] >= 10].sort_values('mean', ascending=False)
        
        func_ver_filtered['mean'].head(20).plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Verification Rate by Function (min 10 ODEs)')
        axes[0, 1].set_ylabel('Verification Rate')
        
        # 3. Function complexity heatmap
        pivot_table = self.df.pivot_table(
            values='complexity_score',
            index='generator_name',
            columns='function_name',
            aggfunc='mean'
        )
        
        # Select top functions for readability
        top_func_names = self.df['function_name'].value_counts().head(10).index
        pivot_subset = pivot_table[top_func_names]
        
        sns.heatmap(pivot_subset, cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Avg Complexity'})
        axes[1, 0].set_title('Average Complexity: Generators vs Functions')
        
        # 4. Function word cloud
        func_text = ' '.join(self.df['function_name'].tolist())
        wordcloud = WordCloud(width=400, height=400, background_color='white').generate(func_text)
        
        axes[1, 1].imshow(wordcloud, interpolation='bilinear')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Function Word Cloud')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'function_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_complexity(self):
        """Analyze complexity patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Complexity distribution
        self.df['complexity_score'].hist(bins=50, ax=axes[0, 0], edgecolor='black')
        axes[0, 0].set_title('Complexity Score Distribution')
        axes[0, 0].set_xlabel('Complexity Score')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Complexity vs Verification
        verified_complexity = self.df[self.df['verified']]['complexity_score']
        unverified_complexity = self.df[~self.df['verified']]['complexity_score']
        
        axes[0, 1].hist([verified_complexity, unverified_complexity], 
                       bins=30, label=['Verified', 'Unverified'], alpha=0.7)
        axes[0, 1].set_title('Complexity by Verification Status')
        axes[0, 1].set_xlabel('Complexity Score')
        axes[0, 1].legend()
        
        # 3. Complexity components
        comp_cols = ['operation_count', 'atom_count', 'symbol_count']
        if all(col in self.df.columns for col in comp_cols):
            self.df[comp_cols].plot(kind='box', ax=axes[0, 2])
            axes[0, 2].set_title('Complexity Components')
            axes[0, 2].set_ylabel('Count')
        
        # 4. Complexity over time (if timestamp available)
        if 'timestamp' in self.df.columns:
            self.df['timestamp_hour'] = pd.to_datetime(self.df['timestamp']).dt.floor('H')
            complexity_time = self.df.groupby('timestamp_hour')['complexity_score'].mean()
            complexity_time.plot(ax=axes[1, 0])
            axes[1, 0].set_title('Average Complexity Over Time')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Avg Complexity')
        
        # 5. Complexity correlations
        if len(comp_cols) > 1 and all(col in self.df.columns for col in comp_cols):
            corr_matrix = self.df[comp_cols + ['complexity_score']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Complexity Metric Correlations')
        
        # 6. Complexity by parameter values
        if 'parameters' in self.data[0]:
            param_complexity = []
            for item in self.data:
                params = item.get('parameters', {})
                alpha = params.get('alpha', 0)
                beta = params.get('beta', 0)
                complexity = item.get('complexity_score', 0)
                param_complexity.append({'alpha': alpha, 'beta': beta, 'complexity': complexity})
            
            param_df = pd.DataFrame(param_complexity)
            scatter = axes[1, 2].scatter(param_df['alpha'], param_df['beta'], 
                                       c=param_df['complexity'], cmap='viridis', alpha=0.6)
            axes[1, 2].set_xlabel('Alpha')
            axes[1, 2].set_ylabel('Beta')
            axes[1, 2].set_title('Complexity by Parameters')
            plt.colorbar(scatter, ax=axes[1, 2], label='Complexity')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_verification(self):
        """Analyze verification patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Verification methods distribution
        if 'verification_method' in self.df.columns:
            ver_methods = self.df[self.df['verified']]['verification_method'].value_counts()
            ver_methods.plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%')
            axes[0, 0].set_title('Verification Methods Used')
        
        # 2. Verification confidence distribution
        if 'verification_confidence' in self.df.columns:
            self.df['verification_confidence'].hist(bins=50, ax=axes[0, 1])
            axes[0, 1].set_title('Verification Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].axvline(x=0.95, color='r', linestyle='--', label='High Confidence Threshold')
            axes[0, 1].legend()
        
        # 3. Verification rate by complexity bins
        complexity_bins = pd.qcut(self.df['complexity_score'], q=10, labels=False)
        ver_by_complexity = self.df.groupby(complexity_bins)['verified'].mean()
        
        ver_by_complexity.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Verification Rate by Complexity Decile')
        axes[1, 0].set_xlabel('Complexity Decile')
        axes[1, 0].set_ylabel('Verification Rate')
        
        # 4. Failed verification analysis
        failed_df = self.df[~self.df['verified']]
        if len(failed_df) > 0:
            failed_reasons = failed_df['generator_name'].value_counts()
            failed_reasons.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Failed Verifications by Generator')
            axes[1, 1].set_xlabel('Generator')
            axes[1, 1].set_ylabel('Failed Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'verification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_patterns(self):
        """Analyze ODE patterns and structures"""
        
        # Extract patterns from ODE strings
        patterns = defaultdict(int)
        operator_counts = defaultdict(int)
        function_patterns = defaultdict(int)
        
        for item in self.data:
            ode_str = item.get('ode_symbolic', '')
            
            # Count operators
            for op in ['+', '-', '*', '/', '**', '^']:
                operator_counts[op] += ode_str.count(op)
            
            # Count function calls
            for func in ['sin', 'cos', 'exp', 'log', 'sqrt', 'tan']:
                if func in ode_str:
                    function_patterns[func] += 1
            
            # Identify patterns
            if "y''" in ode_str and "y'" in ode_str:
                patterns['mixed_order'] += 1
            elif "y''" in ode_str:
                patterns['second_order'] += 1
            elif "y'" in ode_str:
                patterns['first_order'] += 1
            
            if 'exp' in ode_str and 'sin' in ode_str:
                patterns['exp_trig'] += 1
            
            if ode_str.count('(') > 5:
                patterns['highly_nested'] += 1
        
        # Visualize patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Operator distribution
        pd.Series(operator_counts).plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Operator Usage Distribution')
        axes[0, 0].set_xlabel('Operator')
        axes[0, 0].set_ylabel('Total Count')
        
        # 2. Function usage
        pd.Series(function_patterns).plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Mathematical Function Usage')
        axes[0, 1].set_xlabel('Function')
        axes[0, 1].set_ylabel('ODE Count')
        
        # 3. ODE patterns
        pd.Series(patterns).plot(kind='pie', ax=axes[1, 0], autopct='%1.1f%%')
        axes[1, 0].set_title('ODE Structure Patterns')
        
        # 4. Complexity vs pattern
        pattern_complexity = []
        for item in self.data:
            ode_str = item.get('ode_symbolic', '')
            complexity = item.get('complexity_score', 0)
            
            if "y''" in ode_str and "y'" in ode_str:
                pattern_type = 'Mixed Order'
            elif "y''" in ode_str:
                pattern_type = 'Second Order'
            elif "y'" in ode_str:
                pattern_type = 'First Order'
            else:
                pattern_type = 'Other'
            
            pattern_complexity.append({'pattern': pattern_type, 'complexity': complexity})
        
        pattern_df = pd.DataFrame(pattern_complexity)
        pattern_df.boxplot(column='complexity', by='pattern', ax=axes[1, 1])
        axes[1, 1].set_title('Complexity by ODE Pattern')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_relationships(self):
        """Network analysis of generator-function relationships"""
        
        # Create bipartite graph
        G = nx.Graph()
        
        # Add nodes
        generators = self.df['generator_name'].unique()
        functions = self.df['function_name'].unique()
        
        G.add_nodes_from(generators, bipartite=0, node_type='generator')
        G.add_nodes_from(functions, bipartite=1, node_type='function')
        
        # Add edges with weights
        edge_weights = self.df.groupby(['generator_name', 'function_name']).size()
        
        for (gen, func), weight in edge_weights.items():
            G.add_edge(gen, func, weight=weight)
        
        # Visualize network
        plt.figure(figsize=(15, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        gen_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'generator']
        func_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'function']
        
        nx.draw_networkx_nodes(G, pos, nodelist=gen_nodes, 
                              node_color='lightblue', node_size=1000, label='Generators')
        nx.draw_networkx_nodes(G, pos, nodelist=func_nodes, 
                              node_color='lightgreen', node_size=500, label='Functions')
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=np.array(weights)/10, alpha=0.5)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Generator-Function Relationship Network')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'relationship_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Compute network statistics
        network_stats = {
            'n_generators': len(gen_nodes),
            'n_functions': len(func_nodes),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
        }
        
        with open(self.output_dir / 'network_stats.json', 'w') as f:
            json.dump(network_stats, f, indent=2)
    
    def generate_report(self):
        """Generate comprehensive HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ODE Dataset Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .stats {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                          background-color: #e0e0e0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>ODE Dataset Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now()}</p>
            
            <h2>Dataset Overview</h2>
            <div class="stats">
                <div class="metric">Total ODEs: {len(self.df)}</div>
                <div class="metric">Verified: {self.df['verified'].sum()} ({100*self.df['verified'].mean():.1f}%)</div>
                <div class="metric">Generators: {self.df['generator_name'].nunique()}</div>
                <div class="metric">Functions: {self.df['function_name'].nunique()}</div>
                <div class="metric">Avg Complexity: {self.df['complexity_score'].mean():.1f}</div>
            </div>
            
            <h2>Generator Analysis</h2>
            <img src="generator_analysis.png" alt="Generator Analysis">
            
            <h2>Function Analysis</h2>
            <img src="function_analysis.png" alt="Function Analysis">
            
            <h2>Complexity Analysis</h2>
            <img src="complexity_analysis.png" alt="Complexity Analysis">
            
            <h2>Verification Analysis</h2>
            <img src="verification_analysis.png" alt="Verification Analysis">
            
            <h2>Pattern Analysis</h2>
            <img src="pattern_analysis.png" alt="Pattern Analysis">
            
            <h2>Relationship Network</h2>
            <img src="relationship_network.png" alt="Relationship Network">
            
            <h2>Key Findings</h2>
            <ul>
                <li>Most successful generator: {self.df.groupby('generator_name')['verified'].mean().idxmax()}</li>
                <li>Most complex generator: {self.df.groupby('generator_name')['complexity_score'].mean().idxmax()}</li>
                <li>Most used function: {self.df['function_name'].value_counts().index[0]}</li>
                <li>Complexity range: {self.df['complexity_score'].min():.0f} - {self.df['complexity_score'].max():.0f}</li>
            </ul>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'analysis_report.html', 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {self.output_dir / 'analysis_report.html'}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive ODE dataset analysis')
    parser.add_argument('dataset', help='Path to ODE dataset (JSONL)')
    parser.add_argument('--output-dir', default='analysis_output', help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveODEAnalyzer(args.dataset, args.output_dir)
    analyzer.analyze_all()

if __name__ == "__main__":
    main()