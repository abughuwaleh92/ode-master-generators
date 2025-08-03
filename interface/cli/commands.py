import click
import json
import sys
from pathlib import Path
from tabulate import tabulate
import pandas as pd
from typing import Optional

from utils.config import ConfigManager
from pipeline.generator import ODEDatasetGenerator
from utils.features import FeatureExtractor
from core.functions import AnalyticFunctionLibrary

@click.group()
@click.version_option(version='2.0.0')
def cli():
    """ODE Master Generators Command Line Interface"""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--samples', '-s', type=int, default=5,
              help='Samples per combination')
@click.option('--generators', '-g', multiple=True,
              help='Specific generators to use (e.g., -g L1 -g N1)')
@click.option('--functions', '-f', multiple=True,
              help='Specific functions to use')
@click.option('--output', '-o', type=click.Path(),
              default='ode_dataset.jsonl',
              help='Output file path')
@click.option('--parallel/--no-parallel', default=False,
              help='Use parallel generation')
@click.option('--extract-features/--no-extract-features', default=False,
              help='Extract features after generation')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def generate(config, samples, generators, functions, output, 
             parallel, extract_features, verbose):
    """Generate ODE dataset"""
    
    click.echo(click.style('ODE Master Generators', fg='blue', bold=True))
    click.echo('=' * 50)
    
    # Load configuration
    config_mgr = ConfigManager(config) if config else ConfigManager()
    
    # Override with CLI options
    if samples:
        config_mgr.config['generation']['samples_per_combo'] = samples
    
    # Create generator
    generator = ODEDatasetGenerator(config_mgr)
    
    # Filter generators if specified
    if generators:
        linear_gens = [g for g in generators if g.startswith('L')]
        nonlinear_gens = [g for g in generators if g.startswith('N')]
        
        generator.linear_generators = {
            k: v for k, v in generator.linear_generators.items()
            if k in linear_gens
        }
        generator.nonlinear_generators = {
            k: v for k, v in generator.nonlinear_generators.items()
            if k in nonlinear_gens
        }
    
    # Filter functions if specified
    if functions:
        available_functions = list(generator.f_library.keys())
        selected_functions = [f for f in functions if f in available_functions]
        generator.f_library = {
            k: v for k, v in generator.f_library.items()
            if k in selected_functions
        }
    
    # Show configuration
    if verbose:
        click.echo('\nConfiguration:')
        click.echo(f'  Samples per combo: {samples}')
        click.echo(f'  Linear generators: {list(generator.linear_generators.keys())}')
        click.echo(f'  Nonlinear generators: {list(generator.nonlinear_generators.keys())}')
        click.echo(f'  Functions: {len(generator.f_library)}')
        click.echo(f'  Output: {output}')
        click.echo(f'  Parallel: {parallel}')
        click.echo()
    
    # Generate dataset
    with click.progressbar(label='Generating ODEs') as bar:
        # Custom progress callback
        def progress_callback(current, total):
            bar.update(current - bar.pos)
        
        generator._log_progress = progress_callback
        
        if parallel:
            # Use parallel generation
            from pipeline.parallel import ParallelODEGenerator
            
            # Test generators first
            working_generators = generator.test_generators()
            
            parallel_gen = ParallelODEGenerator(config_mgr)
            tasks = parallel_gen.create_generation_tasks(
                working_generators,
                list(generator.f_library.keys()),
                samples
            )
            
            dataset = parallel_gen.generate_batch_parallel(tasks)
        else:
            dataset = generator.generate_dataset(samples)
    
    # Save results
    click.echo(f'\nSaving results to {output}...')
    
    with open(output, 'w') as f:
        for ode in dataset:
            f.write(json.dumps(ode.to_dict(), default=str) + '\n')
    
    # Generate report
    generator.save_report()
    
    # Extract features if requested
    if extract_features:
        click.echo('\nExtracting features...')
        extractor = FeatureExtractor()
        features_df = extractor.extract_features(dataset)
        
        features_file = Path(output).with_suffix('.parquet')
        extractor.save_features(features_df, str(features_file))
        
        click.echo(f'Features saved to {features_file}')
    
    # Show summary
    click.echo('\n' + click.style('Summary:', fg='green', bold=True))
    click.echo(f'  Total ODEs generated: {len(dataset)}')
    click.echo(f'  Verified: {sum(1 for ode in dataset if ode.verified)}')
    click.echo(f'  Verification rate: {100*sum(1 for ode in dataset if ode.verified)/len(dataset):.1f}%')
    
    if verbose and dataset:
        # Show sample ODEs
        click.echo('\nSample ODEs:')
        for i, ode in enumerate(dataset[:3]):
            click.echo(f'\n{i+1}. {ode.generator_name} with {ode.function_name}:')
            click.echo(f'   ODE: {ode.ode_symbolic[:80]}...')
            click.echo(f'   Verified: {"✓" if ode.verified else "✗"}')

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['json', 'jsonl', 'parquet']),
              help='Input file format')
@click.option('--limit', '-l', type=int, default=10,
              help='Number of ODEs to display')
@click.option('--filter-verified/--all', default=False,
              help='Show only verified ODEs')
@click.option('--generator', '-g', help='Filter by generator name')
@click.option('--function', '-fn', help='Filter by function name')
def view(file, format, limit, filter_verified, generator, function):
    """View ODE dataset"""
    
    # Load data based on format
    if format == 'parquet' or file.endswith('.parquet'):
        df = pd.read_parquet(file)
        data = df.to_dict('records')
    elif format == 'jsonl' or file.endswith('.jsonl'):
        data = []
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(file, 'r') as f:
            data = json.load(f)
    
    # Apply filters
    if filter_verified:
        data = [d for d in data if d.get('verified', False)]
    
    if generator:
        data = [d for d in data if d.get('generator_name') == generator]
    
    if function:
        data = [d for d in data if d.get('function_name') == function]
    
    # Display data
    if not data:
        click.echo('No ODEs found matching filters')
        return
    
    click.echo(f'Showing {min(limit, len(data))} of {len(data)} ODEs:\n')
    
    for i, ode in enumerate(data[:limit]):
        click.echo(click.style(f'ODE #{i+1}', fg='blue', bold=True))
        click.echo(f'Generator: {ode.get("generator_name")} ({ode.get("generator_type")})')
        click.echo(f'Function: {ode.get("function_name")}')
        click.echo(f'Verified: {"✓" if ode.get("verified") else "✗"} ({ode.get("verification_method", "N/A")})')
        click.echo(f'Complexity: {ode.get("complexity_score", "N/A")}')
        
        if 'ode_symbolic' in ode:
            ode_str = ode['ode_symbolic']
            if len(ode_str) > 100:
                ode_str = ode_str[:97] + '...'
            click.echo(f'ODE: {ode_str}')
        
        if 'initial_conditions' in ode:
            click.echo(f'Initial conditions: {ode["initial_conditions"]}')
        
        click.echo()

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file for report')
@click.option('--format', '-f', 
              type=click.Choice(['text', 'json', 'html']),
              default='text',
              help='Report format')
def analyze(file, output, format):
    """Analyze ODE dataset"""
    
    click.echo('Analyzing ODE dataset...\n')
    
    # Load data
    if file.endswith('.parquet'):
        df = pd.read_parquet(file)
    elif file.endswith('.jsonl'):
        data = []
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        with open(file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    
    # Basic statistics
    stats = {
        'Total ODEs': len(df),
        'Verified': df['verified'].sum() if 'verified' in df else 0,
        'Verification Rate': f"{100*df['verified'].mean():.1f}%" if 'verified' in df else 'N/A',
        'Linear ODEs': len(df[df['generator_type'] == 'linear']) if 'generator_type' in df else 0,
        'Nonlinear ODEs': len(df[df['generator_type'] == 'nonlinear']) if 'generator_type' in df else 0,
        'Pantograph': df['has_pantograph'].sum() if 'has_pantograph' in df else 0
    }
    
    # Generator performance
    if 'generator_name' in df and 'verified' in df:
        gen_perf = df.groupby('generator_name')['verified'].agg(['count', 'sum', 'mean'])
        gen_perf.columns = ['Total', 'Verified', 'Rate']
        gen_perf['Rate'] = (gen_perf['Rate'] * 100).round(1)
    else:
        gen_perf = None
    
    # Complexity distribution
    if 'complexity_score' in df:
        complexity_stats = {
            'Mean': df['complexity_score'].mean(),
            'Std': df['complexity_score'].std(),
            'Min': df['complexity_score'].min(),
            'Max': df['complexity_score'].max()
        }
    else:
        complexity_stats = None
    
    # Format output
    if format == 'text':
        click.echo(click.style('Dataset Statistics:', fg='green', bold=True))
        for key, value in stats.items():
            click.echo(f'  {key}: {value}')
        
        if gen_perf is not None:
            click.echo('\n' + click.style('Generator Performance:', fg='green', bold=True))
            click.echo(tabulate(gen_perf, headers=gen_perf.columns, tablefmt='grid'))
        
        if complexity_stats:
            click.echo('\n' + click.style('Complexity Statistics:', fg='green', bold=True))
            for key, value in complexity_stats.items():
                click.echo(f'  {key}: {value:.2f}')
    
    elif format == 'json':
        report = {
            'statistics': stats,
            'generator_performance': gen_perf.to_dict() if gen_perf is not None else None,
            'complexity_statistics': complexity_stats
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f'Report saved to {output}')
        else:
            click.echo(json.dumps(report, indent=2))
    
    elif format == 'html':
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ODE Dataset Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .stat {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>ODE Dataset Analysis</h1>
            
            <h2>Dataset Statistics</h2>
            {"".join(f'<div class="stat"><strong>{k}:</strong> {v}</div>' for k, v in stats.items())}
            
            {f'<h2>Generator Performance</h2>{gen_perf.to_html()}' if gen_perf is not None else ''}
            
            {f'<h2>Complexity Statistics</h2>{"".join(f"<div class=\"stat\"><strong>{k}:</strong> {v:.2f}</div>" for k, v in complexity_stats.items())}' if complexity_stats else ''}
        </body>
        </html>
        """
        
        if output:
            with open(output, 'w') as f:
                f.write(html_content)
            click.echo(f'HTML report saved to {output}')
        else:
            click.echo('Please specify output file for HTML format')

@cli.command()
def list_generators():
    """List available generators"""
    
    click.echo(click.style('Available Generators:', fg='blue', bold=True))
    click.echo('\nLinear Generators:')
    
    generators = {
        'L1': 'y\'\'(x) + y(x) = RHS',
        'L2': 'y\'\'(x) + y\'(x) = RHS', 
        'L3': 'y(x) + y\'(x) = RHS',
        'L4': 'y\'\'(x) + y(x/a) - y(x) = RHS (Pantograph)'
    }
    
    for name, eq in generators.items():
        click.echo(f'  {name}: {eq}')
    
    click.echo('\nNonlinear Generators:')
    
    nonlinear = {
        'N1': '(y\'\'(x))^q + y(x) = RHS',
        'N2': '(y\'\'(x))^q + (y\'(x))^v = RHS',
        'N3': 'y(x) + (y\'(x))^v = RHS'
    }
    
    for name, eq in nonlinear.items():
        click.echo(f'  {name}: {eq}')

@cli.command()
def list_functions():
    """List available analytic functions"""
    
    functions = AnalyticFunctionLibrary.get_safe_library()
    
    click.echo(click.style('Available Functions:', fg='blue', bold=True))
    
    categories = {
        'Polynomial': ['identity', 'quadratic', 'cubic', 'quartic', 'quintic'],
        'Exponential': ['exponential', 'exp_scaled', 'exp_quadratic', 'exp_negative'],
        'Trigonometric': ['sine', 'cosine', 'tangent_safe', 'sine_scaled', 'cosine_scaled'],
        'Hyperbolic': ['sinh', 'cosh', 'tanh', 'sinh_scaled', 'cosh_scaled'],
        'Logarithmic': ['log_safe', 'log_shifted', 'log_scaled'],
        'Rational': ['rational_simple', 'rational_stable', 'rational_cubic'],
        'Composite': ['exp_sin', 'sin_exp', 'gaussian', 'bessel_like']
    }
    
    for category, func_list in categories.items():
        click.echo(f'\n{category}:')
        for func in func_list:
            if func in functions:
                click.echo(f'  - {func}')

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--from-format', '-f', 
              type=click.Choice(['json', 'jsonl', 'parquet']),
              help='Input format')
@click.option('--to-format', '-t',
              type=click.Choice(['json', 'jsonl', 'parquet', 'csv']),
              required=True,
              help='Output format')
def convert(input_file, output_file, from_format, to_format):
    """Convert dataset between formats"""
    
    click.echo(f'Converting {input_file} to {output_file}...')
    
    # Detect input format if not specified
    if not from_format:
        if input_file.endswith('.jsonl'):
            from_format = 'jsonl'
        elif input_file.endswith('.parquet'):
            from_format = 'parquet'
        else:
            from_format = 'json'
    
    # Load data
    if from_format == 'parquet':
        df = pd.read_parquet(input_file)
        data = df.to_dict('records')
    elif from_format == 'jsonl':
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(input_file, 'r') as f:
            data = json.load(f)
    
    # Save in new format
    if to_format == 'json':
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    elif to_format == 'jsonl':
        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item, default=str) + '\n')
    
    elif to_format == 'parquet':
        df = pd.DataFrame(data)
        df.to_parquet(output_file)
    
    elif to_format == 'csv':
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
    
    click.echo(f'Successfully converted {len(data)} records')

if __name__ == '__main__':
    cli()