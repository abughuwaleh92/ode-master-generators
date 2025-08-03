#!/usr/bin/env python
"""
ODE Master Generators - Main Entry Point
"""

import logging
import argparse
from pathlib import Path
import time
from datetime import datetime
import json
import sys
import os
import locale

# Fix encoding issues on Windows
if sys.platform == 'win32':
    # Set UTF-8 as default encoding
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import from the packages
from utils.config import ConfigManager
from utils.features import FeatureExtractor
from pipeline.generator import ODEDatasetGenerator
from pipeline.parallel import ParallelODEGenerator

# Setup logging
def setup_logging(config: ConfigManager):
    """Configure logging based on config"""
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.file', 'ode_generation.log')
    
    # Create logs directory if needed
    log_dir = Path(log_file).parent
    if log_dir.name and log_dir != Path('.'):
        log_dir.mkdir(exist_ok=True)
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("ODE Master Generators - Production System")
    logger.info("="*70)
    
    return logger

def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='ODE Master Generators System')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Use parallel generation'
    )
    parser.add_argument(
        '--samples', 
        type=int, 
        default=None,
        help='Samples per combination (overrides config)'
    )
    parser.add_argument(
        '--extract-features', 
        action='store_true',
        help='Extract features after generation'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    logger = setup_logging(config)
    
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Parallel mode: {'Enabled' if args.parallel else 'Disabled'}")
    
    # Override samples if provided
    if args.samples:
        config.config['generation']['samples_per_combo'] = args.samples
        logger.info(f"Samples per combination: {args.samples}")
    
    # Start generation
    start_time = time.time()
    
    try:
        if args.parallel:
            # Parallel generation
            logger.info("Using parallel generation")
            
            # First test generators
            test_generator = ODEDatasetGenerator(config)
            working_generators = test_generator.test_generators()
            
            # Create parallel generator
            parallel_gen = ParallelODEGenerator(config)
            
            # Create tasks
            tasks = parallel_gen.create_generation_tasks(
                working_generators,
                list(test_generator.f_library.keys()),
                config.get('generation.samples_per_combo', 5)
            )
            
            logger.info(f"Created {len(tasks)} generation tasks")
            
            # Generate in parallel
            dataset = parallel_gen.generate_batch_parallel(tasks)
            
            # Save results
            output_file = config.get('output.streaming_file')
            with open(output_file, 'w') as f:
                for ode in dataset:
                    f.write(json.dumps(ode.to_dict(), default=str) + '\n')
            
            logger.info(f"Results saved to {output_file}")
            
        else:
            # Sequential generation
            generator = ODEDatasetGenerator(config)
            dataset = generator.generate_dataset(args.samples)
            
            # Save report
            generator.save_report()
        
        # Extract features if requested
        if args.extract_features and dataset:
            logger.info("Extracting features...")
            
            extractor = FeatureExtractor()
            features_df = extractor.extract_features(dataset)
            
            features_file = config.get('output.features_file')
            extractor.save_features(features_df, features_file)
            
            logger.info(f"Features saved to {features_file}")
            logger.info(f"Feature shape: {features_df.shape}")
            
            # Print feature summary
            logger.info("\nFeature Summary:")
            logger.info(f"Total ODEs: {len(features_df)}")
            logger.info(f"Verified ODEs: {features_df['verified'].sum()}")
            logger.info(f"Linear ODEs: {(features_df['generator_type'] == 'linear').sum()}")
            logger.info(f"Nonlinear ODEs: {(features_df['generator_type'] == 'nonlinear').sum()}")
            logger.info(f"Pantograph equations: {features_df['has_pantograph'].sum()}")
            
            # Complexity distribution
            logger.info("\nComplexity Distribution:")
            logger.info(f"Mean complexity: {features_df['complexity_score'].mean():.1f}")
            logger.info(f"Max complexity: {features_df['complexity_score'].max()}")
            logger.info(f"High complexity (>100): {(features_df['complexity_score'] > 100).sum()}")
        
        # Final statistics
        elapsed_time = time.time() - start_time
        logger.info(f"\n✅ Total execution time: {elapsed_time:.1f} seconds")
        logger.info(f"✅ ODEs generated: {len(dataset) if 'dataset' in locals() else 0}")
        
        if 'dataset' in locals() and dataset:
            verified_count = sum(1 for ode in dataset if ode.verified)
            logger.info(f"✅ Verification rate: {100*verified_count/len(dataset):.1f}%")
        
        logger.info("\n🎉 ODE Master Generators completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("="*70)

if __name__ == "__main__":
    main()