#!/usr/bin/env python
"""Monitor ODE generation progress in real-time"""

import time
import json
import os
from pathlib import Path
from datetime import datetime
import sys

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_time(seconds):
    """Format seconds to readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def monitor_generation(dataset_file='ode_dataset.jsonl', report_file='generation_report.json'):
    """Monitor ODE generation progress"""
    
    print("Monitoring ODE Generation Progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    start_time = datetime.now()
    last_count = 0
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("="*60)
            print("ODE GENERATION MONITOR")
            print("="*60)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Elapsed: {format_time((datetime.now() - start_time).total_seconds())}")
            print()
            
            # Check dataset file
            if Path(dataset_file).exists():
                # Count ODEs
                with open(dataset_file, 'r') as f:
                    ode_count = sum(1 for line in f if line.strip())
                
                # Read last ODE
                if ode_count > 0:
                    with open(dataset_file, 'r') as f:
                        lines = f.readlines()
                        last_ode = json.loads(lines[-1])
                    
                    # Display statistics
                    print(f"Total ODEs Generated: {ode_count}/272")
                    progress = 100 * ode_count / 272
                    print(f"Progress: {progress:.1f}%")
                    print(f"[{'#' * int(progress/2)}{'-' * (50-int(progress/2))}]")
                    print()
                    
                    # Generation rate
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 0:
                        rate = ode_count / elapsed
                        print(f"Generation Rate: {rate:.1f} ODEs/s")
                        
                        # ETA
                        if rate > 0:
                            remaining = 272 - ode_count
                            eta = remaining / rate
                            print(f"Estimated Time Remaining: {format_time(eta)}")
                    
                    print()
                    print("Last Generated ODE:")
                    print(f"  Generator: {last_ode.get('generator_name', 'N/A')}")
                    print(f"  Function: {last_ode.get('function_name', 'N/A')}")
                    print(f"  Verified: {'✓' if last_ode.get('verified') else '✗'}")
                    print(f"  Complexity: {last_ode.get('complexity_score', 'N/A')}")
                    
                    # Count by generator
                    print("\nODEs by Generator:")
                    generators_count = {}
                    verified_count = 0
                    
                    with open(dataset_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                ode = json.loads(line)
                                gen = ode.get('generator_name', 'Unknown')
                                generators_count[gen] = generators_count.get(gen, 0) + 1
                                if ode.get('verified'):
                                    verified_count += 1
                    
                    for gen, count in sorted(generators_count.items()):
                        print(f"  {gen}: {count}")
                    
                    print(f"\nVerification Rate: {100*verified_count/ode_count:.1f}%")
                    
                    # Check if generation increased
                    if ode_count > last_count:
                        print(f"\n✓ Generated {ode_count - last_count} new ODEs")
                        last_count = ode_count
                    
                    # Check if complete
                    if ode_count >= 272:
                        print("\n🎉 GENERATION COMPLETE!")
                        
                        # Check for report
                        if Path(report_file).exists():
                            with open(report_file, 'r') as f:
                                report = json.load(f)
                            
                            print("\nFinal Statistics:")
                            gen_info = report.get('generation_info', {})
                            print(f"  Total Time: {gen_info.get('duration_seconds', 0):.1f}s")
                            print(f"  Success Rate: {100*gen_info.get('total_generated', 0)/272:.1f}%")
                            
                            break
                else:
                    print("Waiting for ODEs to be generated...")
            else:
                print(f"Waiting for {dataset_file} to be created...")
            
            # Refresh every 2 seconds
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print(f"Final count: {last_count} ODEs")
    except Exception as e:
        print(f"\nError: {e}")

def quick_stats(dataset_file='ode_dataset.jsonl'):
    """Quick statistics of the dataset"""
    if not Path(dataset_file).exists():
        print("Dataset file not found!")
        return
    
    total = 0
    verified = 0
    by_generator = {}
    by_type = {'linear': 0, 'nonlinear': 0}
    
    with open(dataset_file, 'r') as f:
        for line in f:
            if line.strip():
                ode = json.loads(line)
                total += 1
                
                if ode.get('verified'):
                    verified += 1
                
                gen = ode.get('generator_name', 'Unknown')
                by_generator[gen] = by_generator.get(gen, 0) + 1
                
                gen_type = ode.get('generator_type', 'unknown')
                if gen_type in by_type:
                    by_type[gen_type] += 1
    
    print(f"\nDataset Statistics:")
    print(f"Total ODEs: {total}")
    print(f"Verified: {verified} ({100*verified/total:.1f}%)" if total > 0 else "Verified: 0")
    print(f"\nBy Type:")
    for t, count in by_type.items():
        print(f"  {t}: {count}")
    print(f"\nBy Generator:")
    for gen, count in sorted(by_generator.items()):
        print(f"  {gen}: {count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor ODE generation')
    parser.add_argument('--stats', action='store_true', help='Show quick statistics')
    parser.add_argument('--file', default='ode_dataset.jsonl', help='Dataset file to monitor')
    
    args = parser.parse_args()
    
    if args.stats:
        quick_stats(args.file)
    else:
        monitor_generation(args.file)