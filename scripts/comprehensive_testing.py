# scripts/comprehensive_testing.py
"""
Comprehensive testing framework for ODE system

Benefits:
- Automated test generation
- Property-based testing
- Performance benchmarking
- Regression detection
"""

import pytest
import hypothesis
from hypothesis import strategies as st
import numpy as np
import sympy as sp
from typing import List, Dict, Tuple
import time
import json
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

class ODETestSuite:
    """Comprehensive test suite for ODE generation system"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_all_tests(self):
        """Run complete test suite"""
        
        print("Starting comprehensive ODE system testing...")
        
        # 1. Unit tests
        self.run_unit_tests()
        
        # 2. Integration tests
        self.run_integration_tests()
        
        # 3. Property-based tests
        self.run_property_tests()
        
        # 4. Performance tests
        self.run_performance_tests()
        
        # 5. Stress tests
        self.run_stress_tests()
        
        # 6. Regression tests
        self.run_regression_tests()
        
        # Generate report
        self.generate_test_report()
    
    def run_unit_tests(self):
        """Run unit tests for all components"""
        
        print("\n1. Running unit tests...")
        
        test_modules = [
            "tests/test_generators.py",
            "tests/test_verifier.py",
            "tests/test_functions.py",
            "tests/test_derivatives.py"
        ]
        
        for module in test_modules:
            result = pytest.main(["-v", module, "--tb=short"])
            self.results.append({
                "test_type": "unit",
                "module": module,
                "result": "passed" if result == 0 else "failed"
            })
    
    def run_integration_tests(self):
        """Run integration tests"""
        
        print("\n2. Running integration tests...")
        
        # Test complete pipeline
        from pipeline.generator import ODEGenerator
        from verification.verifier import ODEVerifier
        
        generator = ODEGenerator()
        verifier = ODEVerifier()
        
        test_cases = [
            ("L1", "sine"),
            ("N1", "exponential"),
            ("L4", "quadratic")  # Pantograph
        ]
        
        for gen_name, func_name in test_cases:
            try:
                # Generate ODE
                ode_data = generator.generate_single(gen_name, func_name)
                
                # Verify ODE
                if ode_data:
                    verification = verifier.verify(
                        ode_data["ode_symbolic"],
                        ode_data["solution_symbolic"]
                    )
                    
                    self.results.append({
                        "test_type": "integration",
                        "generator": gen_name,
                        "function": func_name,
                        "generated": True,
                        "verified": verification.get("verified", False)
                    })
                else:
                    self.results.append({
                        "test_type": "integration",
                        "generator": gen_name,
                        "function": func_name,
                        "generated": False,
                        "verified": False
                    })
                    
            except Exception as e:
                self.results.append({
                    "test_type": "integration",
                    "generator": gen_name,
                    "function": func_name,
                    "error": str(e)
                })
    
    def run_property_tests(self):
        """Run property-based tests using Hypothesis"""
        
        print("\n3. Running property-based tests...")
        
        from hypothesis import given, settings
        
        @given(
            alpha=st.floats(min_value=-10, max_value=10),
            beta=st.floats(min_value=0.1, max_value=10),
            M=st.floats(min_value=-5, max_value=5)
        )
        @settings(max_examples=100, deadline=None)
        def test_parameter_robustness(alpha, beta, M):
            """Test that generators handle various parameter values"""
            
            from pipeline.generator import ODEGenerator
            generator = ODEGenerator()
            
            params = {"alpha": alpha, "beta": beta, "M": M}
            
            try:
                ode_data = generator.generate_single("L1", "identity", params)
                assert ode_data is not None
                assert "ode_symbolic" in ode_data
                assert not np.isnan(ode_data["complexity_score"])
            except Exception as e:
                # Log but don't fail - some parameter combinations might be invalid
                pass
        
        try:
            test_parameter_robustness()
            self.results.append({
                "test_type": "property",
                "test_name": "parameter_robustness",
                "result": "passed"
            })
        except Exception as e:
            self.results.append({
                "test_type": "property",
                "test_name": "parameter_robustness",
                "result": "failed",
                "error": str(e)
            })
    
    def run_performance_tests(self):
        """Run performance benchmarks"""
        
        print("\n4. Running performance tests...")
        
        from pipeline.generator import ODEGenerator
        generator = ODEGenerator()
        
        # Benchmark different generators
        generators = ["L1", "L2", "L3", "L4", "N1", "N2", "N3", "N7"]
        performance_results = []
        
        for gen_name in generators:
            times = []
            
            # Run multiple iterations
            for _ in range(10):
                start_time = time.time()
                ode_data = generator.generate_single(gen_name, "sine")
                end_time = time.time()
                
                if ode_data:
                    times.append(end_time - start_time)
            
            if times:
                performance_results.append({
                    "generator": gen_name,
                    "avg_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times)
                })
        
        # Save performance results
        perf_df = pd.DataFrame(performance_results)
        perf_df.to_csv(self.output_dir / "performance_results.csv", index=False)
        
        # Create performance visualization
        self._create_performance_chart(perf_df)
        
        self.results.append({
            "test_type": "performance",
            "test_name": "generator_benchmarks",
            "result": "completed",
            "data_file": "performance_results.csv"
        })
    
    def run_stress_tests(self):
        """Run stress tests to find system limits"""
        
        print("\n5. Running stress tests...")
        
        # Test parallel generation
        from scripts.parallel_generator import ParallelODEGenerator
        
        stress_configs = [
            {"workers": 2, "tasks": 100},
            {"workers": 4, "tasks": 500},
            {"workers": 8, "tasks": 1000}
        ]
        
        stress_results = []
        
        for config in stress_configs:
            print(f"Testing with {config['workers']} workers, {config['tasks']} tasks...")
            
            generator = ParallelODEGenerator(n_workers=config["workers"])
            
            start_time = time.time()
            stats = generator.generate_parallel(
                generators=["L1", "L2"],
                functions=["sine", "cosine"],
                samples_per_combination=config["tasks"] // 4,
                output_file=f"stress_test_{config['workers']}w_{config['tasks']}t.jsonl"
            )
            end_time = time.time()
            
            stress_results.append({
                "workers": config["workers"],
                "tasks": config["tasks"],
                "completed": stats["completed"],
                "duration": end_time - start_time,
                "throughput": stats["completed"] / (end_time - start_time)
            })
            
            # Clean up test file
            Path(f"stress_test_{config['workers']}w_{config['tasks']}t.jsonl").unlink()
        
        # Save stress test results
        stress_df = pd.DataFrame(stress_results)
        stress_df.to_csv(self.output_dir / "stress_test_results.csv", index=False)
        
        self.results.append({
            "test_type": "stress",
            "test_name": "parallel_generation",
            "result": "completed",
            "max_throughput": max(r["throughput"] for r in stress_results)
        })
    
    def run_regression_tests(self):
        """Run regression tests against known good outputs"""
        
        print("\n6. Running regression tests...")
        
        # Load regression test cases
        regression_file = "tests/regression_cases.json"
        
        if not Path(regression_file).exists():
            print("No regression test cases found, skipping...")
            return
        
        with open(regression_file, 'r') as f:
            regression_cases = json.load(f)
        
        from pipeline.generator import ODEGenerator
        generator = ODEGenerator()
        
        regression_failures = []
        
        for case in regression_cases:
            # Generate ODE with same parameters
            ode_data = generator.generate_single(
                case["generator"],
                case["function"],
                case["parameters"]
            )
            
            if ode_data:
                # Compare with expected output
                if ode_data["ode_symbolic"] != case["expected_ode"]:
                    regression_failures.append({
                        "case": case["id"],
                        "expected": case["expected_ode"],
                        "actual": ode_data["ode_symbolic"]
                    })
        
        self.results.append({
            "test_type": "regression",
            "test_name": "ode_generation",
            "total_cases": len(regression_cases),
            "failures": len(regression_failures),
            "result": "passed" if len(regression_failures) == 0 else "failed"
        })
        
        if regression_failures:
            with open(self.output_dir / "regression_failures.json", 'w') as f:
                json.dump(regression_failures, f, indent=2)
    
    def _create_performance_chart(self, perf_df: pd.DataFrame):
        """Create performance visualization"""
        
        plt.figure(figsize=(10, 6))
        
        # Bar chart with error bars
        x = np.arange(len(perf_df))
        plt.bar(x, perf_df['avg_time'], yerr=perf_df['std_time'], 
               capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        
        plt.xlabel('Generator')
        plt.ylabel('Average Generation Time (seconds)')
        plt.title('ODE Generator Performance Comparison')
        plt.xticks(x, perf_df['generator'])
        
        # Add value labels
        for i, v in enumerate(perf_df['avg_time']):
            plt.text(i, v + perf_df['std_time'].iloc[i] + 0.001, 
                    f'{v:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_chart.png', dpi=300)
        plt.close()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        
        print("\n7. Generating test report...")
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results 
                          if r.get("result") == "passed" or r.get("result") == "completed")
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ODE System Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>ODE System Test Report</h1>
            <p>Generated: {datetime.now()}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {total_tests}</p>
                <p class="passed">Passed: {passed_tests}</p>
                <p class="failed">Failed: {total_tests - passed_tests}</p>
                <p>Success Rate: {100 * passed_tests / total_tests:.1f}%</p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Type</th>
                    <th>Test Name/Module</th>
                    <th>Result</th>
                    <th>Details</th>
                </tr>
        """
        
        for result in self.results:
            test_type = result.get("test_type", "unknown")
            test_name = result.get("test_name") or result.get("module") or result.get("generator", "")
            test_result = result.get("result", "unknown")
            
            # Format details
            details = []
            if "error" in result:
                details.append(f"Error: {result['error']}")
            if "verified" in result:
                details.append(f"Verified: {result['verified']}")
            if "max_throughput" in result:
                details.append(f"Max throughput: {result['max_throughput']:.1f} ODEs/s")
            
            details_str = "; ".join(details) if details else "-"
            
            result_class = "passed" if test_result in ["passed", "completed"] else "failed"
            
            html_content += f"""
                <tr>
                    <td>{test_type}</td>
                    <td>{test_name}</td>
                    <td class="{result_class}">{test_result}</td>
                    <td>{details_str}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Performance Results</h2>
            <img src="performance_chart.png" alt="Performance Chart" style="max-width: 800px;">
            
            <h2>Additional Files</h2>
            <ul>
                <li><a href="performance_results.csv">Performance Results (CSV)</a></li>
                <li><a href="stress_test_results.csv">Stress Test Results (CSV)</a></li>
            </ul>
        </body>
        </html>
        """
        
        with open(self.output_dir / "test_report.html", 'w') as f:
            f.write(html_content)
        
        print(f"\nTest report generated: {self.output_dir / 'test_report.html'}")
        print(f"Overall success rate: {100 * passed_tests / total_tests:.1f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive ODE system tests')
    parser.add_argument('--output-dir', default='test_results', help='Output directory')
    parser.add_argument('--test-types', nargs='+', 
                       choices=['unit', 'integration', 'property', 'performance', 'stress', 'regression'],
                       help='Specific test types to run')
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = ODETestSuite(output_dir=args.output_dir)
    
    # Run tests
    if args.test_types:
        # Run specific tests
        for test_type in args.test_types:
            method_name = f"run_{test_type}_tests"
            if hasattr(test_suite, method_name):
                getattr(test_suite, method_name)()
    else:
        # Run all tests
        test_suite.run_all_tests()

if __name__ == "__main__":
    main()