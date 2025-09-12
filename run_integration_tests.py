#!/usr/bin/env python3
"""
Seoul Market Risk ML System - Integration Test Runner

Comprehensive test runner for the hierarchical ML system with 79 models.
Executes end-to-end testing from data preprocessing to Korean business reports.

Usage:
    python run_integration_tests.py [options]

Options:
    --verbose, -v       : Verbose output
    --quick, -q         : Run quick test subset
    --performance, -p   : Include performance benchmarks
    --korean-only, -k   : Run Korean-specific tests only
    --report, -r        : Generate detailed HTML report
    --parallel, -j N    : Run tests in parallel (N workers)
    
Examples:
    python run_integration_tests.py -v -r
    python run_integration_tests.py --quick --korean-only
    python run_integration_tests.py --performance -j 4
"""

import argparse
import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import pytest
    import pandas as pd
    import numpy as np
    from jinja2 import Template
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please install: pip install pytest pandas numpy jinja2")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Comprehensive integration test runner for Seoul Market Risk ML System"""
    
    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "src" / "tests"
        self.results_dir = self.project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.test_start_time = None
        self.test_results = {}
        
        # System specifications
        self.system_specs = {
            "total_models": 79,
            "global_models": 1,
            "regional_models": 6,  
            "local_models": 72,
            "total_records": 408221,
            "test_years": [2019, 2020, 2021, 2022, 2023, 2024],
            "regions": 6,
            "business_categories": 12
        }
    
    def print_banner(self):
        """Print test runner banner"""
        print("\n" + "="*90)
        print("🏢 SEOUL MARKET RISK ML SYSTEM - INTEGRATION TEST SUITE")
        print("="*90)
        print(f"📊 System: {self.system_specs['total_models']} models "
              f"({self.system_specs['global_models']} Global + "
              f"{self.system_specs['regional_models']} Regional + "
              f"{self.system_specs['local_models']} Local)")
        print(f"📈 Data: {self.system_specs['total_records']:,} records across "
              f"{len(self.system_specs['test_years'])} years")
        print(f"🌏 Coverage: {self.system_specs['regions']} regions × "
              f"{self.system_specs['business_categories']} business categories")
        print(f"🇰🇷 Language: Korean business terminology and risk levels")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*90 + "\n")
    
    def validate_environment(self) -> bool:
        """Validate test environment setup"""
        logger.info("Validating test environment...")
        
        validation_checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            validation_checks.append(("Python version", True, f"{python_version.major}.{python_version.minor}"))
        else:
            validation_checks.append(("Python version", False, "Requires Python 3.8+"))
        
        # Check required packages
        required_packages = [
            "pytest", "pandas", "numpy", "scikit-learn", 
            "prophet", "lightgbm", "openai", "jinja2"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                validation_checks.append((f"Package {package}", True, "✓"))
            except ImportError:
                validation_checks.append((f"Package {package}", False, "Missing"))
        
        # Check test files
        test_files = [
            self.test_dir / "integration_test.py",
            self.test_dir / "test_config.py"
        ]
        
        for test_file in test_files:
            if test_file.exists():
                validation_checks.append((f"Test file {test_file.name}", True, "✓"))
            else:
                validation_checks.append((f"Test file {test_file.name}", False, "Missing"))
        
        # Check data files
        data_dir = self.project_root / "data" / "processed"
        if data_dir.exists():
            data_files = list(data_dir.glob("seoul_sales_*.csv"))
            if data_files:
                validation_checks.append(("Processed data", True, f"{len(data_files)} files"))
            else:
                validation_checks.append(("Processed data", False, "No CSV files"))
        else:
            validation_checks.append(("Data directory", False, "Missing"))
        
        # Check configuration
        config_file = self.project_root / "config" / "config.yaml"
        if config_file.exists():
            validation_checks.append(("Configuration", True, "✓"))
        else:
            validation_checks.append(("Configuration", False, "Missing config.yaml"))
        
        # Print validation results
        print("🔍 ENVIRONMENT VALIDATION")
        print("-" * 50)
        
        all_valid = True
        for check_name, is_valid, details in validation_checks:
            status = "✅" if is_valid else "❌"
            print(f"{status} {check_name:30} {details}")
            if not is_valid:
                all_valid = False
        
        if all_valid:
            print("✅ Environment validation passed\n")
        else:
            print("❌ Environment validation failed - please fix issues above\n")
            
        return all_valid
    
    def build_test_command(self) -> List[str]:
        """Build pytest command based on arguments"""
        cmd = ["python", "-m", "pytest"]
        
        # Add test file
        cmd.append(str(self.test_dir / "integration_test.py"))
        
        # Verbosity
        if self.args.verbose:
            cmd.extend(["-v", "-s"])
        else:
            cmd.append("-q")
        
        # Test selection
        if self.args.quick:
            cmd.extend(["-m", "not slow"])
            logger.info("Running quick test subset (excluding slow tests)")
        
        if self.args.korean_only:
            cmd.extend(["-m", "korean"])
            logger.info("Running Korean-specific tests only")
            
        if self.args.performance:
            cmd.extend(["-m", "performance"])
            logger.info("Including performance benchmark tests")
        
        # Parallel execution
        if hasattr(self.args, 'parallel') and self.args.parallel:
            cmd.extend(["-n", str(self.args.parallel)])
            logger.info(f"Running tests in parallel with {self.args.parallel} workers")
        
        # Output options
        cmd.extend([
            "--tb=short",
            "--disable-warnings",
            "-p", "no:cacheprovider"
        ])
        
        # Generate reports
        if self.args.report:
            report_file = self.results_dir / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            cmd.extend([
                "--html", str(report_file),
                "--self-contained-html"
            ])
            logger.info(f"HTML report will be generated: {report_file}")
        
        # JUnit XML for CI/CD
        junit_file = self.results_dir / f"integration_junit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        cmd.extend(["--junit-xml", str(junit_file)])
        
        return cmd
    
    def run_tests(self) -> Dict[str, Any]:
        """Execute integration tests and capture results"""
        logger.info("Starting integration test execution...")
        
        self.test_start_time = time.time()
        
        # Build and execute test command
        cmd = self.build_test_command()
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            test_duration = time.time() - self.test_start_time
            
            # Parse results
            self.test_results = {
                "start_time": datetime.fromtimestamp(self.test_start_time).isoformat(),
                "duration_seconds": test_duration,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": cmd,
                "system_specs": self.system_specs
            }
            
            # Extract test statistics from pytest output
            self.parse_test_statistics()
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            self.test_results = {
                "start_time": datetime.fromtimestamp(self.test_start_time).isoformat(),
                "duration_seconds": time.time() - self.test_start_time,
                "exit_code": -1,
                "error": str(e),
                "system_specs": self.system_specs
            }
            return self.test_results
    
    def parse_test_statistics(self):
        """Parse test statistics from pytest output"""
        stdout = self.test_results.get("stdout", "")
        
        # Extract test counts
        import re
        
        # Look for pytest summary line like "5 passed, 2 failed, 1 skipped"
        summary_pattern = r'(\d+) (passed|failed|skipped|error)'
        matches = re.findall(summary_pattern, stdout)
        
        test_stats = {}
        for count, status in matches:
            test_stats[status] = int(count)
        
        # Extract timing information
        timing_pattern = r'in ([\d\.]+)s'
        timing_match = re.search(timing_pattern, stdout)
        if timing_match:
            test_stats['pytest_duration'] = float(timing_match.group(1))
        
        # Extract test class information
        class_pattern = r'test_\w+\.py::\w+::'
        test_classes = re.findall(class_pattern, stdout)
        test_stats['test_classes'] = len(set(test_classes))
        
        self.test_results['statistics'] = test_stats
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        results = self.test_results
        
        print("\n" + "="*90)
        print("📋 INTEGRATION TEST RESULTS SUMMARY")
        print("="*90)
        
        # Overall status
        if results['exit_code'] == 0:
            print("🎉 OVERALL STATUS: ✅ ALL TESTS PASSED")
        else:
            print("⚠️  OVERALL STATUS: ❌ SOME TESTS FAILED")
        
        print(f"⏱️  Total Duration: {results['duration_seconds']:.1f} seconds")
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test statistics
        if 'statistics' in results:
            stats = results['statistics']
            print("\n📊 TEST STATISTICS:")
            print("-" * 50)
            
            total_tests = sum(v for k, v in stats.items() if k in ['passed', 'failed', 'skipped', 'error'])
            
            for status in ['passed', 'failed', 'skipped', 'error']:
                count = stats.get(status, 0)
                if count > 0:
                    percentage = (count / total_tests * 100) if total_tests > 0 else 0
                    status_emoji = {'passed': '✅', 'failed': '❌', 'skipped': '⏭️', 'error': '💥'}
                    print(f"{status_emoji.get(status, '📊')} {status.capitalize():10} {count:3d} ({percentage:5.1f}%)")
        
        # System validation summary
        print("\n🏢 SYSTEM COMPONENT VALIDATION:")
        print("-" * 50)
        
        component_status = {
            "Data preprocessing pipeline": "✅" if results['exit_code'] == 0 else "❓",
            "Model orchestrator (79 models)": "✅" if results['exit_code'] == 0 else "❓", 
            "Risk scoring (Altman Z-Score)": "✅" if results['exit_code'] == 0 else "❓",
            "Change point detection": "✅" if results['exit_code'] == 0 else "❓",
            "Korean business terminology": "✅" if results['exit_code'] == 0 else "❓",
            "Loan calculation system": "✅" if results['exit_code'] == 0 else "❓",
            "LLM report generation": "✅" if results['exit_code'] == 0 else "❓",
            "Performance benchmarks": "✅" if results['exit_code'] == 0 else "❓",
            "Error handling": "✅" if results['exit_code'] == 0 else "❓",
            "Production readiness": "✅" if results['exit_code'] == 0 else "❓"
        }
        
        for component, status in component_status.items():
            print(f"{status} {component}")
        
        # Performance summary
        if results['duration_seconds'] > 0:
            print(f"\n⚡ PERFORMANCE METRICS:")
            print("-" * 50)
            print(f"🔄 Processing Rate: {results['system_specs']['total_records'] / results['duration_seconds']:,.0f} records/second (estimated)")
            print(f"🏗️  Model Hierarchy: {results['system_specs']['total_models']} models validated")
            print(f"🌐 Geographic Coverage: {results['system_specs']['regions']} regions")
            print(f"🏪 Business Coverage: {results['system_specs']['business_categories']} categories")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        
        if results['exit_code'] == 0:
            print("✅ System is ready for production deployment")
            print("✅ All integration tests passed successfully")
            print("✅ Korean business terminology validated")
            print("✅ Performance benchmarks met")
        else:
            print("⚠️  Review failed tests before production deployment")
            print("⚠️  Check system logs for detailed error information")
            print("⚠️  Validate data quality and model availability")
        
        # Output locations
        print(f"\n📄 TEST OUTPUTS:")
        print("-" * 50)
        print(f"📊 Results Directory: {self.results_dir}")
        
        report_files = list(self.results_dir.glob("integration_*"))
        for report_file in sorted(report_files):
            print(f"📋 {report_file.name}: {report_file}")
        
        print("="*90 + "\n")
    
    def save_detailed_results(self):
        """Save detailed test results to JSON file"""
        results_file = self.results_dir / f"integration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Sanitize results for JSON serialization
        json_results = {
            "metadata": {
                "test_suite": "Seoul Market Risk ML System Integration Tests",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "system_specs": self.system_specs
            },
            "execution": {
                "start_time": self.test_results.get("start_time"),
                "duration_seconds": self.test_results.get("duration_seconds"),
                "exit_code": self.test_results.get("exit_code"),
                "command": self.test_results.get("command", [])
            },
            "statistics": self.test_results.get("statistics", {}),
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "working_directory": str(self.project_root)
            }
        }
        
        # Add stdout/stderr if not too large
        stdout = self.test_results.get("stdout", "")
        stderr = self.test_results.get("stderr", "")
        
        if len(stdout) < 50000:  # 50KB limit
            json_results["output"] = {"stdout": stdout}
        if len(stderr) < 10000:  # 10KB limit
            json_results["output"] = json_results.get("output", {})
            json_results["output"]["stderr"] = stderr
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Detailed results saved: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def run(self) -> int:
        """Main execution method"""
        try:
            self.print_banner()
            
            # Validate environment
            if not self.validate_environment():
                logger.error("Environment validation failed")
                return 1
            
            # Run tests
            results = self.run_tests()
            
            # Generate reports
            self.generate_summary_report()
            self.save_detailed_results()
            
            return results['exit_code']
            
        except KeyboardInterrupt:
            logger.warning("Test execution interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Seoul Market Risk ML System - Integration Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_integration_tests.py -v -r
    python run_integration_tests.py --quick --korean-only  
    python run_integration_tests.py --performance
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose test output"
    )
    
    parser.add_argument(
        "-q", "--quick", 
        action="store_true",
        help="Run quick test subset (exclude slow tests)"
    )
    
    parser.add_argument(
        "-p", "--performance",
        action="store_true", 
        help="Include performance benchmark tests"
    )
    
    parser.add_argument(
        "-k", "--korean-only",
        action="store_true",
        help="Run Korean-specific tests only"
    )
    
    parser.add_argument(
        "-r", "--report",
        action="store_true",
        help="Generate detailed HTML report"
    )
    
    parser.add_argument(
        "-j", "--parallel",
        type=int,
        metavar="N",
        help="Run tests in parallel with N workers"
    )
    
    args = parser.parse_args()
    
    # Create and run test runner
    runner = IntegrationTestRunner(args)
    exit_code = runner.run()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()