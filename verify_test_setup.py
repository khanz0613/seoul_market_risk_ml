#!/usr/bin/env python3
"""
Quick verification script for Seoul Market Risk ML System integration test setup.
Verifies that all components are properly configured before running full integration tests.
"""

import sys
import os
from pathlib import Path
import importlib
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version >= (3, 8):
        logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(f"❌ Python version {version.major}.{version.minor} < 3.8 (required)")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'pytest', 'pandas', 'numpy', 'scikit-learn', 
        'lightgbm', 'prophet', 'openai', 'jinja2',
        'yaml', 'pathlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'yaml':
                importlib.import_module('yaml')
            elif package == 'pathlib':
                importlib.import_module('pathlib')
            else:
                importlib.import_module(package)
            logger.info(f"✅ Package {package}: Available")
        except ImportError:
            logger.error(f"❌ Package {package}: Missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_project_structure():
    """Check project directory structure"""
    project_root = Path(__file__).parent
    required_paths = [
        'src',
        'src/tests', 
        'src/tests/integration_test.py',
        'src/tests/test_config.py',
        'config',
        'config/config.yaml',
        'data',
        'data/processed',
        'pytest.ini',
        'run_integration_tests.py'
    ]
    
    missing_paths = []
    for path_str in required_paths:
        path = project_root / path_str
        if path.exists():
            logger.info(f"✅ Path {path_str}: Exists")
        else:
            logger.error(f"❌ Path {path_str}: Missing")
            missing_paths.append(path_str)
    
    return len(missing_paths) == 0, missing_paths

def check_data_files():
    """Check for processed data files"""
    project_root = Path(__file__).parent
    data_dir = project_root / 'data' / 'processed'
    
    if not data_dir.exists():
        logger.error("❌ Processed data directory not found")
        return False, []
    
    csv_files = list(data_dir.glob('seoul_sales_*.csv'))
    json_files = list(data_dir.glob('*.json'))
    
    if csv_files:
        logger.info(f"✅ Data files: {len(csv_files)} CSV files found")
        for csv_file in csv_files[:3]:  # Show first 3
            logger.info(f"   📄 {csv_file.name}")
    else:
        logger.warning("⚠️  Data files: No seoul_sales_*.csv files found")
    
    if json_files:
        logger.info(f"✅ Metadata files: {len(json_files)} JSON files found")
        
    return len(csv_files) > 0, csv_files

def check_korean_text_support():
    """Check Korean text handling support"""
    try:
        # Test Korean text handling
        korean_text = "한식음식점"
        encoded = korean_text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        
        if decoded == korean_text:
            logger.info("✅ Korean text support: UTF-8 encoding works")
            return True
        else:
            logger.error("❌ Korean text support: UTF-8 encoding failed")
            return False
    except Exception as e:
        logger.error(f"❌ Korean text support: Error - {e}")
        return False

def check_test_imports():
    """Check if test modules can be imported"""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / 'src'))
    
    test_imports = [
        'tests.test_config',
        'utils.config_loader',
        'models.model_orchestrator'
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module in test_imports:
        try:
            importlib.import_module(module)
            logger.info(f"✅ Import {module}: Success")
            successful_imports.append(module)
        except ImportError as e:
            logger.error(f"❌ Import {module}: Failed - {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0, failed_imports

def check_configuration():
    """Check system configuration file"""
    project_root = Path(__file__).parent
    config_file = project_root / 'config' / 'config.yaml'
    
    if not config_file.exists():
        logger.error("❌ Configuration: config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['data', 'models', 'risk_scoring', 'loan_calculation']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            logger.error(f"❌ Configuration: Missing sections - {missing_sections}")
            return False
        
        # Check Korean business multipliers
        if 'business_multipliers' in config.get('loan_calculation', {}):
            multipliers = config['loan_calculation']['business_multipliers']
            korean_types = ['음식점', '소매업', '서비스업', '제조업']
            
            found_korean = any(korean_type in str(multipliers) for korean_type in korean_types)
            if found_korean:
                logger.info("✅ Configuration: Korean business multipliers found")
            else:
                logger.warning("⚠️  Configuration: Korean business multipliers not found")
        
        logger.info("✅ Configuration: config.yaml is valid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration: Error loading config.yaml - {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 SEOUL MARKET RISK ML SYSTEM - TEST SETUP VERIFICATION")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages), 
        ("Project Structure", check_project_structure),
        ("Data Files", check_data_files),
        ("Korean Text Support", check_korean_text_support),
        ("Test Imports", check_test_imports),
        ("Configuration", check_configuration)
    ]
    
    results = {}
    
    for check_name, check_function in checks:
        print(f"\n🔎 Checking {check_name}...")
        try:
            result = check_function()
            if isinstance(result, tuple):
                success, details = result
                results[check_name] = success
                if not success and details:
                    print(f"   Missing: {details}")
            else:
                results[check_name] = result
        except Exception as e:
            logger.error(f"❌ {check_name}: Unexpected error - {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - READY FOR INTEGRATION TESTING")
        print("\nNext steps:")
        print("  python run_integration_tests.py --quick")
        print("  python run_integration_tests.py -v -r")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE TESTING")
        print("\nCommon fixes:")
        print("  pip install -r requirements.txt")
        print("  Check data directory has seoul_sales_*.csv files")
        print("  Verify config/config.yaml exists and is valid")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)