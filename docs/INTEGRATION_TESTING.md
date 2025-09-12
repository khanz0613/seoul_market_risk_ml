# Integration Testing Guide - Seoul Market Risk ML System

## Overview

This document provides comprehensive guidance for running and interpreting integration tests for the Seoul Market Risk ML System. The system processes 408,221 records across 6 years (2019-2024) using a hierarchical ML architecture with 79 models (1 Global + 6 Regional + 72 Local).

## System Architecture Under Test

### 🏗️ Model Hierarchy
- **Global Model**: 1 model (Prophet + ARIMA ensemble)
- **Regional Models**: 6 models (Prophet + ARIMA + LightGBM, one per region)
- **Local Models**: 72 models (6 regions × 12 business categories)
- **Total**: 79 models with intelligent fallback chain (Local → Regional → Global)

### 📊 Data Pipeline
- **Raw Data**: Seoul commercial district sales data (2019-2024)
- **Processed Data**: 408,221 records with Korean business terminology
- **Features**: Revenue patterns, seasonal trends, geographic factors
- **Languages**: Korean business types, risk levels, and terminology

### 🎯 Core Components Tested
1. **Data Preprocessing Pipeline**: Schema validation, Korean text handling
2. **Model Orchestrator**: Intelligent model selection and fallback
3. **Risk Score Calculator**: 5-component Altman Z-Score with Korean risk levels
4. **Change Point Detection**: CUSUM + Bayesian for trend analysis
5. **Loan Calculator**: Risk-neutralizing loan recommendations
6. **LLM Report Generator**: Korean business reports with proper terminology
7. **Performance Benchmarks**: Scalability and timing validation

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install pytest pandas numpy scikit-learn prophet lightgbm openai jinja2

# Verify environment
python -c "import pytest, pandas, numpy; print('Environment OK')"
```

### Basic Test Execution

```bash
# Run all integration tests
python run_integration_tests.py

# Run with verbose output and HTML report
python run_integration_tests.py -v -r

# Run quick test subset (excludes slow tests)
python run_integration_tests.py --quick

# Run Korean-specific tests only
python run_integration_tests.py --korean-only
```

## Test Categories

### 🧪 Integration Test Classes

#### 1. `TestDataPreprocessingPipeline`
**Purpose**: Validate data loading, schema validation, and Korean text handling

**Key Tests**:
- `test_data_loading_and_schema_validation`: Ensures proper data structure and Korean business names
- `test_feature_engineering_pipeline`: Validates feature extraction and Korean business categorization
- `test_data_quality_validation`: Checks data consistency and missing value handling

**Expected Results**:
```
✅ Data loading: 408,221 records processed
✅ Korean business types: 음식점, 소매업, 서비스업, etc.
✅ Feature extraction: revenue_stability, business_diversity
✅ Data quality: <10% missing values in critical columns
```

#### 2. `TestModelHierarchyAndOrchestrator`
**Purpose**: Test 79-model hierarchy and intelligent orchestration

**Key Tests**:
- `test_model_selection_strategy`: Validates Local → Regional → Global fallback
- `test_fallback_chain_execution`: Tests hierarchical fallback logic
- `test_cold_start_handling`: Validates new business/region handling
- `test_performance_metrics_tracking`: Monitors model performance
- `test_batch_prediction_processing`: Validates batch processing efficiency

**Expected Results**:
```
✅ Model hierarchy: 79 models (1+6+72) available
✅ Fallback chain: Local → Regional → Global working
✅ Cold start: Global model handles unknown regions/businesses
✅ Performance: <100ms average prediction time
✅ Batch processing: >10 records/second throughput
```

#### 3. `TestRiskScoringSystem`
**Purpose**: Validate Altman Z-Score calculation and Korean risk levels

**Key Tests**:
- `test_altman_z_score_calculation`: 5-component risk score calculation
- `test_korean_business_risk_factors`: Industry-specific risk adjustments
- `test_change_point_detection`: CUSUM + Bayesian trend analysis
- `test_risk_level_thresholds`: Korean risk level classification accuracy

**Expected Results**:
```
✅ Risk calculation: 5 components (매출변화, 변동성, 트렌드, 계절편차, 업종비교)
✅ Korean risk levels: 매우안전, 안전, 주의, 경계, 위험, 매우위험
✅ Industry multipliers: 음식점(2.5), 소매업(3.0), 서비스업(1.5), 제조업(4.0)
✅ Change points: Detects rapid decline and volatility changes
```

#### 4. `TestLoanCalculationSystem`
**Purpose**: Test risk-neutralizing loan calculations

**Key Tests**:
- `test_risk_neutralizing_loan_calculation`: Loan amount to reduce risk to safe levels
- `test_korean_loan_product_types`: Korean loan product categorization
- `test_repayment_capacity_analysis`: Affordability and repayment analysis
- `test_korean_currency_formatting`: Korean won handling and formatting

**Expected Results**:
```
✅ Loan calculation: Reduces risk score to target level (15 points)
✅ Korean products: 운영자금대출, 긴급자금대출, 성장자금대출
✅ Repayment analysis: <30% of net income for monthly payments
✅ Currency handling: Korean won amounts (millions) properly formatted
```

#### 5. `TestLLMReportGeneration`
**Purpose**: Validate Korean business report generation

**Key Tests**:
- `test_korean_business_report_generation`: Korean terminology and structure
- `test_business_terminology_accuracy`: Industry-specific Korean terms
- `test_report_formatting_and_structure`: Emoji + Korean text formatting

**Expected Results**:
```
✅ Korean reports: Proper business terminology usage
✅ Risk communication: 🎯 위험도, 📊 매출 전망, ⚠️ 주요 원인, 💡 추천
✅ Industry terms: 매출, 고객수, 회전율, 계절성 (restaurants)
✅ Formatting: Structured Korean business reports with emojis
```

#### 6. `TestEndToEndIntegration`
**Purpose**: Comprehensive end-to-end pipeline validation

**Key Tests**:
- `test_complete_data_pipeline_flow`: Full pipeline from data to report
- `test_system_performance_benchmarks`: Performance and scalability
- `test_error_handling_and_recovery`: Comprehensive error scenarios
- `test_data_consistency_across_components`: Data integrity validation

**Expected Results**:
```
✅ End-to-end: Data → Prediction → Risk → Loan → Report flow
✅ Performance: <0.1s per record, >10 records/second throughput
✅ Error handling: Graceful degradation and recovery
✅ Data consistency: Revenue, business types, geographic data aligned
```

#### 7. `TestSystemIntegrationBenchmarks`
**Purpose**: Performance and scalability validation

**Key Tests**:
- `test_model_hierarchy_performance`: 79-model selection performance
- `test_data_volume_handling`: 408,221 records processing
- `test_concurrent_processing_scalability`: Multi-threaded performance

**Expected Results**:
```
✅ Model selection: <10ms average, <50ms worst-case
✅ Data volume: <1 hour for full dataset (408K records)
✅ Concurrency: >10 req/sec with 5 concurrent threads
✅ Memory: <100MB per 1000-record batch
```

#### 8. `TestProductionReadiness`
**Purpose**: Production deployment readiness validation

**Key Tests**:
- `test_system_configuration_validation`: Configuration completeness
- `test_logging_and_monitoring_setup`: Logging infrastructure
- `test_error_reporting_and_alerting`: Error handling systems
- `test_data_backup_and_recovery`: Data integrity and recovery

## Performance Benchmarks

### 📈 Expected Performance Metrics

| Component | Metric | Target | Measurement |
|-----------|--------|--------|-------------|
| **Data Preprocessing** | Records/second | >1,000 | Per 1K batch |
| **Model Orchestrator** | Selection time | <10ms | Per prediction |
| **Risk Calculator** | Calculation time | <20ms | Per assessment |
| **Loan Calculator** | Processing time | <15ms | Per calculation |
| **Report Generator** | Generation time | <500ms | Per report |
| **End-to-End** | Full pipeline | <100ms | Per business |
| **Batch Processing** | Throughput | >10/sec | Concurrent requests |
| **Memory Usage** | Per batch (1K) | <100MB | Peak memory |

### 🎯 Korean Business Validation

| Category | Validation | Expected Result |
|----------|------------|----------------|
| **Business Types** | Korean names | 음식점, 소매업, 서비스업, 제조업 |
| **Risk Levels** | Korean labels | 매우안전, 안전, 주의, 경계, 위험, 매우위험 |
| **Loan Products** | Korean terms | 운영자금대출, 긴급자금대출, 성장자금대출 |
| **Report Format** | Korean + emojis | 🎯 위험도, 📊 매출 전망, ⚠️ 주요 원인 |
| **Currency** | Korean won | 1,000,000원 ~ 100,000,000원 ranges |

## Advanced Test Execution

### 🔧 Command Line Options

```bash
# Basic execution
python run_integration_tests.py

# Verbose output with detailed logging
python run_integration_tests.py -v

# Quick tests (exclude slow/performance tests)
python run_integration_tests.py --quick

# Performance benchmarks only
python run_integration_tests.py --performance

# Korean-specific tests only
python run_integration_tests.py --korean-only

# Generate HTML report
python run_integration_tests.py --report

# Parallel execution (4 workers)
python run_integration_tests.py -j 4

# Combined options
python run_integration_tests.py -v -r --performance
```

### 📊 Using pytest directly

```bash
# All integration tests
pytest src/tests/integration_test.py -v

# Specific test class
pytest src/tests/integration_test.py::TestModelHierarchyAndOrchestrator -v

# Tests with specific marker
pytest src/tests/integration_test.py -m "korean" -v
pytest src/tests/integration_test.py -m "performance" -v
pytest src/tests/integration_test.py -m "not slow" -v

# Generate coverage report
pytest src/tests/integration_test.py --cov=src --cov-report=html
```

## Interpreting Test Results

### ✅ Success Indicators

**Console Output**:
```
🎉 OVERALL STATUS: ✅ ALL TESTS PASSED
📊 TEST STATISTICS:
✅ Passed       45 (100.0%)
⏱️  Total Duration: 120.5 seconds

🏢 SYSTEM COMPONENT VALIDATION:
✅ Data preprocessing pipeline
✅ Model orchestrator (79 models)
✅ Risk scoring (Altman Z-Score)
✅ Change point detection
✅ Korean business terminology
✅ Loan calculation system
✅ LLM report generation
✅ Performance benchmarks
✅ Error handling
✅ Production readiness
```

### ❌ Failure Indicators

**Common Failure Patterns**:

1. **Data Issues**:
```
❌ test_data_loading_and_schema_validation FAILED
AssertionError: Should contain Korean business type names
```
**Solution**: Check data encoding, verify Korean text preservation

2. **Model Unavailability**:
```
❌ test_model_selection_strategy FAILED
AssertionError: Local model should be available
```
**Solution**: Verify model files exist, check training status

3. **Performance Issues**:
```
❌ test_system_performance_benchmarks FAILED
AssertionError: Average processing time too slow: 0.150s
```
**Solution**: Optimize code, check system resources, review algorithms

4. **Korean Text Issues**:
```
❌ test_korean_business_report_generation FAILED
AssertionError: Should use Korean business terminology
```
**Solution**: Check encoding settings, verify LLM configuration

### 📋 Test Report Files

After execution, check the `test_results/` directory:

```
test_results/
├── integration_report_20241201_143022.html    # Detailed HTML report
├── integration_results_20241201_143022.json   # Machine-readable results
├── integration_junit_20241201_143022.xml      # CI/CD compatible results
└── pytest.log                                 # Detailed execution log
```

## Troubleshooting

### 🔍 Common Issues

#### Environment Setup
```bash
# Missing packages
pip install -r requirements.txt

# Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Korean encoding issues
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8
```

#### Data Issues
```python
# Check data files
ls -la data/processed/seoul_sales_*.csv

# Verify Korean text
head -1 data/processed/seoul_sales_2024.csv | grep "음식점"

# Check file encoding
file data/processed/seoul_sales_2024.csv
```

#### Model Issues
```python
# Check model availability
ls -la src/models/*/
ls -la src/models/global/
ls -la src/models/regional/
ls -la src/models/local/
```

#### Configuration Issues
```yaml
# Verify config.yaml
cat config/config.yaml | grep -A 5 "business_multipliers"
```

### 🐛 Debugging Failed Tests

1. **Run specific failing test**:
```bash
pytest src/tests/integration_test.py::TestDataPreprocessingPipeline::test_data_loading_and_schema_validation -v -s
```

2. **Enable debug logging**:
```bash
pytest src/tests/integration_test.py -v -s --log-cli-level=DEBUG
```

3. **Check test data**:
```python
import pandas as pd
df = pd.read_csv('data/processed/seoul_sales_2024.csv', nrows=5)
print(df.columns.tolist())
print(df['business_type_name'].unique())
```

4. **Verify Korean support**:
```python
import locale, sys
print(f"Locale: {locale.getpreferredencoding()}")
print(f"Korean test: {'한글' == '한글'}")
```

## Continuous Integration

### 🔄 CI/CD Integration

**GitHub Actions Example**:
```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run integration tests
      run: |
        python run_integration_tests.py --quick --report
        
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: test_results/
```

### 📈 Performance Monitoring

Monitor these key metrics over time:
- **Test execution time**: Should remain stable or improve
- **Test pass rate**: Should maintain >95% success rate
- **Performance benchmarks**: Should meet or exceed targets
- **Korean text handling**: Should maintain 100% accuracy

## Best Practices

### ✨ Test Development Guidelines

1. **Korean Text Handling**:
   - Always use UTF-8 encoding
   - Test Korean business names and risk levels
   - Validate proper Korean terminology usage

2. **Performance Testing**:
   - Set realistic benchmarks based on production requirements
   - Test with representative data volumes
   - Monitor memory usage and processing time

3. **Error Handling**:
   - Test edge cases and boundary conditions
   - Validate graceful degradation
   - Ensure proper error messages

4. **Data Validation**:
   - Verify data consistency across components
   - Test with both clean and messy data
   - Validate Korean business type mappings

### 🎯 Maintenance

- **Regular Updates**: Update test data and expectations as system evolves
- **Performance Baselines**: Adjust benchmarks based on production metrics
- **Korean Language**: Validate terminology with native speakers
- **Documentation**: Keep this guide updated with new tests and procedures

---

## Contact & Support

For questions about integration testing:
- **Technical Issues**: Check logs in `test_results/pytest.log`
- **Performance Issues**: Review benchmark targets in `pytest.ini`
- **Korean Language Issues**: Verify encoding and terminology mapping
- **Data Issues**: Check data processing pipeline and validation rules

This comprehensive integration test suite ensures the Seoul Market Risk ML System is production-ready with proper Korean business terminology, robust performance, and reliable end-to-end functionality.