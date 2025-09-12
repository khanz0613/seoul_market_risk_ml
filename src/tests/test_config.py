"""
Test Configuration for Seoul Market Risk ML System Integration Tests

Provides test-specific configurations, fixtures, and utilities for comprehensive testing.
"""

import os
import sys
from pathlib import Path
import tempfile
import pytest
import logging
from typing import Dict, Any, List

# Add src to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Test configuration constants
TEST_CONFIG = {
    "data": {
        "sample_size": 1000,
        "test_years": [2023, 2024],
        "korean_encoding": "utf-8",
        "test_regions": [11110515, 11110520, 11110525],  # 청운효자동, 사직동, 삼청동
        "test_business_types": {
            "CS100001": "한식음식점",
            "CS100003": "일식음식점", 
            "CS100004": "양식음식점",
            "CS200001": "의류소매",
            "CS300001": "미용실",
            "CS400001": "카페"
        }
    },
    "models": {
        "total_models": 79,
        "global_models": 1,
        "regional_models": 6,
        "local_models": 72,
        "fallback_chain": ["local", "regional", "global"],
        "confidence_thresholds": {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5
        }
    },
    "risk_scoring": {
        "risk_levels": {
            1: {"korean": "매우안전", "english": "Very Safe", "range": (0, 15)},
            2: {"korean": "안전", "english": "Safe", "range": (16, 35)},
            3: {"korean": "주의", "english": "Caution", "range": (36, 55)},
            4: {"korean": "경계", "english": "Warning", "range": (56, 75)},
            5: {"korean": "위험", "english": "Danger", "range": (76, 85)},
            6: {"korean": "매우위험", "english": "Very Danger", "range": (86, 100)}
        },
        "altman_components": [
            "revenue_change", "volatility", "trend", 
            "seasonal_deviation", "industry_comparison"
        ],
        "target_safe_score": 15
    },
    "loan_calculation": {
        "business_multipliers": {
            "음식점": 2.5,
            "소매업": 3.0,
            "서비스업": 1.5,
            "제조업": 4.0,
            "default": 2.0
        },
        "loan_products": [
            "운영자금대출", "긴급자금대출", "성장자금대출", 
            "안정화자금", "현금흐름지원대출"
        ],
        "max_repayment_ratio": 0.3  # 30% of net income
    },
    "performance": {
        "max_processing_time_per_record": 0.1,  # 100ms
        "min_throughput_records_per_second": 10,
        "max_memory_per_batch_mb": 100,
        "max_total_processing_time_seconds": 3600,  # 1 hour for full dataset
        "concurrent_request_limit": 50
    },
    "korean_language": {
        "business_terms": [
            "매출", "매출액", "수익", "손실", "위험도", "안전성",
            "현금흐름", "유동성", "변동성", "계절성", "트렌드",
            "예측", "분석", "권장", "추천"
        ],
        "risk_terms": [
            "매우안전", "안전", "주의", "경계", "위험", "매우위험"
        ],
        "report_emojis": ["🎯", "📊", "⚠️", "💡", "📈", "📉", "💰", "🔍"]
    }
}

# Test data specifications
SEOUL_DISTRICTS = {
    11110515: "청운효자동",
    11110520: "사직동", 
    11110525: "삼청동",
    11110530: "부암동",
    11110535: "평창동",
    11110540: "무악동",
    11140510: "중구을지로동",
    11140520: "중구명동",
    11170510: "용산구이태원동",
    11200510: "성동구성수동"
}

KOREAN_BUSINESS_CATEGORIES = {
    # Food & Beverage
    "CS100001": "한식음식점",
    "CS100002": "중식음식점", 
    "CS100003": "일식음식점",
    "CS100004": "양식음식점",
    "CS100005": "분식전문점",
    "CS100006": "패스트푸드",
    "CS100007": "치킨전문점",
    "CS100008": "피자전문점",
    "CS100009": "호프-간이주점",
    "CS100010": "일반음주점",
    
    # Retail
    "CS200001": "의류소매",
    "CS200002": "편의점",
    "CS200003": "화장품소매",
    "CS200004": "신발소매",
    "CS200005": "가방소매",
    "CS200006": "시계-귀금속소매",
    "CS200007": "완구-취미소매",
    "CS200008": "문구소매",
    "CS200009": "꽃소매",
    "CS200010": "애완용품소매",
    
    # Services
    "CS300001": "미용실",
    "CS300002": "세탁소",
    "CS300003": "목욕탕",
    "CS300004": "사진관",
    "CS300005": "부동산중개업",
    "CS300006": "여행사",
    "CS300007": "결혼예식장",
    "CS300008": "장례식장",
    "CS300009": "수리서비스",
    "CS300010": "청소서비스",
    
    # Coffee & Cafe
    "CS400001": "카페",
    "CS400002": "커피전문점",
    "CS400003": "차-전통음료점",
    "CS400004": "아이스크림-빙수점",
    
    # Education & Healthcare
    "CS500001": "학원",
    "CS500002": "기타교육기관",
    "CS600001": "병원",
    "CS600002": "의원",
    "CS600003": "치과의원",
    "CS600004": "한의원",
    "CS600005": "약국",
    "CS600006": "안경점"
}

# Test performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "data_preprocessing": {
        "max_time_per_1000_records": 5.0,  # seconds
        "memory_efficiency": 0.95,  # 95% memory utilization
        "error_rate_threshold": 0.01  # 1% error rate
    },
    "model_orchestrator": {
        "model_selection_time_ms": 10,
        "prediction_time_ms": 100,
        "fallback_success_rate": 0.95,
        "confidence_accuracy": 0.85
    },
    "risk_calculator": {
        "calculation_time_ms": 20,
        "altman_z_accuracy": 0.90,
        "korean_term_coverage": 1.0
    },
    "loan_calculator": {
        "calculation_time_ms": 15,
        "repayment_accuracy": 0.95,
        "business_type_coverage": 1.0
    },
    "report_generator": {
        "generation_time_ms": 500,
        "korean_quality_score": 0.90,
        "format_compliance": 1.0
    }
}

# Mock data generators for testing
class TestDataSpecs:
    """Specifications for generating realistic test data"""
    
    @staticmethod
    def get_revenue_distribution_by_business_type():
        """Get realistic revenue distributions for Korean business types"""
        return {
            "한식음식점": {"mean": 5000000, "std": 2000000, "min": 500000, "max": 20000000},
            "일식음식점": {"mean": 8000000, "std": 3000000, "min": 1000000, "max": 30000000},
            "양식음식점": {"mean": 7000000, "std": 2500000, "min": 800000, "max": 25000000},
            "중식음식점": {"mean": 4000000, "std": 1500000, "min": 400000, "max": 15000000},
            "카페": {"mean": 3500000, "std": 1200000, "min": 300000, "max": 12000000},
            "편의점": {"mean": 15000000, "std": 5000000, "min": 5000000, "max": 50000000},
            "의류소매": {"mean": 8000000, "std": 3000000, "min": 1000000, "max": 30000000},
            "미용실": {"mean": 2000000, "std": 800000, "min": 200000, "max": 8000000},
            "학원": {"mean": 6000000, "std": 2500000, "min": 500000, "max": 20000000},
            "병원": {"mean": 20000000, "std": 8000000, "min": 3000000, "max": 100000000}
        }
    
    @staticmethod 
    def get_seasonal_patterns_by_business_type():
        """Get seasonal patterns for Korean business types"""
        return {
            "한식음식점": [1.0, 0.9, 1.1, 1.2, 1.0, 0.8, 0.7, 0.8, 1.0, 1.1, 1.2, 1.3],
            "카페": [0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9],
            "편의점": [1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.2, 1.0, 1.0, 1.0, 1.1],
            "의류소매": [0.7, 0.8, 1.2, 1.3, 1.1, 1.0, 0.9, 0.8, 1.0, 1.2, 1.4, 1.3],
            "학원": [1.2, 1.3, 1.1, 0.8, 0.7, 0.6, 0.5, 0.6, 1.2, 1.3, 1.2, 1.1],
            "default": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    
    @staticmethod
    def get_risk_factor_weights():
        """Get risk factor weights for Korean business environment"""
        return {
            "revenue_volatility": 0.25,
            "seasonal_dependency": 0.20,
            "location_risk": 0.15,
            "competition_intensity": 0.15,
            "economic_sensitivity": 0.10,
            "regulatory_risk": 0.10,
            "operational_risk": 0.05
        }

# Test environment setup utilities
class TestEnvironment:
    """Utilities for setting up test environment"""
    
    @staticmethod
    def setup_test_logging():
        """Configure logging for testing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('tests/integration_test.log')
            ]
        )
        
        # Create logs directory if it doesn't exist
        Path('tests').mkdir(exist_ok=True)
        
        return logging.getLogger('integration_test')
    
    @staticmethod
    def create_temporary_config(custom_config: Dict[str, Any] = None) -> Path:
        """Create temporary configuration file for testing"""
        import yaml
        
        config = TEST_CONFIG.copy()
        if custom_config:
            config.update(custom_config)
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, encoding='utf-8'
        )
        
        yaml.dump(config, temp_file, default_flow_style=False, allow_unicode=True)
        temp_file.close()
        
        return Path(temp_file.name)
    
    @staticmethod
    def validate_korean_text(text: str) -> bool:
        """Validate Korean text contains proper Korean characters"""
        korean_chars = set('가나다라마바사아자차카타파하')
        return any(char in korean_chars for char in text)
    
    @staticmethod
    def cleanup_test_files(file_paths: List[Path]):
        """Clean up temporary test files"""
        for file_path in file_paths:
            try:
                if file_path.exists():
                    os.unlink(file_path)
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")

# Test assertion helpers
class TestAssertions:
    """Custom assertion helpers for Seoul Market Risk ML System"""
    
    @staticmethod
    def assert_korean_business_name(business_name: str):
        """Assert business name is valid Korean business name"""
        korean_business_indicators = [
            '음식점', '소매', '서비스', '카페', '학원', '병원', '미용실', '편의점'
        ]
        assert any(indicator in business_name for indicator in korean_business_indicators), \
            f"Business name should contain Korean business indicators: {business_name}"
    
    @staticmethod
    def assert_korean_risk_level(risk_level: str):
        """Assert risk level is valid Korean risk level"""
        valid_levels = ["매우안전", "안전", "주의", "경계", "위험", "매우위험"]
        assert risk_level in valid_levels, \
            f"Risk level should be valid Korean level: {risk_level}"
    
    @staticmethod
    def assert_revenue_reasonable(revenue: float, business_type: str = None):
        """Assert revenue is within reasonable bounds for Korean businesses"""
        assert revenue > 0, "Revenue should be positive"
        assert revenue < 1e12, "Revenue should be less than 1 trillion won"
        
        if business_type:
            revenue_specs = TestDataSpecs.get_revenue_distribution_by_business_type()
            if business_type in revenue_specs:
                spec = revenue_specs[business_type]
                min_reasonable = spec["min"]
                max_reasonable = spec["max"]
                assert min_reasonable <= revenue <= max_reasonable, \
                    f"Revenue {revenue} not reasonable for {business_type} " \
                    f"(expected {min_reasonable}-{max_reasonable})"
    
    @staticmethod
    def assert_model_hierarchy_valid(model_counts: Dict[str, int]):
        """Assert model hierarchy matches expected structure"""
        expected = TEST_CONFIG["models"]
        assert model_counts.get("global", 0) == expected["global_models"], \
            f"Should have {expected['global_models']} global models"
        assert model_counts.get("regional", 0) == expected["regional_models"], \
            f"Should have {expected['regional_models']} regional models"
        assert model_counts.get("local", 0) == expected["local_models"], \
            f"Should have {expected['local_models']} local models"
        
        total = sum(model_counts.values())
        assert total == expected["total_models"], \
            f"Should have {expected['total_models']} total models, got {total}"
    
    @staticmethod
    def assert_performance_acceptable(
        metric_name: str, 
        measured_value: float, 
        benchmark_type: str = "data_preprocessing"
    ):
        """Assert performance metric meets benchmark requirements"""
        benchmarks = PERFORMANCE_BENCHMARKS.get(benchmark_type, {})
        
        if metric_name in benchmarks:
            threshold = benchmarks[metric_name]
            assert measured_value <= threshold, \
                f"{metric_name} performance {measured_value} exceeds threshold {threshold}"

# Export test configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test") 
    config.addinivalue_line("markers", "korean: mark test as Korean-specific test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

# Test fixtures for common use
@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration"""
    return TEST_CONFIG

@pytest.fixture(scope="session") 
def korean_business_types():
    """Provide Korean business type mappings"""
    return KOREAN_BUSINESS_CATEGORIES

@pytest.fixture(scope="session")
def seoul_districts():
    """Provide Seoul district mappings"""
    return SEOUL_DISTRICTS

@pytest.fixture(scope="session")
def performance_benchmarks():
    """Provide performance benchmark specifications"""
    return PERFORMANCE_BENCHMARKS

@pytest.fixture
def temp_config_file():
    """Create temporary configuration file"""
    config_path = TestEnvironment.create_temporary_config()
    yield config_path
    TestEnvironment.cleanup_test_files([config_path])

@pytest.fixture
def test_logger():
    """Provide configured test logger"""
    return TestEnvironment.setup_test_logging()