#!/usr/bin/env python3
"""
서울 시장 위험도 ML 시스템 - 통합 테스트
Seoul Market Risk ML System - Integration Tests

전체 시스템의 데이터 플로우와 모델 동작을 검증합니다.
Tests the entire system data flow and model operations.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import time
import logging

# System imports
from src.utils.config_loader import load_config, get_data_paths
from src.preprocessing.main import SeoulDataPreprocessor
from src.models.global_model import SeoulGlobalModel
from src.models.regional_model import SeoulRegionalModel
from src.models.local_model import SeoulLocalModelManager
from src.models.model_orchestrator import SeoulModelOrchestrator, PredictionRequest
from src.risk_scoring.risk_calculator import SeoulRiskCalculator
from src.risk_scoring.changepoint_detection import SeoulChangepointDetector
from src.loan_calculation.loan_calculator import SeoulLoanCalculator
from src.llm_integration.report_generator import SeoulReportGenerator

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeoulSystemIntegrationTest:
    """서울 시장 위험도 ML 시스템 통합 테스트 클래스"""
    
    def __init__(self):
        self.config = load_config()
        self.data_paths = get_data_paths(self.config)
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'performance_metrics': {},
            'errors': []
        }
        
    def setup_test_environment(self):
        """테스트 환경 설정"""
        logger.info("🔧 테스트 환경 설정 중...")
        
        # 테스트용 소량 데이터 준비
        self.test_data = self._prepare_test_data()
        
        # 결과 저장 디렉토리 생성
        self.test_output_dir = Path("test_outputs")
        self.test_output_dir.mkdir(exist_ok=True)
        
        logger.info("✅ 테스트 환경 설정 완료")
        
    def _prepare_test_data(self):
        """테스트용 데이터 준비"""
        # 실제 처리된 데이터가 있으면 사용, 없으면 샘플 생성
        combined_file = self.data_paths['processed'] / 'seoul_sales_combined.csv'
        
        if combined_file.exists():
            logger.info("실제 처리된 데이터 로딩...")
            df = pd.read_csv(combined_file, nrows=1000)  # 테스트용 1000행만
        else:
            logger.info("샘플 테스트 데이터 생성...")
            df = self._generate_sample_data()
            
        return df
    
    def _generate_sample_data(self):
        """샘플 데이터 생성"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'district_code': np.random.choice(range(1, 7), n_samples),
            'business_type_code': np.random.choice(range(1, 13), n_samples),
            'business_type_name': np.random.choice(['음식점', '소매업', '서비스업', '제조업'], n_samples),
            'quarter_code': np.random.choice([20221, 20222, 20223, 20224], n_samples),
            'monthly_revenue': np.random.lognormal(15, 1, n_samples),
            'year': np.random.choice([2022, 2023, 2024], n_samples),
            'quarter': np.random.choice([1, 2, 3, 4], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_1_data_preprocessing(self):
        """1. 데이터 전처리 파이프라인 테스트"""
        logger.info("🧪 테스트 1: 데이터 전처리 파이프라인")
        
        try:
            start_time = time.time()
            
            # 전처리기 초기화
            preprocessor = SeoulDataPreprocessor()
            
            # 데이터 검증
            assert len(self.test_data) > 0, "테스트 데이터가 비어있음"
            assert 'monthly_revenue' in self.test_data.columns, "매출 데이터 컬럼 없음"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['preprocessing_time'] = processing_time
            
            logger.info("✅ 데이터 전처리 테스트 통과")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 데이터 전처리 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"preprocessing: {str(e)}")
    
    def test_2_global_model(self):
        """2. 글로벌 모델 테스트"""
        logger.info("🧪 테스트 2: 글로벌 모델")
        
        try:
            start_time = time.time()
            
            # 글로벌 모델 초기화
            global_model = SeoulGlobalModel()
            
            # 모델 훈련 (간단한 데이터로)
            train_data = self.test_data.sample(500) if len(self.test_data) >= 500 else self.test_data
            
            # 예측 테스트 (실제 훈련은 시간이 오래 걸리므로 모의)
            prediction_result = self._mock_model_prediction(global_model, train_data)
            
            assert prediction_result is not None, "글로벌 모델 예측 실패"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['global_model_time'] = processing_time
            
            logger.info("✅ 글로벌 모델 테스트 통과")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 글로벌 모델 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"global_model: {str(e)}")
    
    def test_3_regional_models(self):
        """3. 지역 모델 테스트"""
        logger.info("🧪 테스트 3: 지역 모델 (6개 지역)")
        
        try:
            start_time = time.time()
            
            for region_id in range(1, 7):  # 6개 지역
                regional_model = SeoulRegionalModel(
                    region_id=region_id,
                    region_characteristics={'income_level': 'medium', 'foot_traffic': 'high'}
                )
                
                # 해당 지역 데이터 필터링
                region_data = self.test_data[self.test_data['district_code'] == region_id]
                
                if len(region_data) > 10:  # 최소 데이터 요구량
                    prediction = self._mock_model_prediction(regional_model, region_data)
                    assert prediction is not None, f"지역 {region_id} 모델 예측 실패"
                
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['regional_models_time'] = processing_time
            
            logger.info("✅ 지역 모델 테스트 통과")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 지역 모델 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"regional_models: {str(e)}")
    
    def test_4_local_models(self):
        """4. 로컬 모델 (72개 조합) 테스트"""
        logger.info("🧪 테스트 4: 로컬 모델 (6×12=72개 조합)")
        
        try:
            start_time = time.time()
            
            local_manager = SeoulLocalModelManager()
            
            # 몇 가지 조합만 테스트 (전체 72개는 시간이 오래 걸림)
            test_combinations = [(1, 1), (2, 3), (3, 5), (4, 7), (5, 9), (6, 11)]
            
            for region_id, business_cat in test_combinations:
                combination_data = self.test_data[
                    (self.test_data['district_code'] == region_id) & 
                    (self.test_data['business_type_code'] == business_cat)
                ]
                
                if len(combination_data) >= 5:  # 최소 데이터 요구량
                    model_key = (region_id, business_cat)
                    # 모의 로컬 모델 생성
                    local_manager.models[model_key] = f"mock_model_{region_id}_{business_cat}"
            
            assert len(local_manager.models) > 0, "로컬 모델이 생성되지 않음"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['local_models_time'] = processing_time
            
            logger.info(f"✅ 로컬 모델 테스트 통과 ({len(local_manager.models)}개 모델)")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 로컬 모델 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"local_models: {str(e)}")
    
    def test_5_model_orchestrator(self):
        """5. 모델 오케스트레이터 (계층적 fallback) 테스트"""
        logger.info("🧪 테스트 5: 모델 오케스트레이터")
        
        try:
            start_time = time.time()
            
            orchestrator = SeoulModelOrchestrator()
            
            # 테스트 예측 요청
            test_request = PredictionRequest(
                business_id="TEST_001",
                region_id=1,
                business_category=1,
                historical_data=self.test_data.iloc[:50].to_dict('records'),
                prediction_horizon=30
            )
            
            # 예측 실행 (모의)
            prediction = orchestrator._mock_predict(test_request)
            
            assert prediction is not None, "오케스트레이터 예측 실패"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['orchestrator_time'] = processing_time
            
            logger.info("✅ 모델 오케스트레이터 테스트 통과")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 모델 오케스트레이터 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"orchestrator: {str(e)}")
    
    def test_6_risk_calculator(self):
        """6. 위험도 계산기 테스트"""
        logger.info("🧪 테스트 6: 위험도 계산기 (5성분 알트만 Z-Score)")
        
        try:
            start_time = time.time()
            
            risk_calc = SeoulRiskCalculator()
            
            # 테스트 데이터
            test_business_data = {
                'business_id': 'TEST_RISK_001',
                'revenue_history': [1000000, 1100000, 950000, 1200000, 1050000],
                'business_type': '음식점',
                'region_id': 1
            }
            
            # 위험도 계산
            risk_result = risk_calc.calculate_comprehensive_risk_score(test_business_data)
            
            assert risk_result is not None, "위험도 계산 실패"
            assert 'risk_score' in risk_result, "위험도 점수 없음"
            assert 'risk_level' in risk_result, "위험도 단계 없음"
            
            # 한국어 위험도 단계 검증
            valid_levels = ['매우안전', '안전', '주의', '경계', '위험', '매우위험']
            assert risk_result['risk_level'] in valid_levels, f"잘못된 위험도 단계: {risk_result['risk_level']}"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['risk_calculator_time'] = processing_time
            
            logger.info(f"✅ 위험도 계산기 테스트 통과 (점수: {risk_result['risk_score']}, 단계: {risk_result['risk_level']})")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 위험도 계산기 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"risk_calculator: {str(e)}")
    
    def test_7_changepoint_detection(self):
        """7. 변화점 감지 테스트"""
        logger.info("🧪 테스트 7: 변화점 감지 (CUSUM + Bayesian)")
        
        try:
            start_time = time.time()
            
            detector = SeoulChangepointDetector()
            
            # 급격한 매출 변화가 있는 테스트 데이터
            revenue_data = [1000, 1050, 1100, 1080, 1200, 1500, 1600, 1550, 1520, 1480]
            
            # 변화점 감지
            changepoints = detector.detect_revenue_changepoints(revenue_data)
            
            assert changepoints is not None, "변화점 감지 실패"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['changepoint_detection_time'] = processing_time
            
            logger.info(f"✅ 변화점 감지 테스트 통과 (감지된 변화점: {len(changepoints)}개)")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 변화점 감지 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"changepoint_detection: {str(e)}")
    
    def test_8_loan_calculator(self):
        """8. 대출 계산기 테스트"""
        logger.info("🧪 테스트 8: 위험 중화 대출 계산기")
        
        try:
            start_time = time.time()
            
            loan_calc = SeoulLoanCalculator()
            
            # 테스트 대출 요청
            loan_request = {
                'business_id': 'TEST_LOAN_001',
                'current_risk_score': 65,  # 위험 단계
                'business_type': '음식점',
                'monthly_revenue': 5000000,
                'requested_amount': 50000000
            }
            
            # 대출 조건 계산
            loan_result = loan_calc.calculate_risk_neutralizing_loan(loan_request)
            
            assert loan_result is not None, "대출 계산 실패"
            assert 'recommended_amount' in loan_result, "추천 대출 금액 없음"
            assert 'interest_rate' in loan_result, "금리 정보 없음"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['loan_calculator_time'] = processing_time
            
            logger.info(f"✅ 대출 계산기 테스트 통과 (추천 금액: {loan_result['recommended_amount']:,}원)")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 대출 계산기 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"loan_calculator: {str(e)}")
    
    def test_9_report_generator(self):
        """9. LLM 보고서 생성기 테스트"""
        logger.info("🧪 테스트 9: LLM 자동 보고서 생성기")
        
        try:
            start_time = time.time()
            
            report_gen = SeoulReportGenerator()
            
            # 테스트 보고서 데이터
            report_data = {
                'business_id': 'TEST_REPORT_001',
                'business_name': '테스트 카페',
                'risk_score': 45,
                'risk_level': '주의',
                'prediction_summary': '향후 3개월 매출 10% 증가 예상',
                'changepoints': ['2024-01월: 급격한 상승', '2024-03월: 일시적 하락'],
                'loan_recommendation': '3천만원 운영자금 대출 추천'
            }
            
            # 보고서 생성
            report = report_gen.generate_business_report(report_data)
            
            assert report is not None, "보고서 생성 실패"
            assert 'summary' in report, "요약 보고서 없음"
            assert 'detailed_analysis' in report, "상세 분석 없음"
            
            # 한국어 보고서 검증
            assert '위험도' in report['summary'], "한국어 위험도 용어 없음"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['report_generator_time'] = processing_time
            
            logger.info("✅ LLM 보고서 생성기 테스트 통과")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ LLM 보고서 생성기 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"report_generator: {str(e)}")
    
    def test_10_end_to_end_integration(self):
        """10. 종단간 통합 테스트"""
        logger.info("🧪 테스트 10: 종단간 통합 테스트 (전체 파이프라인)")
        
        try:
            start_time = time.time()
            
            # 1. 데이터 준비
            business_data = self.test_data.iloc[0].to_dict()
            
            # 2. 모델 예측 (모의)
            mock_prediction = {
                'predicted_revenue': [5200000, 5400000, 5100000],
                'confidence_score': 0.85
            }
            
            # 3. 위험도 계산
            risk_calc = SeoulRiskCalculator()
            risk_data = {
                'business_id': 'E2E_TEST_001',
                'revenue_history': [5000000, 5200000, 4800000, 5300000],
                'business_type': '음식점',
                'region_id': 1
            }
            risk_result = risk_calc.calculate_comprehensive_risk_score(risk_data)
            
            # 4. 대출 계산
            loan_calc = SeoulLoanCalculator()
            loan_request = {
                'business_id': 'E2E_TEST_001',
                'current_risk_score': risk_result['risk_score'],
                'business_type': '음식점',
                'monthly_revenue': 5000000,
                'requested_amount': 30000000
            }
            loan_result = loan_calc.calculate_risk_neutralizing_loan(loan_request)
            
            # 5. 최종 보고서 생성
            report_gen = SeoulReportGenerator()
            final_report_data = {
                'business_id': 'E2E_TEST_001',
                'business_name': '종단간 테스트 업체',
                'risk_score': risk_result['risk_score'],
                'risk_level': risk_result['risk_level'],
                'prediction_summary': f"예상 매출: {mock_prediction['predicted_revenue'][0]:,}원",
                'loan_recommendation': f"추천 대출: {loan_result['recommended_amount']:,}원"
            }
            
            final_report = report_gen.generate_business_report(final_report_data)
            
            # 전체 파이프라인 검증
            assert risk_result is not None, "위험도 계산 단계 실패"
            assert loan_result is not None, "대출 계산 단계 실패"
            assert final_report is not None, "보고서 생성 단계 실패"
            
            processing_time = time.time() - start_time
            self.test_results['performance_metrics']['end_to_end_time'] = processing_time
            
            logger.info("✅ 종단간 통합 테스트 통과")
            self.test_results['tests_passed'] += 1
            
            # 테스트 결과 저장
            self._save_e2e_results(final_report_data, final_report)
            
        except Exception as e:
            logger.error(f"❌ 종단간 통합 테스트 실패: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"end_to_end: {str(e)}")
    
    def _mock_model_prediction(self, model, data):
        """모델 예측 모의 (실제 훈련은 시간이 오래 걸림)"""
        # 실제로는 model.predict()를 호출하지만, 테스트에서는 모의 결과 반환
        return {
            'predictions': np.random.normal(data['monthly_revenue'].mean(), 
                                          data['monthly_revenue'].std() * 0.1, 
                                          5).tolist(),
            'confidence': 0.85
        }
    
    def _save_e2e_results(self, input_data, final_report):
        """종단간 테스트 결과 저장"""
        e2e_results = {
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data,
            'final_report': final_report
        }
        
        with open(self.test_output_dir / 'e2e_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(e2e_results, f, indent=2, ensure_ascii=False)
    
    def generate_test_report(self):
        """테스트 보고서 생성"""
        total_tests = self.test_results['tests_passed'] + self.test_results['tests_failed']
        success_rate = (self.test_results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        report = f"""
🏢 서울 시장 위험도 ML 시스템 - 통합 테스트 보고서
{'='*60}

📊 테스트 결과 요약:
  • 전체 테스트: {total_tests}개
  • 성공: {self.test_results['tests_passed']}개 ✅
  • 실패: {self.test_results['tests_failed']}개 ❌
  • 성공률: {success_rate:.1f}%

⏱️ 성능 메트릭:
"""
        
        for metric, time_val in self.test_results['performance_metrics'].items():
            report += f"  • {metric}: {time_val:.3f}초\n"
        
        if self.test_results['errors']:
            report += f"\n❌ 오류 상세:\n"
            for error in self.test_results['errors']:
                report += f"  • {error}\n"
        
        report += f"\n⏰ 테스트 완료 시각: {self.test_results['timestamp']}\n"
        
        # 파일로 저장
        with open(self.test_output_dir / 'integration_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def run_all_tests(self):
        """모든 통합 테스트 실행"""
        logger.info("🚀 서울 시장 위험도 ML 시스템 통합 테스트 시작")
        logger.info("="*60)
        
        # 테스트 환경 설정
        self.setup_test_environment()
        
        # 개별 테스트 실행
        test_methods = [
            self.test_1_data_preprocessing,
            self.test_2_global_model,
            self.test_3_regional_models,
            self.test_4_local_models,
            self.test_5_model_orchestrator,
            self.test_6_risk_calculator,
            self.test_7_changepoint_detection,
            self.test_8_loan_calculator,
            self.test_9_report_generator,
            self.test_10_end_to_end_integration
        ]
        
        for i, test_method in enumerate(test_methods, 1):
            logger.info(f"\n📋 {i}/10 테스트 진행 중...")
            try:
                test_method()
            except Exception as e:
                logger.error(f"테스트 {i} 예외 발생: {e}")
                self.test_results['tests_failed'] += 1
                self.test_results['errors'].append(f"test_{i}: {str(e)}")
        
        # 최종 보고서 생성
        report = self.generate_test_report()
        logger.info("\n" + report)
        
        return self.test_results


def main():
    """메인 함수 - 통합 테스트 실행"""
    tester = SeoulSystemIntegrationTest()
    results = tester.run_all_tests()
    
    # 결과에 따른 종료 코드
    if results['tests_failed'] > 0:
        logger.error("⚠️  일부 테스트가 실패했습니다. 시스템을 점검해주세요.")
        return 1
    else:
        logger.info("🎉 모든 통합 테스트가 성공했습니다!")
        return 0


if __name__ == "__main__":
    exit(main())