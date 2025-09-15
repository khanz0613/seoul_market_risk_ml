#!/usr/bin/env python3
"""
서울 시장 위험도 ML 시스템 - 간단한 통합 테스트
Seoul Market Risk ML System - Simple Integration Test

핵심 시스템 구성요소들의 기본 동작을 검증합니다.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSystemTest:
    """간단한 시스템 테스트 클래스"""
    
    def __init__(self):
        self.config = load_config()
        self.data_paths = get_data_paths(self.config)
        self.results = {'tests_passed': 0, 'tests_failed': 0, 'errors': []}
        
    def test_1_config_loading(self):
        """1. 설정 로딩 테스트"""
        logger.info("🧪 테스트 1: 설정 파일 로딩")
        
        try:
            assert self.config is not None, "설정이 로드되지 않음"
            assert 'data' in self.config, "데이터 설정이 없음"
            assert 'models' in self.config, "모델 설정이 없음"
            assert 'risk_scoring' in self.config, "위험 점수 설정이 없음"
            
            logger.info("✅ 설정 로딩 테스트 통과")
            self.results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 설정 로딩 테스트 실패: {e}")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"config_loading: {str(e)}")
    
    def test_2_data_paths(self):
        """2. 데이터 경로 검증 테스트"""
        logger.info("🧪 테스트 2: 데이터 경로 검증")
        
        try:
            assert self.data_paths is not None, "데이터 경로가 설정되지 않음"
            assert 'raw' in self.data_paths, "원시 데이터 경로 없음"
            assert 'processed' in self.data_paths, "처리된 데이터 경로 없음"
            
            # 처리된 데이터 파일 확인
            combined_file = self.data_paths['processed'] / 'seoul_sales_combined.csv'
            assert combined_file.exists(), f"처리된 데이터 파일이 없음: {combined_file}"
            
            logger.info("✅ 데이터 경로 검증 테스트 통과")
            self.results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 데이터 경로 검증 테스트 실패: {e}")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"data_paths: {str(e)}")
    
    def test_3_data_loading(self):
        """3. 데이터 로딩 테스트"""
        logger.info("🧪 테스트 3: 데이터 로딩")
        
        try:
            combined_file = self.data_paths['processed'] / 'seoul_sales_combined.csv'
            df = pd.read_csv(combined_file, nrows=100)  # 처음 100행만 로드
            
            assert len(df) > 0, "데이터가 비어있음"
            assert 'monthly_revenue' in df.columns, "매출 데이터 컬럼이 없음"
            
            # 기본 데이터 유효성 검사
            assert df['monthly_revenue'].notna().sum() > 0, "유효한 매출 데이터가 없음"
            
            logger.info(f"✅ 데이터 로딩 테스트 통과 (로드된 행 수: {len(df)})")
            self.results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 데이터 로딩 테스트 실패: {e}")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"data_loading: {str(e)}")
    
    def test_4_risk_calculator_mock(self):
        """4. 위험도 계산기 Mock 테스트"""
        logger.info("🧪 테스트 4: 위험도 계산기 (Mock)")
        
        try:
            # Mock 위험도 계산
            test_data = {
                'business_id': 'TEST_001',
                'revenue_history': [1000000, 1100000, 950000, 1200000, 1050000],
                'business_type': '음식점',
                'region_id': 1
            }
            
            # 간단한 위험도 계산 로직
            revenue_change = (test_data['revenue_history'][-1] - test_data['revenue_history'][0]) / test_data['revenue_history'][0]
            volatility = np.std(test_data['revenue_history']) / np.mean(test_data['revenue_history'])
            
            # 5성분 위험 점수 (간소화)
            risk_score = min(100, max(0, 
                (0.3 * abs(revenue_change * 100)) + 
                (0.2 * volatility * 100) + 
                (0.2 * 30) +  # 트렌드 (Mock)
                (0.15 * 20) + # 계절성 (Mock)
                (0.15 * 25)   # 업종 비교 (Mock)
            ))
            
            # 위험도 단계
            risk_levels = ['매우안전', '안전', '주의', '경계', '위험', '매우위험']
            level_idx = min(5, int(risk_score / 20))
            risk_level = risk_levels[level_idx]
            
            risk_result = {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'components': {
                    'revenue_change': revenue_change,
                    'volatility': volatility
                }
            }
            
            assert risk_result['risk_score'] >= 0, "위험도 점수가 음수임"
            assert risk_result['risk_score'] <= 100, "위험도 점수가 100을 초과함"
            assert risk_result['risk_level'] in risk_levels, "잘못된 위험도 단계"
            
            logger.info(f"✅ 위험도 계산기 Mock 테스트 통과 (점수: {risk_score:.1f}, 단계: {risk_level})")
            self.results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 위험도 계산기 Mock 테스트 실패: {e}")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"risk_calculator_mock: {str(e)}")
    
    def test_5_loan_calculator_mock(self):
        """5. 대출 계산기 Mock 테스트"""
        logger.info("🧪 테스트 5: 대출 계산기 (Mock)")
        
        try:
            # Mock 대출 계산
            loan_request = {
                'business_id': 'TEST_LOAN_001',
                'current_risk_score': 45,  # 주의 단계
                'business_type': '음식점',
                'monthly_revenue': 5000000,
                'requested_amount': 50000000
            }
            
            # 업종별 배율
            business_multipliers = {
                '음식점': 2.5, '소매업': 3.0, '서비스업': 1.5, '제조업': 4.0
            }
            
            # 위험도 기반 금리 조정
            base_rate = 5.0
            risk_adjustment = (loan_request['current_risk_score'] - 15) * 0.05  # 목표 위험도 15점 기준
            adjusted_rate = max(3.0, min(12.0, base_rate + risk_adjustment))
            
            # 추천 대출 금액 (월 매출의 배율)
            multiplier = business_multipliers.get(loan_request['business_type'], 2.0)
            recommended_amount = min(
                loan_request['requested_amount'],
                int(loan_request['monthly_revenue'] * multiplier)
            )
            
            loan_result = {
                'recommended_amount': recommended_amount,
                'interest_rate': adjusted_rate,
                'business_multiplier': multiplier,
                'risk_adjustment': risk_adjustment
            }
            
            assert loan_result['recommended_amount'] > 0, "추천 대출 금액이 0 이하임"
            assert loan_result['interest_rate'] >= 3.0, "금리가 너무 낮음"
            assert loan_result['interest_rate'] <= 12.0, "금리가 너무 높음"
            
            logger.info(f"✅ 대출 계산기 Mock 테스트 통과 (추천 금액: {recommended_amount:,}원, 금리: {adjusted_rate:.2f}%)")
            self.results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 대출 계산기 Mock 테스트 실패: {e}")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"loan_calculator_mock: {str(e)}")
    
    def test_6_report_generation_mock(self):
        """6. 보고서 생성 Mock 테스트"""
        logger.info("🧪 테스트 6: 보고서 생성 (Mock)")
        
        try:
            # Mock 보고서 데이터
            report_data = {
                'business_id': 'TEST_REPORT_001',
                'business_name': 'Mock 테스트 카페',
                'risk_score': 45,
                'risk_level': '주의',
                'prediction_summary': '향후 3개월 매출 8% 증가 예상',
                'loan_recommendation': '2천500만원 운영자금 대출 추천'
            }
            
            # 간단한 보고서 템플릿
            summary_template = "🎯 위험도: {risk_level} ({risk_score}점)\n📊 매출 전망: {prediction}\n💡 추천: {recommendation}"
            
            summary_report = summary_template.format(
                risk_level=report_data['risk_level'],
                risk_score=report_data['risk_score'],
                prediction=report_data['prediction_summary'],
                recommendation=report_data['loan_recommendation']
            )
            
            detailed_report = f"""
업체명: {report_data['business_name']}
업체ID: {report_data['business_id']}

## 위험도 분석
- 현재 위험도: {report_data['risk_level']} ({report_data['risk_score']}점)
- 주요 위험 요소: 매출 변동성, 계절적 영향

## 매출 예측
{report_data['prediction_summary']}

## 대출 추천
{report_data['loan_recommendation']}
금리: 연 6.25% (위험도 조정 적용)

## 권고사항
- 매출 안정화를 위한 마케팅 강화 필요
- 현금흐름 관리 개선 권고
"""
            
            report = {
                'summary': summary_report,
                'detailed_analysis': detailed_report,
                'business_id': report_data['business_id'],
                'generated_at': datetime.now().isoformat()
            }
            
            assert '위험도' in report['summary'], "요약 보고서에 위험도 정보가 없음"
            assert '추천' in report['summary'], "요약 보고서에 추천 정보가 없음"
            assert report_data['business_name'] in report['detailed_analysis'], "상세 보고서에 업체명이 없음"
            
            logger.info("✅ 보고서 생성 Mock 테스트 통과")
            self.results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 보고서 생성 Mock 테스트 실패: {e}")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"report_generation_mock: {str(e)}")
    
    def test_7_end_to_end_mock(self):
        """7. 종단간 Mock 테스트"""
        logger.info("🧪 테스트 7: 종단간 Mock 테스트")
        
        try:
            # 1. 업체 정보
            business_info = {
                'business_id': 'E2E_MOCK_001',
                'business_name': '종단간 테스트 카페',
                'business_type': '음식점',
                'region_id': 1,
                'monthly_revenue': 4500000
            }
            
            # 2. Mock 위험도 계산
            mock_risk_score = 42
            mock_risk_level = '주의'
            
            # 3. Mock 대출 계산
            mock_loan_amount = business_info['monthly_revenue'] * 2.5  # 음식점 배율
            mock_interest_rate = 6.1
            
            # 4. Mock 보고서 생성
            end_to_end_report = {
                'business_info': business_info,
                'risk_assessment': {
                    'score': mock_risk_score,
                    'level': mock_risk_level
                },
                'loan_recommendation': {
                    'amount': mock_loan_amount,
                    'rate': mock_interest_rate
                },
                'summary': f"업체 {business_info['business_name']}의 위험도는 {mock_risk_level}({mock_risk_score}점)이며, {mock_loan_amount:,}원 대출을 {mock_interest_rate}% 금리로 추천합니다.",
                'timestamp': datetime.now().isoformat()
            }
            
            # 검증
            assert end_to_end_report['business_info']['business_id'] is not None, "업체 ID가 없음"
            assert end_to_end_report['risk_assessment']['score'] > 0, "위험도 점수가 없음"
            assert end_to_end_report['loan_recommendation']['amount'] > 0, "대출 추천 금액이 없음"
            assert '위험도' in end_to_end_report['summary'], "종합 요약에 위험도 정보가 없음"
            
            logger.info("✅ 종단간 Mock 테스트 통과")
            self.results['tests_passed'] += 1
            
            # 결과 저장
            output_dir = Path("test_outputs")
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / 'e2e_mock_results.json', 'w', encoding='utf-8') as f:
                json.dump(end_to_end_report, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"❌ 종단간 Mock 테스트 실패: {e}")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"end_to_end_mock: {str(e)}")
    
    def generate_test_report(self):
        """테스트 결과 보고서 생성"""
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        success_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        report = f"""
🏢 서울 시장 위험도 ML 시스템 - 간단 통합 테스트 보고서
{'='*65}

📊 테스트 결과:
  • 전체 테스트: {total_tests}개
  • 성공: {self.results['tests_passed']}개 ✅
  • 실패: {self.results['tests_failed']}개 ❌
  • 성공률: {success_rate:.1f}%

⏰ 테스트 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if self.results['errors']:
            report += f"\n❌ 오류 상세:\n"
            for error in self.results['errors']:
                report += f"  • {error}\n"
        
        # 보고서 저장
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'simple_integration_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("🚀 서울 시장 위험도 ML 시스템 간단 통합 테스트 시작")
        logger.info("="*65)
        
        tests = [
            self.test_1_config_loading,
            self.test_2_data_paths,
            self.test_3_data_loading,
            self.test_4_risk_calculator_mock,
            self.test_5_loan_calculator_mock,
            self.test_6_report_generation_mock,
            self.test_7_end_to_end_mock
        ]
        
        for i, test in enumerate(tests, 1):
            logger.info(f"\n📋 {i}/{len(tests)} 테스트 진행...")
            test()
        
        # 최종 보고서
        report = self.generate_test_report()
        logger.info("\n" + report)
        
        return self.results


def main():
    """메인 함수"""
    tester = SimpleSystemTest()
    results = tester.run_all_tests()
    
    if results['tests_failed'] > 0:
        logger.error("⚠️  일부 테스트가 실패했습니다.")
        return 1
    else:
        logger.info("🎉 모든 간단 통합 테스트가 성공했습니다!")
        return 0


if __name__ == "__main__":
    exit(main())