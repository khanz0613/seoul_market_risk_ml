"""
Seoul Market Risk ML System v2.0 - 하이브리드 모델
소상공인 맞춤형 위험도 산정 및 대출 추천 시스템

주요 특징:
- Altman Z'-Score 기반 재무 건전성 평가 (40%)
- 영업 안정성 분석 (45%): 매출 트렌드, 변동성, 지속성
- 업종 내 상대적 위치 (15%)
- 구체적인 대출 금액 계산 및 위험도 개선 시뮬레이션
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

from src.data_processing.expense_estimator import ExpenseEstimator
from src.risk_scoring.hybrid_risk_calculator import HybridRiskCalculator
from src.loan_simulation.loan_impact_simulator import LoanImpactSimulator

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SeoulMarketRiskSystem:
    """서울시장 위험도 평가 시스템 메인 클래스"""

    def __init__(self):
        self.expense_estimator = ExpenseEstimator()
        self.risk_calculator = HybridRiskCalculator()
        self.loan_simulator = LoanImpactSimulator()

        logger.info("Seoul Market Risk System v2.0 초기화 완료")

    def prepare_data(self) -> None:
        """데이터 전처리 - 지출 컬럼 추가"""
        logger.info("데이터 전처리 시작...")

        processed_files = self.expense_estimator.process_all_csv_files()
        logger.info(f"처리 완료: {len(processed_files)}개 파일")

    def analyze_business(self,
                        business_id: str,
                        revenue_history: List[float],
                        operating_assets: float,
                        industry_code: str,
                        months_in_business: int = None) -> Dict:
        """개별 사업자 위험도 분석"""

        # 지출 추정
        expense_history = [r * 0.7544867193 for r in revenue_history]

        # 위험도 평가
        assessment = self.risk_calculator.calculate_risk_assessment(
            business_id=business_id,
            revenue_history=revenue_history,
            expense_history=expense_history,
            operating_assets=operating_assets,
            industry_code=industry_code,
            months_in_business=months_in_business
        )

        # 대출 시뮬레이션 (위험군 이상인 경우)
        loan_simulation = None
        if assessment.total_risk_score <= 60:  # 적정 미만
            target_score = 60  # 적정 수준 목표
            loan_simulation = self.loan_simulator.run_comprehensive_simulation(
                current_risk_score=assessment.total_risk_score,
                target_risk_score=target_score,
                monthly_revenue=revenue_history[-1],
                current_assets=operating_assets
            )

        return {
            'assessment': assessment,
            'loan_simulation': loan_simulation,
            'recommendations': self._generate_recommendations(assessment, loan_simulation)
        }

    def _generate_recommendations(self, assessment, loan_simulation) -> List[str]:
        """맞춤형 추천 사항 생성"""
        recommendations = []

        # 위험도 등급별 기본 추천
        if assessment.risk_level == "매우위험":
            recommendations.append("💰 긴급 운영자금 지원이 필요합니다")
            recommendations.append("📊 즉시 사업 구조조정 검토가 필요합니다")
        elif assessment.risk_level == "위험군":
            recommendations.append("💳 안정화 대출을 통한 재무구조 개선 권장")
            recommendations.append("📈 매출 다각화 전략 수립이 필요합니다")
        elif assessment.risk_level == "적정":
            recommendations.append("📊 현재 상태 유지 및 정기적 모니터링")
            recommendations.append("💡 성장 기회 발굴을 위한 시장 분석 권장")
        elif assessment.risk_level == "좋음":
            recommendations.append("🚀 성장투자 기회 적극 활용 권장")
            recommendations.append("💼 사업 확장 또는 신규 투자 검토")
        else:  # 매우좋음
            recommendations.append("💎 프리미엄 투자상품 활용 검토")
            recommendations.append("🌟 신사업 진출 기회 모색")

        # 대출 관련 추천
        if loan_simulation:
            recommendations.append(f"💰 {loan_simulation.loan_amount:,.0f}원 대출로 위험도 {loan_simulation.score_improvement:.1f}점 개선 가능")
            recommendations.append(f"📈 월 상환액: {loan_simulation.monthly_payment:,.0f}원 ({loan_simulation.recommendation})")

        return recommendations

def demo_analysis():
    """시스템 데모"""
    print("\n" + "="*60)
    print("Seoul Market Risk ML System v2.0 - 하이브리드 모델")
    print("="*60)

    # 시스템 초기화
    system = SeoulMarketRiskSystem()

    # 샘플 사업자 데이터
    sample_business = {
        'business_id': 'DEMO_001',
        'revenue_history': [5000000, 5200000, 4800000, 5100000, 5300000, 5150000],  # 6개월 매출
        'operating_assets': 25000000,  # 운용자산 2500만원
        'industry_code': 'CS100001',  # 한식음식점
        'months_in_business': 24  # 2년 운영
    }

    print(f"\n📊 사업자 분석: {sample_business['business_id']}")
    print(f"   업종: 한식음식점")
    print(f"   운영기간: {sample_business['months_in_business']}개월")
    print(f"   운용자산: {sample_business['operating_assets']:,}원")
    print(f"   최근매출: {sample_business['revenue_history'][-1]:,}원")

    # 위험도 분석 실행
    result = system.analyze_business(**sample_business)
    assessment = result['assessment']
    loan_simulation = result['loan_simulation']

    # 결과 출력
    print(f"\n🎯 위험도 평가 결과:")
    print(f"   총점: {assessment.total_risk_score:.1f}점")
    print(f"   등급: {assessment.risk_level}")
    print(f"   신뢰도: {assessment.confidence:.1f}%")

    print(f"\n📈 구성 요소별 점수:")
    print(f"   재무건전성 (40%): {assessment.financial_health_score:.1f}점 (Z-Score: {assessment.altman_zscore:.2f})")
    print(f"   영업안정성 (45%): {assessment.operational_stability_score:.1f}점")
    print(f"   업종내위치 (15%): {assessment.industry_position_score:.1f}점")

    # 대출 추천
    if loan_simulation:
        print(f"\n💰 대출 추천:")
        print(f"   추천금액: {loan_simulation.loan_amount:,.0f}원")
        print(f"   예상개선: {loan_simulation.score_improvement:.1f}점")
        print(f"   개선후등급: {loan_simulation.new_risk_level}")
        print(f"   월상환액: {loan_simulation.monthly_payment:,.0f}원")
        print(f"   투자회수: {loan_simulation.roi_months}개월")
    else:
        print(f"\n✅ 대출 불필요 - 현재 재무상태 양호")

    print(f"\n📋 맞춤 추천사항:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"   {i}. {rec}")

    print(f"\n" + "="*60)
    print("분석 완료 - 하이브리드 모델 시스템이 정상 작동합니다")
    print("="*60)

if __name__ == "__main__":
    demo_analysis()