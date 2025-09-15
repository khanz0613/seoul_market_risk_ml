"""
ML 기반 Seoul Market Risk System v2.0
학습된 머신러닝 모델을 사용한 실제 예측 시스템
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from src.ml_pipeline.predictor import HybridRiskPredictor
from src.loan_simulation.loan_impact_simulator import LoanImpactSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLSeoulMarketRiskSystem:
    """머신러닝 기반 위험도 평가 시스템"""

    def __init__(self):
        self.predictor = HybridRiskPredictor()
        self.loan_simulator = LoanImpactSimulator()

        logger.info("ML Seoul Market Risk System v2.0 초기화 완료")

    def analyze_business_with_ml(self,
                                business_id: str,
                                revenue_history: List[float],
                                operating_assets: float,
                                industry_code: str,
                                months_in_business: int = None) -> Dict:
        """머신러닝 모델 기반 사업자 분석"""

        # 피처 준비 (기존 계산기 로직 활용)
        if months_in_business is None:
            months_in_business = len(revenue_history)

        # 운영 지표 계산
        avg_revenue = np.mean(revenue_history)
        revenue_growth = 0.0
        revenue_cv = 0.0

        if len(revenue_history) >= 2:
            growth_rates = []
            for i in range(1, len(revenue_history)):
                if revenue_history[i-1] > 0:
                    growth = (revenue_history[i] - revenue_history[i-1]) / revenue_history[i-1]
                    growth_rates.append(growth)
            revenue_growth = np.mean(growth_rates) if growth_rates else 0.0

        if avg_revenue > 0:
            revenue_cv = np.std(revenue_history) / avg_revenue

        # Altman Z-Score 추정
        estimated_profit = avg_revenue * 0.15  # 15% 순이익률 가정
        estimated_assets = operating_assets * 1.5
        altman_zscore = max(0.5, min(5.0, 1.2 + (estimated_profit / estimated_assets) * 10))

        # ML 모델 입력 데이터
        ml_input = {
            'business_id': business_id,
            'latest_revenue': revenue_history[-1],
            'latest_profit_margin': 0.15,  # 기본 15%
            'avg_growth_rate': revenue_growth,
            'revenue_cv': revenue_cv,
            'business_quarters': months_in_business // 3,
            'quarters_active': len(revenue_history),
            'altman_zscore': altman_zscore,
            'working_capital_ratio': 0.1,
            'retained_earnings_ratio': 0.05,
            'ebit_ratio': 0.03,
            'equity_debt_ratio': 1.5,
            'asset_turnover': avg_revenue / estimated_assets,
            'revenue_consistency': sum(1 for r in revenue_history if r > 0) / len(revenue_history),
            'industry_percentile_revenue': 50,  # 기본값
            'industry_percentile_profit': 50
        }

        # ML 예측 실행
        ml_result = self.predictor.comprehensive_prediction(ml_input)

        # 대출 시뮬레이션 (위험군 이상인 경우)
        loan_simulation = None
        if ml_result['risk_score'] <= 60:
            target_score = 60
            loan_simulation = self.loan_simulator.run_comprehensive_simulation(
                current_risk_score=ml_result['risk_score'],
                target_risk_score=target_score,
                monthly_revenue=revenue_history[-1],
                current_assets=operating_assets
            )

        return {
            'ml_prediction': ml_result,
            'loan_simulation': loan_simulation,
            'input_features': ml_input,
            'recommendations': self._generate_ml_recommendations(ml_result, loan_simulation)
        }

    def _generate_ml_recommendations(self, ml_result, loan_simulation) -> List[str]:
        """ML 예측 기반 추천 생성"""
        recommendations = []

        risk_score = ml_result['risk_score']
        risk_level = ml_result['risk_level']

        # 기본 추천
        if risk_level == "매우위험":
            recommendations.extend([
                f"🚨 ML 모델 예측: 매우 높은 위험도 ({risk_score:.1f}점)",
                "💰 긴급 자금 지원이 필요합니다",
                "📊 즉시 사업 재구조화 검토 권장"
            ])
        elif risk_level == "위험군":
            recommendations.extend([
                f"⚠️ ML 모델 예측: 위험 신호 감지 ({risk_score:.1f}점)",
                "💳 안정화 대출을 통한 개선 필요",
                "📈 매출 안정성 확보 방안 모색"
            ])
        elif risk_level == "적정":
            recommendations.extend([
                f"✅ ML 모델 예측: 안정적 상태 ({risk_score:.1f}점)",
                "📊 현재 상태 유지 및 성장 기회 모색"
            ])
        elif risk_level == "좋음":
            recommendations.extend([
                f"🎯 ML 모델 예측: 양호한 상태 ({risk_score:.1f}점)",
                "🚀 적극적인 성장 투자 검토"
            ])
        else:  # 매우좋음
            recommendations.extend([
                f"⭐ ML 모델 예측: 매우 우수 ({risk_score:.1f}점)",
                "💎 고수익 투자 기회 적극 활용"
            ])

        # 대출 관련 추천
        if loan_simulation:
            recommendations.append(
                f"💰 {loan_simulation.loan_amount:,.0f}원 대출로 "
                f"{loan_simulation.score_improvement:.1f}점 개선 예상"
            )

        # 예측 신뢰도
        confidence = ml_result['confidence']
        if confidence >= 80:
            recommendations.append(f"🎯 예측 신뢰도: 매우 높음 ({confidence:.0f}%)")
        elif confidence >= 60:
            recommendations.append(f"🎯 예측 신뢰도: 보통 ({confidence:.0f}%)")
        else:
            recommendations.append(f"⚠️ 예측 신뢰도: 낮음 ({confidence:.0f}%) - 추가 데이터 필요")

        return recommendations


def ml_demo():
    """머신러닝 시스템 데모"""
    print("\n" + "="*70)
    print("Seoul Market Risk ML System v2.0 - 머신러닝 모델")
    print("="*70)

    # 시스템 초기화
    ml_system = MLSeoulMarketRiskSystem()

    # 샘플 사업자 (위험군)
    risky_business = {
        'business_id': 'ML_RISKY_001',
        'revenue_history': [3000000, 2800000, 2500000, 2200000, 2000000, 1800000],  # 하락 추세
        'operating_assets': 15000000,
        'industry_code': 'CS100001',
        'months_in_business': 18
    }

    print(f"\n📊 위험 사업자 분석: {risky_business['business_id']}")
    print(f"   매출 추이: {risky_business['revenue_history'][-1]:,}원 (하락)")
    print(f"   운용자산: {risky_business['operating_assets']:,}원")

    risky_result = ml_system.analyze_business_with_ml(**risky_business)
    ml_pred = risky_result['ml_prediction']

    print(f"\n🤖 ML 예측 결과:")
    print(f"   위험도 점수: {ml_pred['risk_score']:.1f}점")
    print(f"   위험도 등급: {ml_pred['risk_level']}")
    print(f"   위험 확률: {ml_pred['risk_probability']:.1%}")
    print(f"   예측 신뢰도: {ml_pred['confidence']:.0f}%")

    if risky_result['loan_simulation']:
        loan_sim = risky_result['loan_simulation']
        print(f"\n💰 대출 시뮬레이션:")
        print(f"   추천 금액: {loan_sim.loan_amount:,.0f}원")
        print(f"   예상 개선: {loan_sim.score_improvement:.1f}점")
        print(f"   월 상환액: {loan_sim.monthly_payment:,.0f}원")

    print(f"\n📋 AI 추천사항:")
    for i, rec in enumerate(risky_result['recommendations'], 1):
        print(f"   {i}. {rec}")

    # 샘플 사업자 (우수군)
    print(f"\n" + "-"*70)

    good_business = {
        'business_id': 'ML_GOOD_001',
        'revenue_history': [4000000, 4200000, 4500000, 4800000, 5100000, 5400000],  # 상승 추세
        'operating_assets': 30000000,
        'industry_code': 'CS100001',
        'months_in_business': 24
    }

    print(f"\n📊 우수 사업자 분석: {good_business['business_id']}")
    print(f"   매출 추이: {good_business['revenue_history'][-1]:,}원 (상승)")
    print(f"   운용자산: {good_business['operating_assets']:,}원")

    good_result = ml_system.analyze_business_with_ml(**good_business)
    good_pred = good_result['ml_prediction']

    print(f"\n🤖 ML 예측 결과:")
    print(f"   위험도 점수: {good_pred['risk_score']:.1f}점")
    print(f"   위험도 등급: {good_pred['risk_level']}")
    print(f"   위험 확률: {good_pred['risk_probability']:.1%}")
    print(f"   예측 신뢰도: {good_pred['confidence']:.0f}%")

    print(f"\n📋 AI 추천사항:")
    for i, rec in enumerate(good_result['recommendations'], 1):
        print(f"   {i}. {rec}")

    print(f"\n" + "="*70)
    print("✅ 머신러닝 기반 위험도 평가 시스템 데모 완료")
    print("🎯 실제 학습된 모델이 있으면 더 정확한 예측 가능")
    print("="*70)


if __name__ == "__main__":
    ml_demo()