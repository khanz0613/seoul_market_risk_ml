"""
Hybrid Risk Calculator for Seoul Market Risk ML System
하이브리드 위험도 산정 모델 v2.0

구성:
- 재무 건전성 (40%): Altman Z'-Score 기반
- 영업 안정성 (45%): 매출 트렌드, 변동성, 지속성
- 상대적 위치 (15%): 업종 내 경쟁력
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass

from ..financial_analysis.altman_zscore import AltmanZScoreCalculator
from ..financial_analysis.operational_stability import OperationalStabilityCalculator
from ..financial_analysis.industry_comparison import IndustryComparisonCalculator

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessmentResult:
    """위험도 평가 결과"""
    business_id: str
    total_risk_score: float
    risk_level: str
    confidence: float

    # 구성 요소 점수
    financial_health_score: float      # 40%
    operational_stability_score: float # 45%
    industry_position_score: float     # 15%

    # 세부 분석
    altman_zscore: float
    recommended_action: str
    loan_amount_needed: float

    # 메타데이터
    assessment_date: str
    data_quality_score: float

class HybridRiskCalculator:
    """하이브리드 위험도 계산기"""

    def __init__(self):
        # 구성 요소별 가중치
        self.weights = {
            'financial_health': 0.40,     # 재무 건전성 40%
            'operational_stability': 0.45, # 영업 안정성 45%
            'industry_position': 0.15     # 상대적 위치 15%
        }

        # 각 계산기 초기화
        self.altman_calculator = AltmanZScoreCalculator()
        self.operational_calculator = OperationalStabilityCalculator()
        self.industry_calculator = IndustryComparisonCalculator()

        # 위험도 등급 기준
        self.risk_levels = {
            (0, 20): "매우위험",
            (21, 40): "위험군",
            (41, 60): "적정",
            (61, 80): "좋음",
            (81, 100): "매우좋음"
        }

        logger.info("Hybrid Risk Calculator 초기화 완료")

    def calculate_financial_health_score(self,
                                       revenue: float,
                                       expenses: float,
                                       operating_assets: float,
                                       months_in_business: int = 12) -> Dict[str, float]:
        """재무 건전성 점수 계산 (Altman Z'-Score 기반)"""

        try:
            # Z'-Score 계산
            zscore_result = self.altman_calculator.calculate_from_business_data(
                revenue, expenses, operating_assets, months_in_business)

            # Z'-Score를 0-100 점수로 변환
            zscore = zscore_result['zscore']
            if zscore >= 2.99:
                score = 90 + min(10, (zscore - 2.99) * 5)  # 90-100점
            elif zscore >= 1.81:
                score = 60 + ((zscore - 1.81) / (2.99 - 1.81)) * 30  # 60-90점
            else:
                score = max(0, (zscore / 1.81) * 60)  # 0-60점

            return {
                'financial_health_score': score,
                'zscore': zscore,
                'zscore_components': zscore_result['components'],
                'interpretation': zscore_result['interpretation']
            }

        except Exception as e:
            logger.error(f"Financial health calculation failed: {e}")
            return {
                'financial_health_score': 50.0,  # 기본값
                'zscore': 1.8,
                'zscore_components': {},
                'interpretation': {'risk_level': '불명', 'description': '계산 오류'}
            }

    def calculate_risk_assessment(self,
                                business_id: str,
                                revenue_history: List[float],
                                expense_history: List[float],
                                operating_assets: float,
                                industry_code: str,
                                months_in_business: int = None) -> RiskAssessmentResult:
        """종합 위험도 평가"""

        if months_in_business is None:
            months_in_business = len(revenue_history)

        # 최신 재무 데이터
        current_revenue = revenue_history[-1] if revenue_history else 0
        current_expenses = expense_history[-1] if expense_history else 0

        # 1. 재무 건전성 (40%)
        financial_health = self.calculate_financial_health_score(
            current_revenue, current_expenses, operating_assets, months_in_business)

        # 2. 영업 안정성 (45%)
        operational_stability = self.operational_calculator.calculate_operational_stability_score(
            revenue_history, months_in_business)

        # 3. 업종 내 위치 (15%)
        industry_position = self.industry_calculator.calculate_industry_position_score(
            current_revenue, current_expenses, industry_code)

        # 가중 평균 점수 계산
        total_risk_score = (
            financial_health['financial_health_score'] * self.weights['financial_health'] +
            operational_stability['operational_stability_score'] * self.weights['operational_stability'] +
            industry_position['industry_position_score'] * self.weights['industry_position']
        )

        # 위험도 등급 결정
        risk_level = self._determine_risk_level(total_risk_score)

        # 추천 액션 및 대출 금액 계산
        recommended_action, loan_amount = self._calculate_loan_recommendation(
            total_risk_score, current_revenue, financial_health['zscore'])

        # 데이터 품질 점수
        data_quality_score = self._calculate_data_quality_score(
            revenue_history, expense_history, operating_assets)

        # 신뢰도 점수
        confidence = self._calculate_confidence_score(
            data_quality_score, months_in_business, len(revenue_history))

        return RiskAssessmentResult(
            business_id=business_id,
            total_risk_score=total_risk_score,
            risk_level=risk_level,
            confidence=confidence,
            financial_health_score=financial_health['financial_health_score'],
            operational_stability_score=operational_stability['operational_stability_score'],
            industry_position_score=industry_position['industry_position_score'],
            altman_zscore=financial_health['zscore'],
            recommended_action=recommended_action,
            loan_amount_needed=loan_amount,
            assessment_date=datetime.now().isoformat(),
            data_quality_score=data_quality_score
        )

    def _determine_risk_level(self, score: float) -> str:
        """점수 기반 위험도 등급 결정"""
        for (min_score, max_score), level in self.risk_levels.items():
            if min_score <= score <= max_score:
                return level
        return "적정"  # 기본값

    def _calculate_loan_recommendation(self,
                                     current_score: float,
                                     monthly_revenue: float,
                                     zscore: float) -> Tuple[str, float]:
        """대출 추천 및 필요 금액 계산"""

        if current_score <= 20:  # 매우위험
            target_score = 40  # 위험군까지 개선
            score_gap = target_score - current_score
            loan_amount = monthly_revenue * (score_gap / 100.0) * 3.0
            return "emergency_loan", loan_amount

        elif current_score <= 40:  # 위험군
            target_score = 60  # 적정까지 개선
            score_gap = target_score - current_score
            loan_amount = monthly_revenue * (score_gap / 100.0) * 2.0
            return "stabilization_loan", loan_amount

        elif current_score <= 60:  # 적정
            return "monitoring", 0.0

        elif current_score <= 80:  # 좋음
            return "growth_investment", 0.0

        else:  # 매우좋음
            return "premium_investment", 0.0

    def _calculate_data_quality_score(self,
                                    revenue_history: List[float],
                                    expense_history: List[float],
                                    operating_assets: float) -> float:
        """데이터 품질 점수 계산"""

        quality_factors = []

        # 매출 데이터 품질
        if len(revenue_history) >= 6:  # 6개월 이상 데이터
            quality_factors.append(0.9)
        elif len(revenue_history) >= 3:  # 3개월 이상
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)

        # 지출 데이터 품질 (추정값이므로 낮은 가중치)
        quality_factors.append(0.6)

        # 운용자산 데이터 (사용자 입력)
        if operating_assets > 0:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.2)

        return np.mean(quality_factors) * 100

    def _calculate_confidence_score(self,
                                  data_quality_score: float,
                                  months_in_business: int,
                                  data_points: int) -> float:
        """신뢰도 점수 계산"""

        # 데이터 품질 (40%)
        quality_component = data_quality_score * 0.4

        # 사업 경험 (30%)
        experience_component = min(100, (months_in_business / 36.0) * 100) * 0.3

        # 데이터 충분성 (30%)
        data_sufficiency = min(100, (data_points / 12.0) * 100) * 0.3

        return quality_component + experience_component + data_sufficiency

    def simulate_loan_impact(self,
                           current_assessment: RiskAssessmentResult,
                           loan_amount: float) -> Dict[str, float]:
        """대출 후 위험도 변화 시뮬레이션"""

        # 대출로 운용자산 증가 효과 계산
        # 간소화된 시뮬레이션 - 실제로는 더 복잡한 모델 필요

        # 운용자산 증가로 인한 Z'-Score 개선 추정
        zscore_improvement = (loan_amount / 10000000) * 0.3  # 1000만원당 0.3점 개선

        # 새로운 재무 건전성 점수 계산
        new_zscore = current_assessment.altman_zscore + zscore_improvement
        new_financial_score = min(100, current_assessment.financial_health_score + zscore_improvement * 20)

        # 새로운 총 점수
        new_total_score = (
            new_financial_score * self.weights['financial_health'] +
            current_assessment.operational_stability_score * self.weights['operational_stability'] +
            current_assessment.industry_position_score * self.weights['industry_position']
        )

        return {
            'original_score': current_assessment.total_risk_score,
            'projected_score': new_total_score,
            'score_improvement': new_total_score - current_assessment.total_risk_score,
            'new_risk_level': self._determine_risk_level(new_total_score),
            'loan_amount': loan_amount,
            'new_zscore': new_zscore
        }