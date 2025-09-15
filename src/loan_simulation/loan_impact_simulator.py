"""
Loan Impact Simulator
대출 영향도 시뮬레이션 엔진
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LoanSimulationResult:
    """대출 시뮬레이션 결과"""
    loan_amount: float
    current_risk_score: float
    projected_risk_score: float
    score_improvement: float
    new_risk_level: str
    monthly_payment: float
    roi_months: int
    recommendation: str

class LoanImpactSimulator:
    """대출 영향도 시뮬레이터"""

    def __init__(self):
        # 대출 조건 설정 (기본값)
        self.loan_conditions = {
            'interest_rate': 0.035,  # 연 3.5% (소상공인 정책자금 기준)
            'loan_term_months': 60,  # 5년
            'processing_fee': 0.005  # 0.5% 수수료
        }

    def calculate_loan_payment(self,
                             loan_amount: float,
                             annual_interest_rate: float = None,
                             term_months: int = None) -> Dict[str, float]:
        """대출 월 상환액 계산"""

        if annual_interest_rate is None:
            annual_interest_rate = self.loan_conditions['interest_rate']
        if term_months is None:
            term_months = self.loan_conditions['loan_term_months']

        # 월 이자율
        monthly_rate = annual_interest_rate / 12

        if monthly_rate == 0:
            monthly_payment = loan_amount / term_months
        else:
            # 원리금균등상환 공식
            monthly_payment = loan_amount * (
                monthly_rate * (1 + monthly_rate) ** term_months
            ) / ((1 + monthly_rate) ** term_months - 1)

        total_payment = monthly_payment * term_months
        total_interest = total_payment - loan_amount

        return {
            'monthly_payment': monthly_payment,
            'total_payment': total_payment,
            'total_interest': total_interest,
            'effective_rate': total_interest / loan_amount
        }

    def simulate_risk_score_improvement(self,
                                      current_risk_score: float,
                                      loan_amount: float,
                                      monthly_revenue: float,
                                      current_assets: float) -> Dict[str, float]:
        """대출 후 위험도 점수 개선 시뮬레이션"""

        # 운용자산 증가 효과
        new_assets = current_assets + loan_amount
        asset_improvement_factor = loan_amount / current_assets if current_assets > 0 else 0

        # Z'-Score 개선 추정 (운전자본 비율 개선)
        zscore_improvement = min(15, asset_improvement_factor * 25)  # 최대 15점 개선

        # 현금흐름 개선 효과
        cash_flow_improvement = min(10, (loan_amount / monthly_revenue) * 2) if monthly_revenue > 0 else 0

        # 총 점수 개선
        total_improvement = zscore_improvement + cash_flow_improvement

        # 개선된 점수 (100점 상한)
        new_risk_score = min(100, current_risk_score + total_improvement)

        return {
            'score_improvement': total_improvement,
            'new_risk_score': new_risk_score,
            'zscore_improvement': zscore_improvement,
            'cash_flow_improvement': cash_flow_improvement,
            'improvement_ratio': total_improvement / current_risk_score if current_risk_score > 0 else 0
        }

    def calculate_optimal_loan_amount(self,
                                    current_risk_score: float,
                                    target_risk_score: float,
                                    monthly_revenue: float,
                                    current_assets: float,
                                    max_debt_ratio: float = 0.4) -> Dict[str, float]:
        """최적 대출 금액 계산"""

        if current_risk_score >= target_risk_score:
            return {
                'optimal_loan_amount': 0,
                'reason': '이미 목표 점수 달성',
                'current_score': current_risk_score,
                'target_score': target_risk_score
            }

        score_gap = target_risk_score - current_risk_score

        # 점수 개선에 필요한 대출 금액 추정
        # 경험식: 점수 1점 개선에 필요한 대출 = 월매출의 0.5배
        estimated_loan = monthly_revenue * score_gap * 0.5

        # 상환 능력 제한 (월 매출의 40% 이하로 상환액 제한)
        max_monthly_payment = monthly_revenue * max_debt_ratio
        max_affordable_loan = self._calculate_max_loan_from_payment(max_monthly_payment)

        # 최종 대출 금액 (둘 중 작은 값)
        optimal_loan = min(estimated_loan, max_affordable_loan)

        return {
            'optimal_loan_amount': optimal_loan,
            'estimated_loan': estimated_loan,
            'max_affordable_loan': max_affordable_loan,
            'constraint': 'affordability' if optimal_loan == max_affordable_loan else 'target_score',
            'max_monthly_payment': max_monthly_payment
        }

    def _calculate_max_loan_from_payment(self, max_monthly_payment: float) -> float:
        """월 상환 가능 금액으로부터 최대 대출 금액 계산"""
        monthly_rate = self.loan_conditions['interest_rate'] / 12
        term_months = self.loan_conditions['loan_term_months']

        if monthly_rate == 0:
            return max_monthly_payment * term_months

        # 원리금균등상환 공식 역산
        max_loan = max_monthly_payment * (
            ((1 + monthly_rate) ** term_months - 1) /
            (monthly_rate * (1 + monthly_rate) ** term_months)
        )

        return max_loan

    def run_comprehensive_simulation(self,
                                   current_risk_score: float,
                                   target_risk_score: float,
                                   monthly_revenue: float,
                                   current_assets: float) -> LoanSimulationResult:
        """종합 대출 시뮬레이션"""

        # 최적 대출 금액 계산
        loan_calc = self.calculate_optimal_loan_amount(
            current_risk_score, target_risk_score, monthly_revenue, current_assets)

        optimal_loan = loan_calc['optimal_loan_amount']

        if optimal_loan == 0:
            return LoanSimulationResult(
                loan_amount=0,
                current_risk_score=current_risk_score,
                projected_risk_score=current_risk_score,
                score_improvement=0,
                new_risk_level=self._get_risk_level(current_risk_score),
                monthly_payment=0,
                roi_months=0,
                recommendation="대출 불필요"
            )

        # 위험도 개선 시뮬레이션
        score_simulation = self.simulate_risk_score_improvement(
            current_risk_score, optimal_loan, monthly_revenue, current_assets)

        # 대출 조건 계산
        payment_info = self.calculate_loan_payment(optimal_loan)

        # ROI 계산 (개선 효과가 대출 비용을 상회하는 시점)
        monthly_benefit = score_simulation['score_improvement'] * monthly_revenue * 0.01  # 점수 1점당 1% 매출 증가 가정
        if monthly_benefit > payment_info['monthly_payment']:
            roi_months = int(optimal_loan / (monthly_benefit - payment_info['monthly_payment']))
        else:
            roi_months = 999  # 투자 회수 어려움

        # 추천 메시지
        recommendation = self._generate_recommendation(
            score_simulation['score_improvement'],
            payment_info['monthly_payment'],
            monthly_revenue,
            roi_months
        )

        return LoanSimulationResult(
            loan_amount=optimal_loan,
            current_risk_score=current_risk_score,
            projected_risk_score=score_simulation['new_risk_score'],
            score_improvement=score_simulation['score_improvement'],
            new_risk_level=self._get_risk_level(score_simulation['new_risk_score']),
            monthly_payment=payment_info['monthly_payment'],
            roi_months=roi_months,
            recommendation=recommendation
        )

    def _get_risk_level(self, score: float) -> str:
        """점수를 위험도 등급으로 변환"""
        if score >= 81:
            return "매우좋음"
        elif score >= 61:
            return "좋음"
        elif score >= 41:
            return "적정"
        elif score >= 21:
            return "위험군"
        else:
            return "매우위험"

    def _generate_recommendation(self,
                               score_improvement: float,
                               monthly_payment: float,
                               monthly_revenue: float,
                               roi_months: int) -> str:
        """추천 메시지 생성"""

        payment_ratio = monthly_payment / monthly_revenue if monthly_revenue > 0 else 0

        if score_improvement < 5:
            return "대출 효과 미미 - 다른 개선 방안 검토 권장"
        elif payment_ratio > 0.3:
            return "상환 부담이 큼 - 소액 대출부터 시작 권장"
        elif roi_months <= 12:
            return "우수한 투자 - 적극 추천"
        elif roi_months <= 24:
            return "양호한 투자 - 추천"
        elif roi_months <= 36:
            return "신중한 검토 후 결정 권장"
        else:
            return "투자 효과 불확실 - 재검토 필요"