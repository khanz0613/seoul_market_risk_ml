"""
Altman Z'-Score Calculator for Small Business Risk Assessment
소상공인 맞춤형 재무 건전성 평가
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AltmanZScoreCalculator:
    """Altman Z'-Score 계산기 (소상공인 맞춤형)"""

    def __init__(self):
        # Altman Z'-Score 가중치 (소상공인용 수정 버전)
        self.weights = {
            'working_capital_to_assets': 6.56,    # X1: 운전자본/총자산
            'retained_earnings_to_assets': 3.26,  # X2: 이익잉여금/총자산
            'ebit_to_assets': 6.72,              # X3: 세전이익/총자산
            'equity_to_debt': 1.05               # X4: 자기자본/총부채
        }

    def calculate_working_capital_ratio(self, current_assets: float,
                                      current_liabilities: float,
                                      total_assets: float) -> float:
        """X1: 운전자본/총자산 비율"""
        if total_assets <= 0:
            return 0.0
        working_capital = current_assets - current_liabilities
        return working_capital / total_assets

    def calculate_retained_earnings_ratio(self, retained_earnings: float,
                                        total_assets: float) -> float:
        """X2: 이익잉여금/총자산 비율"""
        if total_assets <= 0:
            return 0.0
        return retained_earnings / total_assets

    def calculate_ebit_ratio(self, ebit: float, total_assets: float) -> float:
        """X3: 세전이익/총자산 비율"""
        if total_assets <= 0:
            return 0.0
        return ebit / total_assets

    def calculate_equity_to_debt_ratio(self, equity: float, total_debt: float) -> float:
        """X4: 자기자본/총부채 비율"""
        if total_debt <= 0:
            return 10.0  # 부채가 없으면 매우 안전
        return equity / total_debt

    def calculate_zscore(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Altman Z'-Score 계산"""

        # 필수 데이터 검증
        required_fields = ['current_assets', 'current_liabilities', 'total_assets',
                          'retained_earnings', 'ebit', 'equity', 'total_debt']

        for field in required_fields:
            if field not in financial_data or financial_data[field] is None:
                raise ValueError(f"Required field missing: {field}")

        # 각 구성 요소 계산
        x1 = self.calculate_working_capital_ratio(
            financial_data['current_assets'],
            financial_data['current_liabilities'],
            financial_data['total_assets']
        )

        x2 = self.calculate_retained_earnings_ratio(
            financial_data['retained_earnings'],
            financial_data['total_assets']
        )

        x3 = self.calculate_ebit_ratio(
            financial_data['ebit'],
            financial_data['total_assets']
        )

        x4 = self.calculate_equity_to_debt_ratio(
            financial_data['equity'],
            financial_data['total_debt']
        )

        # Z'-Score 계산
        zscore = (self.weights['working_capital_to_assets'] * x1 +
                 self.weights['retained_earnings_to_assets'] * x2 +
                 self.weights['ebit_to_assets'] * x3 +
                 self.weights['equity_to_debt'] * x4)

        # 결과 반환
        return {
            'zscore': zscore,
            'components': {
                'x1_working_capital_ratio': x1,
                'x2_retained_earnings_ratio': x2,
                'x3_ebit_ratio': x3,
                'x4_equity_to_debt_ratio': x4
            },
            'weighted_components': {
                'x1_weighted': self.weights['working_capital_to_assets'] * x1,
                'x2_weighted': self.weights['retained_earnings_to_assets'] * x2,
                'x3_weighted': self.weights['ebit_to_assets'] * x3,
                'x4_weighted': self.weights['equity_to_debt'] * x4
            },
            'interpretation': self.interpret_zscore(zscore)
        }

    def interpret_zscore(self, zscore: float) -> Dict[str, str]:
        """Z'-Score 해석"""
        if zscore >= 2.99:
            risk_level = "안전"
            description = "재무적으로 매우 안정적"
        elif zscore >= 1.81:
            risk_level = "주의"
            description = "재무상태 모니터링 필요"
        else:
            risk_level = "위험"
            description = "재무적 어려움 예상"

        return {
            'risk_level': risk_level,
            'description': description,
            'score': zscore
        }

    def calculate_from_business_data(self, revenue: float, expenses: float,
                                   operating_assets: float,
                                   months_of_operation: int = 12) -> Dict[str, float]:
        """사업 데이터로부터 Z'-Score 추정"""

        # 기본 재무 수치 추정
        monthly_profit = revenue - expenses
        annual_profit = monthly_profit * 12

        # 추정 재무상태표 생성
        estimated_total_assets = operating_assets * 1.5  # 운용자산의 1.5배로 추정
        estimated_current_liabilities = operating_assets * 0.3  # 30%로 추정
        estimated_total_debt = estimated_current_liabilities * 1.2  # 20% 마진
        estimated_equity = estimated_total_assets - estimated_total_debt

        # 이익잉여금 추정 (운영 기간 기반)
        estimated_retained_earnings = annual_profit * (months_of_operation / 12.0) * 0.7

        financial_data = {
            'current_assets': operating_assets,
            'current_liabilities': estimated_current_liabilities,
            'total_assets': estimated_total_assets,
            'retained_earnings': estimated_retained_earnings,
            'ebit': annual_profit,  # 간소화: 순이익을 EBIT로 사용
            'equity': estimated_equity,
            'total_debt': estimated_total_debt
        }

        return self.calculate_zscore(financial_data)