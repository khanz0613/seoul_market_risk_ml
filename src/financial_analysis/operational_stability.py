"""
Operational Stability Calculator
영업 안정성 평가 (45% 가중치)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OperationalStabilityCalculator:
    """영업 안정성 계산기"""

    def __init__(self):
        # 영업 안정성 구성 요소 가중치
        self.weights = {
            'revenue_growth': 0.20,      # 매출 성장성 20%
            'revenue_volatility': 0.15,  # 매출 변동성 15%
            'business_continuity': 0.10  # 영업 지속성 10%
        }

    def calculate_revenue_growth_trend(self, revenue_data: List[float],
                                     periods: int = 3) -> Dict[str, float]:
        """매출 성장성 트렌드 분석 (최근 N개월 평균)"""
        if len(revenue_data) < periods + 1:
            return {'growth_rate': 0.0, 'trend_score': 50.0}

        recent_data = revenue_data[-periods-1:]

        # 성장률 계산
        growth_rates = []
        for i in range(1, len(recent_data)):
            if recent_data[i-1] > 0:
                growth_rate = (recent_data[i] - recent_data[i-1]) / recent_data[i-1]
                growth_rates.append(growth_rate)

        if not growth_rates:
            avg_growth_rate = 0.0
        else:
            avg_growth_rate = np.mean(growth_rates)

        # 점수화 (성장률을 0-100점으로 변환)
        # 월 5% 성장 = 100점, 0% = 50점, -5% = 0점
        trend_score = max(0, min(100, 50 + (avg_growth_rate * 1000)))

        return {
            'growth_rate': avg_growth_rate,
            'trend_score': trend_score,
            'periods_analyzed': len(growth_rates)
        }

    def calculate_revenue_volatility(self, revenue_data: List[float]) -> Dict[str, float]:
        """매출 변동성 분석"""
        if len(revenue_data) < 2:
            return {'volatility': 0.0, 'volatility_score': 100.0}

        # 변동계수 계산 (표준편차/평균)
        mean_revenue = np.mean(revenue_data)
        if mean_revenue <= 0:
            return {'volatility': 1.0, 'volatility_score': 0.0}

        std_revenue = np.std(revenue_data)
        cv = std_revenue / mean_revenue  # 변동계수

        # 점수화 (낮은 변동성 = 높은 점수)
        # CV 0.1 = 100점, CV 0.5 = 0점
        volatility_score = max(0, min(100, 100 - (cv * 250)))

        return {
            'volatility': cv,
            'volatility_score': volatility_score,
            'mean_revenue': mean_revenue,
            'std_revenue': std_revenue
        }

    def calculate_business_continuity(self, months_in_business: int,
                                    revenue_consistency_months: int) -> Dict[str, float]:
        """영업 지속성 평가"""
        # 업력 점수 (5년 = 100점)
        age_score = min(100, (months_in_business / 60.0) * 100)

        # 매출 지속성 점수 (연속 매출 발생 개월 수)
        consistency_score = min(100, (revenue_consistency_months / months_in_business) * 100)

        # 전체 지속성 점수
        continuity_score = (age_score * 0.6 + consistency_score * 0.4)

        return {
            'months_in_business': months_in_business,
            'revenue_consistency_months': revenue_consistency_months,
            'age_score': age_score,
            'consistency_score': consistency_score,
            'continuity_score': continuity_score
        }

    def calculate_operational_stability_score(self,
                                            revenue_data: List[float],
                                            months_in_business: int) -> Dict[str, float]:
        """전체 영업 안정성 점수 계산"""

        # 각 구성 요소 계산
        growth_analysis = self.calculate_revenue_growth_trend(revenue_data)
        volatility_analysis = self.calculate_revenue_volatility(revenue_data)

        # 매출 지속성 (0이 아닌 매출 개월 수)
        revenue_consistency_months = sum(1 for r in revenue_data if r > 0)
        continuity_analysis = self.calculate_business_continuity(
            months_in_business, revenue_consistency_months)

        # 가중 평균 점수
        operational_score = (
            growth_analysis['trend_score'] * (self.weights['revenue_growth'] / 0.45) +
            volatility_analysis['volatility_score'] * (self.weights['revenue_volatility'] / 0.45) +
            continuity_analysis['continuity_score'] * (self.weights['business_continuity'] / 0.45)
        )

        return {
            'operational_stability_score': operational_score,
            'components': {
                'revenue_growth': growth_analysis,
                'revenue_volatility': volatility_analysis,
                'business_continuity': continuity_analysis
            },
            'weighted_scores': {
                'growth_weighted': growth_analysis['trend_score'] * self.weights['revenue_growth'],
                'volatility_weighted': volatility_analysis['volatility_score'] * self.weights['revenue_volatility'],
                'continuity_weighted': continuity_analysis['continuity_score'] * self.weights['business_continuity']
            }
        }

    def analyze_business_from_csv_data(self, business_df: pd.DataFrame,
                                     business_start_date: Optional[str] = None) -> Dict[str, float]:
        """CSV 데이터로부터 영업 안정성 분석"""

        # 매출 데이터 추출
        revenue_data = business_df['당월_매출_금액'].tolist()

        # 사업 기간 계산
        if business_start_date:
            # TODO: 날짜 계산 로직 추가
            months_in_business = len(revenue_data)  # 임시로 데이터 개수 사용
        else:
            months_in_business = len(revenue_data)

        return self.calculate_operational_stability_score(revenue_data, months_in_business)