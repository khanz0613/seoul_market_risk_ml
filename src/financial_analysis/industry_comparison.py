"""
Industry Comparison Calculator
업종 내 상대적 위치 평가 (15% 가중치)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class IndustryComparisonCalculator:
    """업종 비교 계산기"""

    def __init__(self):
        # 업종별 벤치마크 (임시 데이터 - 실제 한국은행 데이터로 교체 예정)
        self.industry_benchmarks = {
            'CS100001': {'name': '한식음식점', 'avg_profit_margin': 0.15, 'avg_revenue_per_month': 8000000},
            'CS100002': {'name': '중식음식점', 'avg_profit_margin': 0.12, 'avg_revenue_per_month': 6500000},
            'CS100003': {'name': '일식음식점', 'avg_profit_margin': 0.18, 'avg_revenue_per_month': 9200000},
            'CS100004': {'name': '양식음식점', 'avg_profit_margin': 0.16, 'avg_revenue_per_month': 7800000},
            'CS200001': {'name': '편의점', 'avg_profit_margin': 0.08, 'avg_revenue_per_month': 12000000},
            'CS200002': {'name': '슈퍼마켓', 'avg_profit_margin': 0.06, 'avg_revenue_per_month': 15000000}
        }

    def get_industry_benchmark(self, industry_code: str) -> Optional[Dict]:
        """업종 벤치마크 조회"""
        return self.industry_benchmarks.get(industry_code)

    def calculate_relative_profit_margin(self,
                                       business_revenue: float,
                                       business_expenses: float,
                                       industry_code: str) -> Dict[str, float]:
        """업종 대비 상대적 수익성 비교"""

        # 개별 사업자 수익률
        if business_revenue <= 0:
            business_profit_margin = 0.0
        else:
            business_profit_margin = (business_revenue - business_expenses) / business_revenue

        # 업종 평균 수익률
        benchmark = self.get_industry_benchmark(industry_code)
        if not benchmark:
            logger.warning(f"No benchmark found for industry: {industry_code}")
            return {
                'business_profit_margin': business_profit_margin,
                'industry_avg_margin': 0.10,  # 기본값 10%
                'relative_performance': 1.0,
                'performance_score': 50.0
            }

        industry_avg_margin = benchmark['avg_profit_margin']

        # 상대적 성과 (업종 평균 대비 배수)
        if industry_avg_margin > 0:
            relative_performance = business_profit_margin / industry_avg_margin
        else:
            relative_performance = 1.0

        # 점수화 (업종 평균 = 50점, 2배 = 100점, 0배 = 0점)
        performance_score = min(100, max(0, relative_performance * 50))

        return {
            'business_profit_margin': business_profit_margin,
            'industry_avg_margin': industry_avg_margin,
            'relative_performance': relative_performance,
            'performance_score': performance_score,
            'industry_name': benchmark['name']
        }

    def calculate_relative_revenue_scale(self,
                                       business_revenue: float,
                                       industry_code: str) -> Dict[str, float]:
        """업종 대비 상대적 매출 규모 비교"""

        benchmark = self.get_industry_benchmark(industry_code)
        if not benchmark:
            return {
                'business_revenue': business_revenue,
                'industry_avg_revenue': 10000000,  # 기본값 1000만원
                'revenue_scale_ratio': 1.0,
                'scale_score': 50.0
            }

        industry_avg_revenue = benchmark['avg_revenue_per_month']

        # 매출 규모 비율
        if industry_avg_revenue > 0:
            revenue_scale_ratio = business_revenue / industry_avg_revenue
        else:
            revenue_scale_ratio = 1.0

        # 점수화 (업종 평균 = 50점, 2배 = 100점)
        scale_score = min(100, max(0, revenue_scale_ratio * 50))

        return {
            'business_revenue': business_revenue,
            'industry_avg_revenue': industry_avg_revenue,
            'revenue_scale_ratio': revenue_scale_ratio,
            'scale_score': scale_score
        }

    def calculate_industry_position_score(self,
                                        business_revenue: float,
                                        business_expenses: float,
                                        industry_code: str) -> Dict[str, float]:
        """업종 내 종합 위치 점수"""

        # 수익성 비교
        profit_analysis = self.calculate_relative_profit_margin(
            business_revenue, business_expenses, industry_code)

        # 매출 규모 비교
        scale_analysis = self.calculate_relative_revenue_scale(
            business_revenue, industry_code)

        # 종합 점수 (수익성 70%, 규모 30% 가중 평균)
        industry_position_score = (
            profit_analysis['performance_score'] * 0.7 +
            scale_analysis['scale_score'] * 0.3
        )

        return {
            'industry_position_score': industry_position_score,
            'profit_analysis': profit_analysis,
            'scale_analysis': scale_analysis,
            'industry_code': industry_code,
            'benchmark_available': industry_code in self.industry_benchmarks
        }

    def update_industry_benchmarks(self, new_benchmarks: Dict[str, Dict]) -> None:
        """업종 벤치마크 업데이트"""
        self.industry_benchmarks.update(new_benchmarks)
        logger.info(f"Updated benchmarks for {len(new_benchmarks)} industries")

    def analyze_business_industry_position(self,
                                         business_data: pd.Series) -> Dict[str, float]:
        """사업자의 업종 내 위치 종합 분석"""

        revenue = business_data.get('당월_매출_금액', 0)
        expenses = business_data.get('추정지출금액', revenue * 0.75)  # 기본 지출률
        industry_code = business_data.get('서비스_업종_코드', 'UNKNOWN')

        return self.calculate_industry_position_score(revenue, expenses, industry_code)