"""
향상된 위험도 산정 모델
업종별 특화 + 비용 구조 분석 통합
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
from ..financial_analysis.cost_structure_analyzer import CostStructureAnalyzer, CostAnalysisResult
from ..data_processing.industry_mapper import IndustryMapper

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRiskAssessmentResult:
    """향상된 위험도 평가 결과"""
    business_id: str
    industry_category: str
    total_risk_score: float
    risk_level: str
    confidence: float

    # 기본 위험도 구성 요소 (기존)
    financial_health_score: float      # 30% (Altman Z-Score)
    operational_stability_score: float # 30% (매출 안정성)
    industry_position_score: float     # 25% (업종 내 위치)

    # 새로운 구성 요소
    cost_structure_score: float        # 15% (비용 구조 위험도)

    # 세부 분석
    altman_zscore: float
    cost_analysis_result: CostAnalysisResult

    # 위험 요인 분석 (핵심 기능)
    risk_factor_analysis: Dict[str, float]  # 재료비, 인건비, 임대료, 기타 각각 비중
    primary_risk_factor: str                # 가장 큰 위험 원인
    primary_risk_contribution: float        # 전체 원인의 몇%

    # 개선 방안
    improvement_to_level3: Dict[str, float] # 3단계 달성에 필요한 조치
    investment_opportunity: Dict[str, float] # 투자 기회 (여유 자금 있는 경우)

    # 현금 흐름 예측
    daily_cashflow_forecast: List[Dict]     # 7일간 예측
    weekly_comparison: Dict                 # 지난주 동기 대비

    # 메타데이터
    assessment_date: str
    data_quality_score: float

class EnhancedRiskCalculator:
    """향상된 위험도 계산기"""

    def __init__(self):
        # 구성 요소별 가중치 (업데이트됨)
        self.weights = {
            'financial_health': 0.30,        # 재무 건전성 30%
            'operational_stability': 0.30,   # 영업 안정성 30%
            'industry_position': 0.25,       # 상대적 위치 25%
            'cost_structure': 0.15          # 비용 구조 위험 15%
        }

        # 각 계산기 초기화
        self.altman_calculator = AltmanZScoreCalculator()
        self.operational_calculator = OperationalStabilityCalculator()
        self.industry_calculator = IndustryComparisonCalculator()
        self.cost_structure_analyzer = CostStructureAnalyzer()
        self.industry_mapper = IndustryMapper()

        # 위험도 등급 기준
        self.risk_levels = {
            (0, 20): "매우위험",
            (21, 40): "위험군",
            (41, 60): "적정",
            (61, 80): "좋음",
            (81, 100): "매우좋음"
        }

        logger.info("Enhanced Risk Calculator 초기화 완료")

    def calculate_comprehensive_risk_assessment(self,
                                              business_id: str,
                                              industry_code: str,
                                              revenue_history: List[float],
                                              actual_costs: Dict[str, float],
                                              operating_assets: float,
                                              months_in_business: int = None) -> EnhancedRiskAssessmentResult:
        """종합적 위험도 평가"""

        if months_in_business is None:
            months_in_business = len(revenue_history)

        # 업종 카테고리 매핑
        industry_category = self.industry_mapper.map_industry_code(industry_code)

        # 최신 재무 데이터
        current_revenue = revenue_history[-1] if revenue_history else 0
        total_expenses = sum(actual_costs.values())

        # 1. 재무 건전성 (30%) - 기존과 동일
        financial_health = self._calculate_financial_health_score(
            current_revenue, total_expenses, operating_assets, months_in_business)

        # 2. 영업 안정성 (30%) - 기존과 동일
        operational_stability = self.operational_calculator.calculate_operational_stability_score(
            revenue_history, months_in_business)

        # 3. 업종 내 위치 (25%) - 기존과 동일
        industry_position = self.industry_calculator.calculate_industry_position_score(
            current_revenue, total_expenses, industry_code)

        # 4. 비용 구조 분석 (15%) - 새로운 구성 요소
        cost_analysis = self.cost_structure_analyzer.analyze_cost_structure(
            business_id, industry_category, current_revenue, actual_costs)

        cost_structure_score = self._convert_cost_risk_to_score(cost_analysis.total_risk_score)

        # 5. 가중 평균 점수 계산
        total_risk_score = (
            financial_health['financial_health_score'] * self.weights['financial_health'] +
            operational_stability['operational_stability_score'] * self.weights['operational_stability'] +
            industry_position['industry_position_score'] * self.weights['industry_position'] +
            cost_structure_score * self.weights['cost_structure']
        )

        # 6. 위험 요인 분석
        risk_factor_analysis = self._calculate_risk_factor_contributions(
            cost_analysis, financial_health, operational_stability)

        primary_risk_factor = cost_analysis.primary_risk_factor
        primary_risk_contribution = cost_analysis.primary_risk_contribution

        # 7. 개선 방안 계산
        improvement_to_level3 = self._calculate_improvement_to_level3(
            total_risk_score, current_revenue, cost_analysis)

        investment_opportunity = self._calculate_investment_opportunity(
            total_risk_score, current_revenue, operating_assets)

        # 8. 현금 흐름 예측
        daily_cashflow_forecast = self._predict_7day_cashflow(
            revenue_history, actual_costs, industry_category)

        weekly_comparison = self._compare_with_last_week(
            revenue_history, actual_costs)

        # 9. 기타 메트릭
        risk_level = self._determine_risk_level(total_risk_score)
        data_quality_score = self._calculate_data_quality_score(
            revenue_history, actual_costs, operating_assets)
        confidence = self._calculate_confidence_score(
            data_quality_score, months_in_business, len(revenue_history))

        return EnhancedRiskAssessmentResult(
            business_id=business_id,
            industry_category=industry_category,
            total_risk_score=total_risk_score,
            risk_level=risk_level,
            confidence=confidence,
            financial_health_score=financial_health['financial_health_score'],
            operational_stability_score=operational_stability['operational_stability_score'],
            industry_position_score=industry_position['industry_position_score'],
            cost_structure_score=cost_structure_score,
            altman_zscore=financial_health['zscore'],
            cost_analysis_result=cost_analysis,
            risk_factor_analysis=risk_factor_analysis,
            primary_risk_factor=primary_risk_factor,
            primary_risk_contribution=primary_risk_contribution,
            improvement_to_level3=improvement_to_level3,
            investment_opportunity=investment_opportunity,
            daily_cashflow_forecast=daily_cashflow_forecast,
            weekly_comparison=weekly_comparison,
            assessment_date=datetime.now().isoformat(),
            data_quality_score=data_quality_score
        )

    def _calculate_financial_health_score(self, *args, **kwargs):
        """기존 재무 건전성 계산 (hybrid_risk_calculator에서 가져옴)"""
        # 기존 HybridRiskCalculator의 calculate_financial_health_score와 동일
        try:
            revenue, expenses, operating_assets, months_in_business = args

            zscore_result = self.altman_calculator.calculate_from_business_data(
                revenue, expenses, operating_assets, months_in_business)

            zscore = zscore_result['zscore']
            if zscore >= 2.99:
                score = 90 + min(10, (zscore - 2.99) * 5)
            elif zscore >= 1.81:
                score = 60 + ((zscore - 1.81) / (2.99 - 1.81)) * 30
            else:
                score = max(0, (zscore / 1.81) * 60)

            return {
                'financial_health_score': score,
                'zscore': zscore,
                'zscore_components': zscore_result['components'],
                'interpretation': zscore_result['interpretation']
            }

        except Exception as e:
            logger.error(f"Financial health calculation failed: {e}")
            return {
                'financial_health_score': 50.0,
                'zscore': 1.8,
                'zscore_components': {},
                'interpretation': {'risk_level': '불명', 'description': '계산 오류'}
            }

    def _convert_cost_risk_to_score(self, cost_risk_score: float) -> float:
        """비용 구조 위험도를 점수로 변환 (높은 위험 = 낮은 점수)"""
        return max(0, 100 - cost_risk_score)

    def _calculate_risk_factor_contributions(self,
                                           cost_analysis: CostAnalysisResult,
                                           financial_health: Dict,
                                           operational_stability: Dict) -> Dict[str, float]:
        """위험 요인별 기여도 계산"""

        # 비용 구조에서 각 항목별 위험 기여도
        cost_contributions = {}
        total_cost_risk = sum(
            factor['weighted_risk_score']
            for factor in cost_analysis.risk_factors.values()
        )

        for category, factor_data in cost_analysis.risk_factors.items():
            if total_cost_risk > 0:
                contribution = (factor_data['weighted_risk_score'] / total_cost_risk) * 15  # 비용 구조는 전체의 15%
                cost_contributions[category] = contribution
            else:
                cost_contributions[category] = 0

        # 전체 위험 요인에 재무/영업 위험도 추가
        risk_factor_analysis = cost_contributions.copy()

        # 재무 위험은 "기타_재무위험"으로 통합
        financial_risk = (100 - financial_health['financial_health_score']) * 0.30
        risk_factor_analysis['기타_재무위험'] = financial_risk * 0.15  # 전체 기여도로 변환

        # 영업 위험은 "기타_영업위험"으로 통합
        operational_risk = (100 - operational_stability['operational_stability_score']) * 0.30
        risk_factor_analysis['기타_영업위험'] = operational_risk * 0.15  # 전체 기여도로 변환

        return risk_factor_analysis

    def _calculate_improvement_to_level3(self,
                                       current_score: float,
                                       revenue: float,
                                       cost_analysis: CostAnalysisResult) -> Dict[str, float]:
        """3단계(적정) 달성을 위한 개선 방안"""

        target_score = 60  # 3단계 시작점
        if current_score >= target_score:
            return {"message": "이미 3단계 이상입니다", "required_amount": 0}

        score_gap = target_score - current_score

        # 주요 개선 방안별 필요 금액 추정
        improvement_methods = {}

        # 1. 현금 증자를 통한 재무 건전성 개선
        cash_injection_needed = revenue * (score_gap / 100) * 1.5  # 매출의 1.5배 비율로 추정
        improvement_methods["현금_증자"] = cash_injection_needed

        # 2. 비용 절감을 통한 개선
        if cost_analysis.primary_risk_factor in cost_analysis.cost_deviations:
            cost_reduction_needed = abs(cost_analysis.cost_deviations[cost_analysis.primary_risk_factor]) * 0.7
            improvement_methods["비용_절감"] = cost_reduction_needed

        # 3. 매출 증대를 통한 개선
        revenue_increase_needed = revenue * (score_gap / 100) * 2.0
        improvement_methods["매출_증대"] = revenue_increase_needed

        # 가장 현실적인 방법 선택
        min_amount = min(improvement_methods.values())
        improvement_methods["권장_개선_금액"] = min_amount

        return improvement_methods

    def _calculate_investment_opportunity(self,
                                        current_score: float,
                                        revenue: float,
                                        operating_assets: float) -> Dict[str, float]:
        """투자 기회 계산 (좋은 등급인 경우)"""

        if current_score < 60:  # 3단계 미만은 투자보다 개선이 우선
            return {"message": "위험도 개선이 우선입니다", "investable_amount": 0}

        # 여유 현금 추정
        estimated_surplus = max(0, operating_assets - revenue * 0.5)  # 매출의 50%는 운영 자금으로 보존

        investment_opportunities = {
            "추정_여유자금": estimated_surplus,
            "안전투자_가능금액": estimated_surplus * 0.3,  # 30%만 투자
            "성장투자_가능금액": estimated_surplus * 0.5,  # 50% 투자
            "적극투자_가능금액": estimated_surplus * 0.7   # 70% 투자
        }

        return investment_opportunities

    def _predict_7day_cashflow(self,
                              revenue_history: List[float],
                              actual_costs: Dict[str, float],
                              industry_category: str) -> List[Dict]:
        """7일간 현금 흐름 예측"""

        if not revenue_history:
            return []

        # 최근 매출 트렌드 계산
        recent_revenue = np.mean(revenue_history[-7:]) if len(revenue_history) >= 7 else revenue_history[-1]
        daily_revenue = recent_revenue / 30  # 일평균 매출

        # 요일별 매출 패턴 (업종별)
        weekday_patterns = {
            "숙박음식점업": [0.8, 0.9, 1.0, 1.1, 1.3, 1.4, 1.2],  # 주말 높음
            "도매소매업": [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 0.9],    # 평일 위주
            "예술스포츠업": [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.3],   # 주말/저녁 높음
            "개인서비스업": [1.1, 1.1, 1.0, 1.0, 1.0, 0.8, 0.7]    # 평일 위주
        }

        pattern = weekday_patterns.get(industry_category, [1.0] * 7)

        # 일별 비용 (월 비용을 30일로 나눔)
        daily_costs = {k: v / 30 for k, v in actual_costs.items()}
        total_daily_cost = sum(daily_costs.values())

        forecast = []
        for day in range(7):
            day_revenue = daily_revenue * pattern[day % 7]
            net_cashflow = day_revenue - total_daily_cost

            forecast.append({
                "day": day + 1,
                "predicted_revenue": round(day_revenue),
                "predicted_expenses": round(total_daily_cost),
                "net_cashflow": round(net_cashflow),
                "expense_breakdown": {k: round(v) for k, v in daily_costs.items()}
            })

        return forecast

    def _compare_with_last_week(self,
                               revenue_history: List[float],
                               actual_costs: Dict[str, float]) -> Dict:
        """지난주 동기 대비 분석"""

        if len(revenue_history) < 14:  # 2주치 데이터 필요
            return {"message": "비교를 위한 충분한 데이터가 없습니다"}

        # 최근 주 vs 전주 비교
        recent_week_avg = np.mean(revenue_history[-7:])
        last_week_avg = np.mean(revenue_history[-14:-7])

        revenue_change = recent_week_avg - last_week_avg
        revenue_change_percent = (revenue_change / last_week_avg * 100) if last_week_avg > 0 else 0

        # 주요 변화 요인 분석
        change_factors = []
        if abs(revenue_change_percent) > 5:  # 5% 이상 변화
            if revenue_change_percent > 0:
                change_factors.append("매출 증가 추세")
            else:
                change_factors.append("매출 감소 추세")

        return {
            "recent_week_avg": round(recent_week_avg),
            "last_week_avg": round(last_week_avg),
            "revenue_change": round(revenue_change),
            "revenue_change_percent": round(revenue_change_percent, 1),
            "change_factors": change_factors if change_factors else ["안정적 매출 유지"]
        }

    def _determine_risk_level(self, score: float) -> str:
        """점수 기반 위험도 등급 결정"""
        for (min_score, max_score), level in self.risk_levels.items():
            if min_score <= score <= max_score:
                return level
        return "적정"

    def _calculate_data_quality_score(self, revenue_history, actual_costs, operating_assets):
        """데이터 품질 점수 계산 (기존과 동일)"""
        quality_factors = []

        if len(revenue_history) >= 6:
            quality_factors.append(0.9)
        elif len(revenue_history) >= 3:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)

        # 비용 데이터 품질 (세부 항목이 있으면 높은 점수)
        if len(actual_costs) >= 4:
            quality_factors.append(0.9)
        elif len(actual_costs) >= 2:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.5)

        if operating_assets > 0:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.2)

        return np.mean(quality_factors) * 100

    def _calculate_confidence_score(self, data_quality_score, months_in_business, data_points):
        """신뢰도 점수 계산 (기존과 동일)"""
        quality_component = data_quality_score * 0.4
        experience_component = min(100, (months_in_business / 36.0) * 100) * 0.3
        data_sufficiency = min(100, (data_points / 12.0) * 100) * 0.3

        return quality_component + experience_component + data_sufficiency

# 사용 예시
def main():
    calculator = EnhancedRiskCalculator()

    # 테스트 데이터
    test_data = {
        "business_id": "TEST001",
        "industry_code": "CS100001",  # 한식음식점
        "revenue_history": [7500000, 8000000, 7800000, 8200000, 8100000, 7900000],  # 6개월 매출
        "actual_costs": {
            "재료비": 4000000,    # 50%
            "인건비": 1200000,    # 15%
            "임대료": 600000,     # 7.5%
            "제세공과금": 200000, # 2.5%
            "기타": 2000000      # 25%
        },
        "operating_assets": 15000000,  # 1500만원 운용자산
        "months_in_business": 18
    }

    result = calculator.calculate_comprehensive_risk_assessment(**test_data)

    print("=== 향상된 위험도 평가 결과 ===")
    print(f"업종: {result.industry_category}")
    print(f"전체 위험 점수: {result.total_risk_score:.1f} ({result.risk_level})")
    print(f"주요 위험 요인: {result.primary_risk_factor} ({result.primary_risk_contribution:.1f}%)")

    print("\n=== 위험 요인별 기여도 ===")
    for factor, contribution in result.risk_factor_analysis.items():
        print(f"{factor}: {contribution:.1f}%")

    print(f"\n=== 3단계 달성 방안 ===")
    for method, amount in result.improvement_to_level3.items():
        if isinstance(amount, (int, float)) and amount > 0:
            print(f"{method}: {amount:,.0f}원")

    print(f"\n=== 7일 현금흐름 예측 ===")
    for day_forecast in result.daily_cashflow_forecast[:3]:  # 처음 3일만
        print(f"Day {day_forecast['day']}: 매출 {day_forecast['predicted_revenue']:,}원, "
              f"순현금흐름 {day_forecast['net_cashflow']:,}원")

if __name__ == "__main__":
    main()