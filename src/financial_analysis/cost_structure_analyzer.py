"""
업종별 비용 구조 분석기
사용자의 실제 지출과 업종 평균을 비교하여 위험 요인 분석
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from ..data_processing.industry_mapper import IndustryMapper

logger = logging.getLogger(__name__)

@dataclass
class CostAnalysisResult:
    """비용 구조 분석 결과"""
    business_id: str
    industry_category: str
    total_revenue: float

    # 사용자 실제 비용
    actual_costs: Dict[str, float]
    actual_cost_ratios: Dict[str, float]

    # 업종 평균 비용
    industry_avg_costs: Dict[str, float]
    industry_avg_ratios: Dict[str, float]

    # 편차 분석
    cost_deviations: Dict[str, float]  # 절대 편차 (원)
    ratio_deviations: Dict[str, float]  # 비율 편차 (%)

    # 위험 분석
    risk_factors: Dict[str, Dict]  # 각 비용 항목별 위험도
    primary_risk_factor: str  # 가장 큰 위험 요인
    primary_risk_contribution: float  # 주요 위험 기여도 (%)

    total_risk_score: float  # 전체 위험 점수 (0-100)

class CostStructureAnalyzer:
    """비용 구조 분석기"""

    def __init__(self):
        self.industry_mapper = IndustryMapper()

        # 위험도 가중치 (업종별 민감도)
        self.RISK_WEIGHTS = {
            "도매소매업": {
                "재료비": 0.50,  # 재료비가 주요 위험 요인
                "인건비": 0.25,
                "임대료": 0.15,
                "제세공과금": 0.05,
                "기타": 0.05
            },
            "숙박음식점업": {
                "재료비": 0.35,  # 재료비와 인건비 균형
                "인건비": 0.35,
                "임대료": 0.20,
                "제세공과금": 0.05,
                "기타": 0.05
            },
            "예술스포츠업": {
                "재료비": 0.10,  # 인건비와 임대료가 핵심
                "인건비": 0.40,
                "임대료": 0.40,
                "제세공과금": 0.05,
                "기타": 0.05
            },
            "개인서비스업": {
                "재료비": 0.15,  # 인건비가 가장 중요
                "인건비": 0.45,
                "임대료": 0.25,
                "제세공과금": 0.05,
                "기타": 0.10
            }
        }

        logger.info("Cost Structure Analyzer 초기화 완료")

    def analyze_cost_structure(self,
                             business_id: str,
                             industry_category: str,
                             revenue: float,
                             actual_costs: Dict[str, float]) -> CostAnalysisResult:
        """비용 구조 분석"""

        # 1. 업종 평균 비용 구조
        industry_cost_structure = self.industry_mapper.get_cost_structure(industry_category)

        # 2. 업종 평균 절대 비용 계산
        industry_avg_costs = {
            category: revenue * ratio
            for category, ratio in industry_cost_structure.items()
        }

        # 3. 사용자 비용 비율 계산
        total_actual_costs = sum(actual_costs.values())
        actual_cost_ratios = {
            category: cost / total_actual_costs if total_actual_costs > 0 else 0
            for category, cost in actual_costs.items()
        }

        # 4. 편차 계산
        cost_deviations = {
            category: actual_costs.get(category, 0) - industry_avg_costs.get(category, 0)
            for category in industry_cost_structure.keys()
        }

        ratio_deviations = {
            category: (actual_cost_ratios.get(category, 0) -
                      industry_cost_structure.get(category, 0)) * 100
            for category in industry_cost_structure.keys()
        }

        # 5. 위험 요인 분석
        risk_factors = self._calculate_risk_factors(
            industry_category, ratio_deviations, cost_deviations, revenue)

        # 6. 주요 위험 요인 식별
        primary_risk_factor, primary_risk_contribution = self._identify_primary_risk(
            risk_factors, industry_category)

        # 7. 전체 위험 점수 계산
        total_risk_score = self._calculate_total_risk_score(
            risk_factors, industry_category)

        return CostAnalysisResult(
            business_id=business_id,
            industry_category=industry_category,
            total_revenue=revenue,
            actual_costs=actual_costs,
            actual_cost_ratios=actual_cost_ratios,
            industry_avg_costs=industry_avg_costs,
            industry_avg_ratios=industry_cost_structure,
            cost_deviations=cost_deviations,
            ratio_deviations=ratio_deviations,
            risk_factors=risk_factors,
            primary_risk_factor=primary_risk_factor,
            primary_risk_contribution=primary_risk_contribution,
            total_risk_score=total_risk_score
        )

    def _calculate_risk_factors(self,
                               industry_category: str,
                               ratio_deviations: Dict[str, float],
                               cost_deviations: Dict[str, float],
                               revenue: float) -> Dict[str, Dict]:
        """각 비용 항목별 위험도 계산"""

        risk_factors = {}
        weights = self.RISK_WEIGHTS.get(industry_category, {})

        for category, ratio_deviation in ratio_deviations.items():
            # 비율 편차를 위험 점수로 변환 (0-100 점수)
            # 양의 편차(평균 대비 더 많이 지출) = 높은 위험
            raw_risk = min(100, max(0, abs(ratio_deviation) * 2))  # ±50% 편차 = 100점

            # 업종별 가중치 적용
            weight = weights.get(category, 0.2)  # 기본 가중치 20%
            weighted_risk = raw_risk * weight

            # 위험 등급 결정
            if weighted_risk >= 80:
                risk_level = "매우높음"
            elif weighted_risk >= 60:
                risk_level = "높음"
            elif weighted_risk >= 40:
                risk_level = "보통"
            elif weighted_risk >= 20:
                risk_level = "낮음"
            else:
                risk_level = "매우낮음"

            risk_factors[category] = {
                'ratio_deviation': ratio_deviation,
                'cost_deviation': cost_deviations.get(category, 0),
                'raw_risk_score': raw_risk,
                'weighted_risk_score': weighted_risk,
                'risk_level': risk_level,
                'weight': weight,
                'impact_description': self._get_risk_impact_description(
                    category, ratio_deviation, industry_category)
            }

        return risk_factors

    def _identify_primary_risk(self,
                              risk_factors: Dict[str, Dict],
                              industry_category: str) -> Tuple[str, float]:
        """주요 위험 요인 식별"""

        # 가중 위험 점수 기준으로 정렬
        sorted_risks = sorted(
            risk_factors.items(),
            key=lambda x: x[1]['weighted_risk_score'],
            reverse=True
        )

        if not sorted_risks:
            return "없음", 0.0

        primary_factor, primary_data = sorted_risks[0]

        # 전체 위험에서 이 요인의 기여도 계산
        total_weighted_risk = sum(
            data['weighted_risk_score'] for data in risk_factors.values()
        )

        contribution = (
            primary_data['weighted_risk_score'] / total_weighted_risk * 100
            if total_weighted_risk > 0 else 0
        )

        return primary_factor, contribution

    def _calculate_total_risk_score(self,
                                   risk_factors: Dict[str, Dict],
                                   industry_category: str) -> float:
        """전체 위험 점수 계산"""

        # 가중 평균으로 전체 위험 점수 계산
        total_weighted_risk = sum(
            data['weighted_risk_score'] for data in risk_factors.values()
        )

        # 0-100 점수로 정규화
        normalized_score = min(100, total_weighted_risk)

        return normalized_score

    def _get_risk_impact_description(self,
                                   category: str,
                                   deviation: float,
                                   industry_category: str) -> str:
        """위험 요인별 영향 설명"""

        if abs(deviation) < 5:  # 5% 이내 편차
            return f"{category} 지출이 업종 평균과 유사한 수준"

        direction = "높음" if deviation > 0 else "낮음"

        descriptions = {
            "재료비": f"원자재/상품 구매비용이 업종 평균보다 {abs(deviation):.1f}% {direction}",
            "인건비": f"직원 급여 및 인건비가 업종 평균보다 {abs(deviation):.1f}% {direction}",
            "임대료": f"임대료 및 부동산 비용이 업종 평균보다 {abs(deviation):.1f}% {direction}",
            "제세공과금": f"세금 및 공과금이 업종 평균보다 {abs(deviation):.1f}% {direction}",
            "기타": f"기타 운영비용이 업종 평균보다 {abs(deviation):.1f}% {direction}"
        }

        return descriptions.get(category, f"{category}가 업종 평균보다 {abs(deviation):.1f}% {direction}")

    def generate_improvement_recommendations(self,
                                           analysis_result: CostAnalysisResult) -> List[Dict[str, str]]:
        """개선 방안 추천"""

        recommendations = []

        # 위험도 높은 순으로 정렬
        sorted_risks = sorted(
            analysis_result.risk_factors.items(),
            key=lambda x: x[1]['weighted_risk_score'],
            reverse=True
        )

        for category, risk_data in sorted_risks[:3]:  # 상위 3개만
            if risk_data['weighted_risk_score'] > 30:  # 일정 위험도 이상만

                deviation = risk_data['ratio_deviation']
                if deviation > 5:  # 평균보다 5% 이상 높은 경우
                    recommendations.append({
                        'category': category,
                        'priority': self._get_priority_level(risk_data['weighted_risk_score']),
                        'issue': f"{category} 지출이 업종 평균보다 {deviation:.1f}% 높음",
                        'recommendation': self._get_specific_recommendation(
                            category, deviation, analysis_result.industry_category),
                        'potential_saving': analysis_result.cost_deviations[category]
                    })

        return recommendations

    def _get_priority_level(self, risk_score: float) -> str:
        """우선순위 레벨 결정"""
        if risk_score >= 80:
            return "긴급"
        elif risk_score >= 60:
            return "높음"
        elif risk_score >= 40:
            return "보통"
        else:
            return "낮음"

    def _get_specific_recommendation(self,
                                   category: str,
                                   deviation: float,
                                   industry_category: str) -> str:
        """카테고리별 구체적 개선 방안"""

        recommendations = {
            "재료비": [
                "공급업체 다변화를 통한 원가 절감",
                "대량 구매 또는 공동 구매를 통한 단가 인하",
                "재고 회전율 개선으로 폐기 손실 최소화",
                "계절별 원가 변동성 고려한 구매 계획"
            ],
            "인건비": [
                "업무 효율성 개선을 통한 인력 최적화",
                "파트타임/계약직 활용으로 고정비 절감",
                "자동화 도구 도입으로 인력 의존도 감소",
                "성과급 제도 도입으로 생산성 향상"
            ],
            "임대료": [
                "임대료 재협상 또는 더 저렴한 위치 검토",
                "공간 활용도 개선으로 평당 수익성 증대",
                "부분 임대(서브리스) 검토",
                "온라인 채널 확대로 물리적 공간 의존도 감소"
            ],
            "제세공과금": [
                "세무 전문가 상담을 통한 절세 방안 검토",
                "에너지 효율화를 통한 공과금 절감",
                "정부 지원 제도 및 세액 공제 항목 확인"
            ],
            "기타": [
                "불필요한 비용 항목 검토 및 정리",
                "공급업체별 비용 재검토",
                "운영 프로세스 효율화로 간접비용 절감"
            ]
        }

        category_recommendations = recommendations.get(category, ["비용 구조 최적화 검토"])

        # 편차 정도에 따른 추천 선택
        if deviation > 20:  # 20% 이상 편차
            return category_recommendations[0]  # 가장 강력한 조치
        elif deviation > 10:  # 10-20% 편차
            return category_recommendations[1] if len(category_recommendations) > 1 else category_recommendations[0]
        else:  # 5-10% 편차
            return category_recommendations[-1]  # 점진적 개선

    def compare_with_similar_businesses(self,
                                      analysis_result: CostAnalysisResult,
                                      benchmark_data: pd.DataFrame) -> Dict:
        """유사 업종/규모 사업체와 비교"""

        # 동일 업종, 유사 매출 규모 필터링
        similar_businesses = benchmark_data[
            (benchmark_data['통합업종카테고리'] == analysis_result.industry_category) &
            (benchmark_data['당월_매출_금액'] >= analysis_result.total_revenue * 0.7) &
            (benchmark_data['당월_매출_금액'] <= analysis_result.total_revenue * 1.3)
        ]

        if len(similar_businesses) == 0:
            return {"message": "유사한 사업체 데이터가 부족합니다."}

        # 백분위 계산
        percentiles = {}
        for category in analysis_result.actual_cost_ratios.keys():
            if category in analysis_result.industry_avg_ratios:
                user_ratio = analysis_result.actual_cost_ratios[category]
                # 업종 평균 대비 사용자 위치
                avg_ratio = analysis_result.industry_avg_ratios[category]
                relative_position = (user_ratio / avg_ratio * 100) if avg_ratio > 0 else 100
                percentiles[category] = min(100, max(0, relative_position))

        return {
            "similar_business_count": len(similar_businesses),
            "user_percentiles": percentiles,
            "interpretation": self._interpret_percentiles(percentiles)
        }

    def _interpret_percentiles(self, percentiles: Dict[str, float]) -> Dict[str, str]:
        """백분위 해석"""
        interpretations = {}

        for category, percentile in percentiles.items():
            if percentile >= 150:
                interpretations[category] = "상위 10% (매우 높은 지출)"
            elif percentile >= 125:
                interpretations[category] = "상위 25% (높은 지출)"
            elif percentile >= 75:
                interpretations[category] = "평균 수준"
            elif percentile >= 50:
                interpretations[category] = "하위 25% (낮은 지출)"
            else:
                interpretations[category] = "하위 10% (매우 낮은 지출)"

        return interpretations

# 사용 예시
def main():
    analyzer = CostStructureAnalyzer()

    # 테스트 데이터
    test_data = {
        "business_id": "TEST001",
        "industry_category": "숙박음식점업",
        "revenue": 8000000,  # 800만원 매출
        "actual_costs": {
            "재료비": 4000000,    # 50% (업종 평균 42.6%)
            "인건비": 1200000,    # 15% (업종 평균 20.5%)
            "임대료": 600000,     # 7.5% (업종 평균 9.0%)
            "제세공과금": 200000, # 2.5% (업종 평균 3.5%)
            "기타": 2000000      # 25% (업종 평균 24.4%)
        }
    }

    result = analyzer.analyze_cost_structure(**test_data)

    print("=== 비용 구조 분석 결과 ===")
    print(f"업종: {result.industry_category}")
    print(f"주요 위험 요인: {result.primary_risk_factor} ({result.primary_risk_contribution:.1f}%)")
    print(f"전체 위험 점수: {result.total_risk_score:.1f}")

    recommendations = analyzer.generate_improvement_recommendations(result)
    print("\n=== 개선 방안 ===")
    for rec in recommendations:
        print(f"[{rec['priority']}] {rec['issue']}")
        print(f"  → {rec['recommendation']}")

if __name__ == "__main__":
    main()