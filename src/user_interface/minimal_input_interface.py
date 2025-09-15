"""
최소 입력 사용자 인터페이스
UX 최적화를 통한 입력 최소화
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

from ..risk_scoring.enhanced_risk_calculator import EnhancedRiskCalculator
from ..data_processing.industry_mapper import IndustryMapper

logger = logging.getLogger(__name__)

@dataclass
class BusinessInput:
    """비즈니스 입력 데이터"""
    # 필수 입력 (최소 4개)
    업종코드: str              # 업종 선택 (드롭다운)
    월매출: float             # 최근 월 매출
    운용자산: float           # 현재 보유 현금/자산
    업력_개월: int            # 사업 운영 개월 수

    # 선택 입력 (더 정확한 분석을 원할 때)
    실제_인건비: Optional[float] = None
    실제_임대료: Optional[float] = None
    실제_재료비: Optional[float] = None
    매출_이력: Optional[List[float]] = None  # 최근 6개월 매출

    # 자동 계산되는 필드
    업종카테고리: Optional[str] = None
    추정_비용구조: Optional[Dict[str, float]] = None

class MinimalInputInterface:
    """최소 입력 인터페이스"""

    def __init__(self, ml_model=None):
        """
        초기화

        Args:
            ml_model: ML 비용 예측 모델 (None이면 기존 고정 비율 사용)
        """
        self.industry_mapper = IndustryMapper()
        self.risk_calculator = EnhancedRiskCalculator()
        self.ml_model = ml_model

        # 업종 코드 매핑 (사용자 친화적 이름)
        self.INDUSTRY_CHOICES = {
            # 숙박음식점업
            "CS100001": "한식음식점",
            "CS100002": "중식음식점",
            "CS100003": "일식음식점",
            "CS100004": "양식음식점",
            "CS100005": "제과점/베이커리",
            "CS100006": "패스트푸드점",
            "CS100007": "치킨전문점",
            "CS100008": "분식전문점",
            "CS100009": "호프/주점",
            "CS100010": "카페/음료점",

            # 도매소매업
            "CS300001": "슈퍼마켓",
            "CS300002": "편의점",
            "CS300011": "의류매장",
            "CS300022": "화장품매장",
            "CS300031": "가구매장",
            "CS300032": "가전제품매장",

            # 예술스포츠업
            "CS200001": "학원(일반)",
            "CS200002": "외국어학원",
            "CS200003": "예술학원",
            "CS200005": "스포츠강습",
            "CS200017": "골프연습장",
            "CS200019": "PC방",
            "CS200037": "노래방",

            # 개인서비스업
            "CS200006": "병원/의원",
            "CS200007": "치과",
            "CS200028": "미용실",
            "CS200029": "네일샵",
            "CS200030": "피부관리실",
            "CS200031": "세탁소",
            "CS200033": "부동산중개업"
        }

        logger.info("Minimal Input Interface 초기화 완료")

    def get_industry_choices(self) -> Dict[str, str]:
        """업종 선택 리스트 반환"""
        return self.INDUSTRY_CHOICES

    def process_minimal_input(self, business_input: BusinessInput) -> Dict[str, Any]:
        """최소 입력으로 전체 분석 수행"""

        # 1. 업종 카테고리 매핑
        business_input.업종카테고리 = self.industry_mapper.map_industry_code(business_input.업종코드)

        # 2. 비용 구조 자동 추정
        business_input.추정_비용구조 = self._estimate_cost_structure(business_input)

        # 3. 누락된 매출 이력 생성
        if not business_input.매출_이력:
            business_input.매출_이력 = self._generate_revenue_history(business_input.월매출)

        # 4. 종합 위험도 분석
        analysis_result = self.risk_calculator.calculate_comprehensive_risk_assessment(
            business_id=f"USER_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            industry_code=business_input.업종코드,
            revenue_history=business_input.매출_이력,
            actual_costs=business_input.추정_비용구조,
            operating_assets=business_input.운용자산,
            months_in_business=business_input.업력_개월
        )

        # 5. 결과 포맷팅 (사용자 친화적)
        formatted_result = self._format_user_friendly_result(analysis_result, business_input)

        return formatted_result

    def _estimate_cost_structure(self, business_input: BusinessInput) -> Dict[str, float]:
        """비용 구조 자동 추정 - ML 모델 또는 고정 비율 사용"""

        # 사용자가 실제 값을 입력한 경우 우선 사용
        estimated_costs = {}

        # ML 모델이 있는 경우 AI 예측 사용
        if self.ml_model and self.ml_model.is_trained:
            try:
                logger.info("🤖 ML 모델을 사용한 AI 비용 예측 실행")

                # ML 모델로 예측
                ml_predictions = self.ml_model.predict(
                    revenue=business_input.월매출,
                    industry_code=business_input.업종코드,
                    region=None  # 지역 정보가 없으면 None
                )

                # ML 예측 결과를 기본값으로 사용
                estimated_costs['재료비'] = ml_predictions.get('재료비', 0)
                estimated_costs['인건비'] = ml_predictions.get('인건비', 0)
                estimated_costs['임대료'] = ml_predictions.get('임대료', 0)
                estimated_costs['기타'] = ml_predictions.get('기타', 0)

                # 사용자 실제 입력이 있으면 해당 값으로 대체
                if business_input.실제_재료비:
                    estimated_costs['재료비'] = business_input.실제_재료비
                if business_input.실제_인건비:
                    estimated_costs['인건비'] = business_input.실제_인건비
                if business_input.실제_임대료:
                    estimated_costs['임대료'] = business_input.실제_임대료

                # 제세공과금은 기존 방식 사용 (ML 모델에서 예측하지 않음)
                standard_structure = self.industry_mapper.get_cost_structure(business_input.업종카테고리)
                if '제세공과금' in standard_structure:
                    estimated_costs['제세공과금'] = business_input.월매출 * standard_structure['제세공과금']

                logger.info("✅ ML 예측 완료")
                return estimated_costs

            except Exception as e:
                logger.warning(f"⚠️ ML 예측 실패, 기존 방식 사용: {e}")

        # ML 모델이 없거나 실패한 경우 기존 고정 비율 방식 사용
        logger.info("📊 고정 비율 방식으로 비용 구조 추정")

        # 업종별 표준 비용 구조 가져오기
        standard_structure = self.industry_mapper.get_cost_structure(business_input.업종카테고리)

        for category, ratio in standard_structure.items():
            if category == "재료비" and business_input.실제_재료비:
                estimated_costs[category] = business_input.실제_재료비
            elif category == "인건비" and business_input.실제_인건비:
                estimated_costs[category] = business_input.실제_인건비
            elif category == "임대료" and business_input.실제_임대료:
                estimated_costs[category] = business_input.실제_임대료
            else:
                # 표준 비율로 추정
                estimated_costs[category] = business_input.월매출 * ratio

        return estimated_costs

    def _generate_revenue_history(self, current_revenue: float) -> List[float]:
        """매출 이력 생성 (변동성 고려)"""

        # 6개월 매출 이력 생성 (현실적인 변동 반영)
        base_revenue = current_revenue
        revenue_history = []

        # 월별 변동률 (계절성 및 랜덤 변동 반영)
        variations = [0.95, 0.98, 1.02, 0.97, 1.01, 1.00]  # 마지막이 현재 매출

        for i, variation in enumerate(variations):
            # 약간의 랜덤 노이즈 추가 (±5%)
            noise = np.random.uniform(0.95, 1.05)
            monthly_revenue = base_revenue * variation * noise
            revenue_history.append(monthly_revenue)

        return revenue_history

    def _format_user_friendly_result(self,
                                   analysis_result,
                                   business_input: BusinessInput) -> Dict[str, Any]:
        """사용자 친화적 결과 포맷팅"""

        # 업종명 가져오기
        industry_name = self.INDUSTRY_CHOICES.get(business_input.업종코드, business_input.업종카테고리)

        result = {
            # 기본 정보
            "사업정보": {
                "업종": industry_name,
                "업종카테고리": business_input.업종카테고리,
                "월매출": f"{business_input.월매출:,.0f}원",
                "업력": f"{business_input.업력_개월}개월"
            },

            # 핵심 결과
            "위험도평가": {
                "종합점수": f"{analysis_result.total_risk_score:.1f}점",
                "위험등급": analysis_result.risk_level,
                "신뢰도": f"{analysis_result.confidence:.1f}%"
            },

            # 위험 원인 분석 (핵심 요구사항)
            "위험원인분석": self._format_risk_factor_analysis(analysis_result),

            # 7일간 현금 흐름 예측 (핵심 요구사항)
            "현금흐름예측": {
                "7일예측": analysis_result.daily_cashflow_forecast,
                "지난주대비": analysis_result.weekly_comparison
            },

            # 개선 방안
            "개선방안": {
                "3단계달성방법": analysis_result.improvement_to_level3,
                "투자기회": analysis_result.investment_opportunity
            },

            # 비용 구조 비교
            "비용구조비교": self._format_cost_comparison(analysis_result, business_input),

            # 메타 정보
            "분석정보": {
                "분석일시": analysis_result.assessment_date,
                "데이터품질": f"{analysis_result.data_quality_score:.1f}점"
            }
        }

        return result

    def _format_risk_factor_analysis(self, analysis_result) -> Dict[str, Any]:
        """위험 원인 분석 포맷팅"""

        # 비용 항목별 위험 기여도 계산
        cost_risk_factors = {}
        total_risk_contribution = 0

        for category, contribution in analysis_result.risk_factor_analysis.items():
            if category in ["재료비", "인건비", "임대료", "기타"]:
                cost_risk_factors[category] = {
                    "기여도": f"{contribution:.1f}%",
                    "업종평균대비": f"{analysis_result.cost_analysis_result.ratio_deviations.get(category, 0):+.1f}%"
                }
                total_risk_contribution += contribution

        # 기타 위험 요인들 합계
        other_factors = 100 - total_risk_contribution
        if other_factors > 0:
            cost_risk_factors["기타위험"] = {
                "기여도": f"{other_factors:.1f}%",
                "설명": "재무건전성, 영업안정성 등 기타 요인"
            }

        return {
            "항목별비중": cost_risk_factors,
            "가장큰원인": {
                "항목": analysis_result.primary_risk_factor,
                "기여도": f"{analysis_result.primary_risk_contribution:.1f}%"
            },
            "요약": f"{analysis_result.primary_risk_factor}가 전체 위험의 {analysis_result.primary_risk_contribution:.1f}%를 차지"
        }

    def _format_cost_comparison(self, analysis_result, business_input: BusinessInput) -> Dict[str, Any]:
        """비용 구조 비교 포맷팅"""

        cost_comparison = {}

        for category in ["재료비", "인건비", "임대료", "기타"]:
            actual_ratio = analysis_result.cost_analysis_result.actual_cost_ratios.get(category, 0)
            industry_avg = analysis_result.cost_analysis_result.industry_avg_ratios.get(category, 0)

            cost_comparison[category] = {
                "사용자": f"{actual_ratio*100:.1f}%",
                "업종평균": f"{industry_avg*100:.1f}%",
                "편차": f"{(actual_ratio - industry_avg)*100:+.1f}%",
                "상태": "높음" if actual_ratio > industry_avg * 1.1 else
                       "낮음" if actual_ratio < industry_avg * 0.9 else "적정"
            }

        return cost_comparison

    def create_simple_demo(self) -> Dict[str, Any]:
        """간단한 데모 예시"""

        # 데모용 입력 데이터
        demo_input = BusinessInput(
            업종코드="CS100001",  # 한식음식점
            월매출=8000000,      # 800만원
            운용자산=15000000,   # 1500만원
            업력_개월=18,        # 18개월
            실제_인건비=1500000, # 150만원 (선택 입력)
            실제_임대료=700000   # 70만원 (선택 입력)
        )

        return self.process_minimal_input(demo_input)

    def validate_input(self, business_input: BusinessInput) -> Tuple[bool, List[str]]:
        """입력 데이터 검증"""

        errors = []

        # 필수 입력 검증
        if not business_input.업종코드 or business_input.업종코드 not in self.INDUSTRY_CHOICES:
            errors.append("올바른 업종을 선택해주세요")

        if business_input.월매출 <= 0:
            errors.append("월 매출은 0원보다 커야 합니다")

        if business_input.운용자산 < 0:
            errors.append("운용자산은 0원 이상이어야 합니다")

        if business_input.업력_개월 <= 0:
            errors.append("업력은 1개월 이상이어야 합니다")

        # 선택 입력 검증
        if business_input.실제_인건비 and business_input.실제_인건비 < 0:
            errors.append("인건비는 0원 이상이어야 합니다")

        if business_input.실제_임대료 and business_input.실제_임대료 < 0:
            errors.append("임대료는 0원 이상이어야 합니다")

        # 논리적 검증
        total_specified_costs = 0
        if business_input.실제_인건비:
            total_specified_costs += business_input.실제_인건비
        if business_input.실제_임대료:
            total_specified_costs += business_input.실제_임대료
        if business_input.실제_재료비:
            total_specified_costs += business_input.실제_재료비

        if total_specified_costs > business_input.월매출 * 0.95:  # 95% 이상이면 경고
            errors.append("지정된 비용의 합이 매출의 95%를 초과합니다. 다시 확인해주세요")

        return len(errors) == 0, errors

    def get_input_guide(self) -> Dict[str, str]:
        """입력 가이드 제공"""

        return {
            "필수입력": {
                "업종": "드롭다운에서 가장 유사한 업종 선택",
                "월매출": "최근 한 달 매출액 (원 단위)",
                "운용자산": "현재 보유하고 있는 현금 및 운용 가능한 자산",
                "업력": "사업을 시작한 후 경과한 개월 수"
            },
            "선택입력": {
                "실제_인건비": "직원 급여 등 실제 인건비 (더 정확한 분석)",
                "실제_임대료": "월 임대료 (더 정확한 분석)",
                "실제_재료비": "원자재/상품 구입비 (더 정확한 분석)"
            },
            "자동계산": {
                "비용구조": "업종별 평균 비용 구조로 자동 추정",
                "매출이력": "현재 매출 기준으로 변동성 고려하여 생성",
                "위험점수": "다양한 요소를 종합하여 자동 계산"
            },
            "팁": {
                "정확도향상": "선택 입력을 더 많이 제공할수록 분석 정확도가 높아집니다",
                "업종선택": "정확한 업종을 선택하는 것이 가장 중요합니다",
                "데이터품질": "최근 6개월 매출 데이터가 있으면 더욱 정확한 분석이 가능합니다"
            }
        }

# 사용 예시 및 테스트
def main():
    interface = MinimalInputInterface()

    print("=== 업종 선택 옵션 ===")
    choices = interface.get_industry_choices()
    for code, name in list(choices.items())[:10]:
        print(f"{code}: {name}")

    print("\n=== 데모 실행 ===")
    demo_result = interface.create_simple_demo()

    print(f"업종: {demo_result['사업정보']['업종']}")
    print(f"위험등급: {demo_result['위험도평가']['위험등급']}")
    print(f"가장 큰 위험 원인: {demo_result['위험원인분석']['가장큰원인']['항목']}")

    print("\n=== 7일 현금흐름 예측 (처음 3일) ===")
    for i, day_data in enumerate(demo_result['현금흐름예측']['7일예측'][:3]):
        print(f"Day {i+1}: 매출 {day_data['predicted_revenue']:,}원, "
              f"순현금흐름 {day_data['net_cashflow']:,}원")

    print(f"\n=== 입력 가이드 ===")
    guide = interface.get_input_guide()
    print("필수 입력 항목:")
    for item, desc in guide['필수입력'].items():
        print(f"  {item}: {desc}")

if __name__ == "__main__":
    main()