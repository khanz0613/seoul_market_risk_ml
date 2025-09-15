#!/usr/bin/env python3
"""
3-Tier Investment/Loan Recommendation System
Converts risk scores to investment storytelling with product recommendations
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RecommendationTier(Enum):
    """3-Tier recommendation system"""
    INVESTMENT = "투자추천"      # 0-35 points: Investment opportunities
    BALANCED = "적정수준"        # 36-65 points: Balanced management
    SUPPORT = "지원필요"         # 66-100 points: Financial support needed

class ProductType(Enum):
    """Product categories"""
    INVESTMENT = "investment"
    LOAN = "loan"
    HYBRID = "hybrid"

@dataclass
class FinancialProduct:
    """Financial product recommendation"""
    name: str
    type: ProductType
    description: str
    risk_level: str
    expected_return: Optional[str] = None
    interest_rate: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    provider: Optional[str] = None

@dataclass
class TierRecommendation:
    """Complete tier-based recommendation"""
    tier: RecommendationTier
    narrative: str
    detailed_explanation: str
    recommended_products: List[FinancialProduct]
    action_items: List[str]
    storytelling_summary: str

class ThreeTierRecommender:
    """
    3-Tier Investment/Loan Recommendation Engine

    Transforms traditional risk scores into storytelling-based financial guidance:
    - Top tier: Investment opportunities for surplus funds
    - Middle tier: Balanced financial management
    - Bottom tier: Support products for financial stability
    """

    def __init__(self):
        self.investment_products = self._load_investment_products()
        self.loan_products = self._load_loan_products()
        self.tier_thresholds = {
            RecommendationTier.INVESTMENT: (0, 35),    # Low risk, surplus funds
            RecommendationTier.BALANCED: (36, 65),     # Moderate risk, maintain
            RecommendationTier.SUPPORT: (66, 100)      # High risk, need support
        }

    def get_recommendation(self, risk_score: float, business_data: Optional[Dict] = None) -> TierRecommendation:
        """
        Generate 3-tier recommendation based on risk score

        Args:
            risk_score: Risk score (0-100, higher = more risky)
            business_data: Optional business context for personalization

        Returns:
            TierRecommendation with narrative, products, and action items
        """
        tier = self._determine_tier(risk_score)

        # Generate base recommendation
        base_rec = self._get_base_recommendation(tier, risk_score)

        # Personalize with business context if available
        if business_data:
            base_rec = self._personalize_recommendation(base_rec, business_data, risk_score)

        return base_rec

    def _determine_tier(self, risk_score: float) -> RecommendationTier:
        """Determine which tier the risk score falls into"""
        for tier, (min_score, max_score) in self.tier_thresholds.items():
            if min_score <= risk_score <= max_score:
                return tier

        # Default to support if outside normal range
        return RecommendationTier.SUPPORT

    def _get_base_recommendation(self, tier: RecommendationTier, risk_score: float) -> TierRecommendation:
        """Generate base recommendation for each tier"""

        if tier == RecommendationTier.INVESTMENT:
            return self._create_investment_recommendation(risk_score)
        elif tier == RecommendationTier.BALANCED:
            return self._create_balanced_recommendation(risk_score)
        else:  # SUPPORT
            return self._create_support_recommendation(risk_score)

    def _create_investment_recommendation(self, risk_score: float) -> TierRecommendation:
        """Create investment-focused recommendation"""

        # Select appropriate investment products based on score
        if risk_score <= 15:  # Very safe
            products = [p for p in self.investment_products if p.risk_level in ["저위험", "초저위험"]]
        elif risk_score <= 25:  # Safe
            products = [p for p in self.investment_products if p.risk_level in ["저위험", "저중위험"]]
        else:  # Cautious (26-35)
            products = [p for p in self.investment_products if p.risk_level in ["저중위험", "중위험"]]

        narrative = self._generate_investment_narrative(risk_score)
        action_items = self._generate_investment_actions(risk_score)

        return TierRecommendation(
            tier=RecommendationTier.INVESTMENT,
            narrative=narrative,
            detailed_explanation=f"현재 위험도 {risk_score:.1f}점으로 안정적인 경영상태를 보이고 있어, 여유자금을 활용한 투자를 고려해볼 수 있는 시점입니다.",
            recommended_products=products[:3],  # Top 3 recommendations
            action_items=action_items,
            storytelling_summary=f"🌟 투자 기회의 문이 열렸습니다! 안정적인 경영 기반 위에서 자산 증식의 기회를 모색해보세요."
        )

    def _create_balanced_recommendation(self, risk_score: float) -> TierRecommendation:
        """Create balanced management recommendation"""

        # Mixed products - some conservative investments, some stability measures
        investment_products = [p for p in self.investment_products if p.risk_level in ["초저위험", "저위험"]][:2]
        loan_products = [p for p in self.loan_products if p.type == ProductType.HYBRID][:1]

        products = investment_products + loan_products

        narrative = self._generate_balanced_narrative(risk_score)
        action_items = self._generate_balanced_actions(risk_score)

        return TierRecommendation(
            tier=RecommendationTier.BALANCED,
            narrative=narrative,
            detailed_explanation=f"현재 위험도 {risk_score:.1f}점으로 적정 수준의 경영상태입니다. 안정성을 유지하면서 점진적인 성장을 도모하는 것이 바람직합니다.",
            recommended_products=products,
            action_items=action_items,
            storytelling_summary=f"⚖️ 균형잡힌 경영 상태입니다. 현재의 안정성을 바탕으로 신중한 성장 전략을 수립하세요."
        )

    def _create_support_recommendation(self, risk_score: float) -> TierRecommendation:
        """Create support-focused recommendation"""

        # Select appropriate loan/support products
        if risk_score >= 85:  # High risk
            products = [p for p in self.loan_products if "긴급" in p.name or "지원" in p.name]
        elif risk_score >= 75:  # Significant risk
            products = [p for p in self.loan_products if p.type == ProductType.LOAN and "저금리" in p.description]
        else:  # Moderate support needed (66-74)
            products = [p for p in self.loan_products if p.type in [ProductType.LOAN, ProductType.HYBRID]]

        narrative = self._generate_support_narrative(risk_score)
        action_items = self._generate_support_actions(risk_score)

        return TierRecommendation(
            tier=RecommendationTier.SUPPORT,
            narrative=narrative,
            detailed_explanation=f"현재 위험도 {risk_score:.1f}점으로 경영상 어려움이 예상됩니다. 적절한 자금 지원을 통해 경영 안정화를 도모해야 합니다.",
            recommended_products=products[:3],
            action_items=action_items,
            storytelling_summary=f"🤝 지금이 바로 지원이 필요한 시점입니다. 적절한 금융 지원으로 위기를 기회로 전환하세요."
        )

    def _generate_investment_narrative(self, risk_score: float) -> str:
        """Generate investment-focused narrative"""
        if risk_score <= 15:
            return "매우 안정적인 경영상태로 다양한 투자 기회를 적극 검토할 수 있습니다. 장기 자산 형성을 위한 포트폴리오 구성을 권장합니다."
        elif risk_score <= 25:
            return "안정적인 기반 위에서 보수적 투자를 통한 자산 증식을 고려해보세요. 안전성과 수익성의 균형을 맞춘 상품이 적합합니다."
        else:
            return "현재의 안정성을 유지하면서 신중한 투자를 통해 추가 수익을 도모할 수 있는 시점입니다."

    def _generate_balanced_narrative(self, risk_score: float) -> str:
        """Generate balanced management narrative"""
        if risk_score <= 45:
            return "전반적으로 양호한 상태이나 변동성에 대비한 안정성 강화가 필요합니다. 보수적 투자와 운영자금 확보의 균형을 맞춰가세요."
        elif risk_score <= 55:
            return "적정 수준의 위험도를 보이고 있어 현 상태 유지에 집중하면서 점진적 개선을 도모하는 것이 바람직합니다."
        else:
            return "주의가 필요한 수준입니다. 안정성 확보를 우선으로 하되, 필요시 적절한 지원을 고려해보세요."

    def _generate_support_narrative(self, risk_score: float) -> str:
        """Generate support-focused narrative"""
        if risk_score >= 85:
            return "긴급한 경영 개선이 필요한 상황입니다. 즉시 자금 지원과 경영 컨설팅을 통해 위기 극복에 집중해야 합니다."
        elif risk_score >= 75:
            return "경영상 어려움이 가시화되고 있어 적극적인 지원 대책이 필요합니다. 저금리 대출과 정부 지원 프로그램을 활용하세요."
        else:
            return "잠재적 위험 요소가 감지됩니다. 예방적 차원에서 운영자금 확보와 경영 안정화 방안을 마련하세요."

    def _generate_investment_actions(self, risk_score: float) -> List[str]:
        """Generate investment action items"""
        actions = [
            "여유자금 규모 파악 및 투자 가능 금액 산정",
            "투자 성향 및 목표 수익률 설정",
            "분산투자를 통한 리스크 관리 방안 수립"
        ]

        if risk_score <= 15:
            actions.append("장기 투자 포트폴리오 구성 검토")
            actions.append("세금 효율적 투자 상품 비교 분석")

        return actions

    def _generate_balanced_actions(self, risk_score: float) -> List[str]:
        """Generate balanced management action items"""
        return [
            "현금 흐름 안정성 점검 및 개선",
            "운영비용 최적화 방안 검토",
            "비상자금 적정 규모 유지",
            "매출 다각화 및 고객 기반 확대"
        ]

    def _generate_support_actions(self, risk_score: float) -> List[str]:
        """Generate support action items"""
        actions = [
            "긴급 자금 조달 방안 수립",
            "경영 컨설팅 및 구조조정 검토",
            "정부 지원 프로그램 신청 자격 확인"
        ]

        if risk_score >= 85:
            actions.extend([
                "채무 재조정 및 상환 계획 재검토",
                "핵심 사업 집중 및 불필요 비용 삭감"
            ])

        return actions

    def _personalize_recommendation(self, recommendation: TierRecommendation,
                                  business_data: Dict, risk_score: float) -> TierRecommendation:
        """Personalize recommendation based on business context"""

        # Add business-specific context to narrative
        business_type = business_data.get('business_type', '일반업종')
        business_name = business_data.get('business_name', '귀 업체')

        # Enhance storytelling with business context
        enhanced_summary = f"{business_name}({business_type})의 " + recommendation.storytelling_summary

        return TierRecommendation(
            tier=recommendation.tier,
            narrative=recommendation.narrative,
            detailed_explanation=recommendation.detailed_explanation,
            recommended_products=recommendation.recommended_products,
            action_items=recommendation.action_items,
            storytelling_summary=enhanced_summary
        )

    def _load_investment_products(self) -> List[FinancialProduct]:
        """Load investment product database"""
        return [
            # Ultra Low Risk
            FinancialProduct(
                name="정기예금 플러스",
                type=ProductType.INVESTMENT,
                description="원금보장 고금리 정기예금",
                risk_level="초저위험",
                expected_return="연 3.5-4.2%",
                conditions=["최소 1,000만원", "12개월 이상"],
                provider="시중은행"
            ),
            FinancialProduct(
                name="국고채 펀드",
                type=ProductType.INVESTMENT,
                description="국가 신용도 기반 안전 투자",
                risk_level="초저위험",
                expected_return="연 3.8-4.5%",
                conditions=["운용수수료 0.3%", "중도해지 가능"],
                provider="자산운용사"
            ),

            # Low Risk
            FinancialProduct(
                name="회사채 혼합형 펀드",
                type=ProductType.INVESTMENT,
                description="우량 회사채 중심 안정 수익",
                risk_level="저위험",
                expected_return="연 4.2-5.8%",
                conditions=["AA급 이상 회사채", "월별 분배금 지급"],
                provider="자산운용사"
            ),
            FinancialProduct(
                name="배당주 ETF",
                type=ProductType.INVESTMENT,
                description="배당 안정성 높은 우량주 투자",
                risk_level="저위험",
                expected_return="연 4.5-6.2%",
                conditions=["분기별 배당", "시장 연동성"],
                provider="증권사"
            ),

            # Low-Medium Risk
            FinancialProduct(
                name="글로벌 채권 펀드",
                type=ProductType.INVESTMENT,
                description="해외 우량 채권 분산투자",
                risk_level="저중위험",
                expected_return="연 5.2-7.1%",
                conditions=["환율 리스크", "운용수수료 0.8%"],
                provider="자산운용사"
            ),
            FinancialProduct(
                name="밸런스드 펀드",
                type=ProductType.INVESTMENT,
                description="주식+채권 균형 포트폴리오",
                risk_level="저중위험",
                expected_return="연 5.8-8.2%",
                conditions=["주식 50% + 채권 50%", "분기 리밸런싱"],
                provider="자산운용사"
            ),

            # Medium Risk
            FinancialProduct(
                name="코스피 인덱스 펀드",
                type=ProductType.INVESTMENT,
                description="한국 대표 주가지수 연동",
                risk_level="중위험",
                expected_return="연 6.5-9.8%",
                conditions=["시장 수익률 추종", "낮은 운용보수"],
                provider="자산운용사"
            )
        ]

    def _load_loan_products(self) -> List[FinancialProduct]:
        """Load loan product database"""
        return [
            # Emergency Support
            FinancialProduct(
                name="긴급운영자금 대출",
                type=ProductType.LOAN,
                description="즉시 지원 가능한 운영자금",
                risk_level="긴급지원",
                interest_rate="연 3.2-4.8%",
                conditions=["담보 불필요", "7일내 심사", "최대 5억원"],
                provider="정책금융공단"
            ),
            FinancialProduct(
                name="코로나19 특별 지원 대출",
                type=ProductType.LOAN,
                description="재해·재난 피해 기업 특별지원",
                risk_level="정부지원",
                interest_rate="연 1.5-2.5%",
                conditions=["매출 감소 20% 이상", "최대 10억원"],
                provider="기술보증기금"
            ),

            # Low Interest Loans
            FinancialProduct(
                name="중소기업 성장사다리 펀드",
                type=ProductType.LOAN,
                description="저금리 중소기업 전용 대출",
                risk_level="저금리",
                interest_rate="연 2.8-3.9%",
                conditions=["중소기업 인증", "신용평가 B급 이상"],
                provider="중소벤처기업진흥공단"
            ),
            FinancialProduct(
                name="신용보증 연계 대출",
                type=ProductType.LOAN,
                description="보증기관 보증서 기반 대출",
                risk_level="저금리",
                interest_rate="연 3.5-4.2%",
                conditions=["신보/기보 보증서", "최대 30억원"],
                provider="시중은행"
            ),

            # Hybrid Products
            FinancialProduct(
                name="운전자금 + 투자 패키지",
                type=ProductType.HYBRID,
                description="운영자금 지원과 여유자금 투자 결합",
                risk_level="복합상품",
                interest_rate="대출 3.8% / 투자 4-6%",
                conditions=["운전자금 70% + 투자 30%", "패키지 할인"],
                provider="종합금융회사"
            ),
            FinancialProduct(
                name="경영안정화 컨설팅 대출",
                type=ProductType.HYBRID,
                description="자금지원 + 경영컨설팅 통합 서비스",
                risk_level="종합지원",
                interest_rate="연 3.2-4.5%",
                conditions=["6개월 컨설팅 필수", "경영개선 약정"],
                provider="정책금융공단"
            )
        ]

    def format_recommendation_output(self, recommendation: TierRecommendation,
                                   business_data: Optional[Dict] = None) -> str:
        """Format recommendation for display output"""

        # Tier emoji mapping
        tier_emojis = {
            RecommendationTier.INVESTMENT: "🌟💰",
            RecommendationTier.BALANCED: "⚖️📊",
            RecommendationTier.SUPPORT: "🤝💪"
        }

        emoji = tier_emojis.get(recommendation.tier, "📋")

        output = f"\n{'='*60}\n"
        output += f"{emoji} {recommendation.tier.value} 추천\n"
        output += f"{'='*60}\n\n"

        # Storytelling Summary
        output += f"📖 스토리텔링 요약\n"
        output += f"{recommendation.storytelling_summary}\n\n"

        # Detailed Explanation
        output += f"📊 상세 분석\n"
        output += f"{recommendation.detailed_explanation}\n\n"

        # Narrative
        output += f"💭 추천 근거\n"
        output += f"{recommendation.narrative}\n\n"

        # Recommended Products
        output += f"🎯 추천 상품 ({len(recommendation.recommended_products)}개)\n"
        output += f"{'-'*40}\n"

        for i, product in enumerate(recommendation.recommended_products, 1):
            output += f"{i}. {product.name} ({product.provider})\n"
            output += f"   📝 {product.description}\n"
            output += f"   📊 위험도: {product.risk_level}\n"

            if product.expected_return:
                output += f"   💰 예상수익: {product.expected_return}\n"
            if product.interest_rate:
                output += f"   💸 금리: {product.interest_rate}\n"

            if product.conditions:
                output += f"   ✅ 조건: {' | '.join(product.conditions)}\n"
            output += "\n"

        # Action Items
        output += f"📋 실행 방안\n"
        output += f"{'-'*40}\n"
        for i, action in enumerate(recommendation.action_items, 1):
            output += f"{i}. {action}\n"

        output += f"\n{'='*60}\n"

        return output