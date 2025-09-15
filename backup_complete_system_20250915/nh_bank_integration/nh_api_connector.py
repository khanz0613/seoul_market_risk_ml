"""
NH Bank API Connector for Proactive Financial Services
Integrates with NH Bank API to provide loan product recommendations based on opportunity scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
import warnings
from dataclasses import dataclass, field
from enum import Enum
import math
warnings.filterwarnings('ignore')

# Internal imports
from ..utils.config_loader import load_config
from ..risk_scoring.risk_calculator import SeoulOpportunityScoreCalculator, OpportunityAssessment, OpportunityLevel

logger = logging.getLogger(__name__)


class NHLoanProductCategory(Enum):
    """NH Bank loan product categories."""
    EMERGENCY_LOAN = ("emergency", "긴급자금대출", "Emergency funding for critical cash flow issues")
    STABILIZATION_LOAN = ("stabilization", "안정화자금대출", "Stabilization funding for business recovery")
    GROWTH_INVESTMENT = ("growth", "성장투자대출", "Growth investment for expanding businesses")
    WORKING_CAPITAL = ("working_capital", "운영자금대출", "Working capital for daily operations")
    
    def __init__(self, code: str, korean: str, description: str):
        self.code = code
        self.korean = korean
        self.description = description


class NHInvestmentProductCategory(Enum):
    """NH Bank investment product categories."""
    SAVINGS_ACCOUNT = ("savings", "예금상품", "High-yield savings products")
    INVESTMENT_FUND = ("fund", "펀드상품", "Diversified investment funds")
    BOND_INVESTMENT = ("bond", "채권투자", "Corporate and government bonds")
    EQUITY_INVESTMENT = ("equity", "주식투자", "Stock market investments")
    MIXED_PORTFOLIO = ("mixed", "혼합포트폴리오", "Balanced portfolio investments")
    
    def __init__(self, code: str, korean: str, description: str):
        self.code = code
        self.korean = korean
        self.description = description


@dataclass
class NHLoanProduct:
    """NH Bank loan product information."""
    product_id: str
    product_name: str
    product_category: NHLoanProductCategory
    interest_rate_min: float  # Minimum annual interest rate
    interest_rate_max: float  # Maximum annual interest rate
    loan_amount_min: int      # Minimum loan amount in KRW
    loan_amount_max: int      # Maximum loan amount in KRW
    loan_term_min: int        # Minimum term in months
    loan_term_max: int        # Maximum term in months
    
    # Eligibility criteria
    min_monthly_revenue: int = 1000000    # Minimum monthly revenue requirement
    max_debt_ratio: float = 0.4           # Maximum debt-to-income ratio
    min_business_age_months: int = 6      # Minimum business operation period
    
    # Additional product features
    collateral_required: bool = False
    guarantor_required: bool = False
    early_repayment_fee: float = 0.0
    processing_fee: float = 0.005         # Processing fee as percentage
    
    # Matching criteria for recommendation
    opportunity_levels: List[OpportunityLevel] = field(default_factory=list)
    business_types: List[str] = field(default_factory=list)


@dataclass
class NHInvestmentProduct:
    """NH Bank investment product information."""
    product_id: str
    product_name: str
    product_category: NHInvestmentProductCategory
    expected_return_min: float    # Minimum expected annual return
    expected_return_max: float    # Maximum expected annual return
    risk_level: int              # Risk level 1-5 (1=lowest risk)
    minimum_investment: int      # Minimum investment amount in KRW
    
    # Investment terms
    lock_in_period_months: int = 0       # Minimum investment period
    early_withdrawal_penalty: float = 0.0
    management_fee: float = 0.01         # Annual management fee
    
    # Matching criteria
    opportunity_levels: List[OpportunityLevel] = field(default_factory=list)
    investment_potential_threshold: float = 50.0  # Minimum investment potential score


@dataclass
class NHRecommendationResult:
    """Result of NH Bank product recommendation."""
    business_id: str
    opportunity_assessment: OpportunityAssessment
    
    # Loan recommendations
    recommended_loans: List[NHLoanProduct] = field(default_factory=list)
    optimal_loan_amount: float = 0.0
    loan_rationale: str = ""
    
    # Investment recommendations  
    recommended_investments: List[NHInvestmentProduct] = field(default_factory=list)
    optimal_investment_amount: float = 0.0
    investment_rationale: str = ""
    
    # API response metadata
    api_response_timestamp: str = ""
    api_call_success: bool = True
    api_error_message: str = ""


class NHBankAPIConnector:
    """
    NH Bank API Connector for Proactive Financial Services.
    
    Integrates opportunity scoring with NH Bank's loan and investment products
    to provide personalized financial recommendations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.nh_config = self.config.get('nh_bank_api', {})
        
        # API configuration
        self.api_base_url = self.nh_config.get('base_url', 'https://api.nhbank.co.kr/v1/')
        self.api_key = self.nh_config.get('api_key', '')
        self.client_id = self.nh_config.get('client_id', '')
        self.timeout = self.nh_config.get('timeout', 30)
        
        # Product catalogs (will be loaded from API or config)
        self.loan_products: List[NHLoanProduct] = []
        self.investment_products: List[NHInvestmentProduct] = []
        
        # Initialize default products if API is not available
        self._initialize_default_products()
        
        logger.info(f"NH Bank API Connector initialized")
    
    def _initialize_default_products(self):
        """Initialize default product catalog for demonstration/fallback."""
        
        # Default loan products
        self.loan_products = [
            NHLoanProduct(
                product_id="NH_EMERGENCY_001",
                product_name="NH 긴급운영자금대출",
                product_category=NHLoanProductCategory.EMERGENCY_LOAN,
                interest_rate_min=0.045, interest_rate_max=0.085,
                loan_amount_min=5000000, loan_amount_max=200000000,
                loan_term_min=6, loan_term_max=36,
                min_monthly_revenue=5000000, max_debt_ratio=0.5,
                opportunity_levels=[OpportunityLevel.VERY_HIGH_RISK, OpportunityLevel.HIGH_RISK]
            ),
            NHLoanProduct(
                product_id="NH_STABILIZATION_001",
                product_name="NH 사업안정화자금",
                product_category=NHLoanProductCategory.STABILIZATION_LOAN,
                interest_rate_min=0.035, interest_rate_max=0.065,
                loan_amount_min=10000000, loan_amount_max=500000000,
                loan_term_min=12, loan_term_max=60,
                min_monthly_revenue=8000000, max_debt_ratio=0.4,
                opportunity_levels=[OpportunityLevel.HIGH_RISK, OpportunityLevel.MODERATE]
            ),
            NHLoanProduct(
                product_id="NH_GROWTH_001",
                product_name="NH 성장기업지원대출",
                product_category=NHLoanProductCategory.GROWTH_INVESTMENT,
                interest_rate_min=0.025, interest_rate_max=0.045,
                loan_amount_min=50000000, loan_amount_max=1000000000,
                loan_term_min=24, loan_term_max=84,
                min_monthly_revenue=20000000, max_debt_ratio=0.3,
                opportunity_levels=[OpportunityLevel.GOOD, OpportunityLevel.VERY_GOOD]
            )
        ]
        
        # Default investment products
        self.investment_products = [
            NHInvestmentProduct(
                product_id="NH_SAVINGS_001",
                product_name="NH 고수익정기예금",
                product_category=NHInvestmentProductCategory.SAVINGS_ACCOUNT,
                expected_return_min=0.035, expected_return_max=0.045,
                risk_level=1, minimum_investment=10000000,
                lock_in_period_months=12,
                opportunity_levels=[OpportunityLevel.MODERATE, OpportunityLevel.GOOD],
                investment_potential_threshold=30.0
            ),
            NHInvestmentProduct(
                product_id="NH_FUND_001",
                product_name="NH 균형성장펀드",
                product_category=NHInvestmentProductCategory.INVESTMENT_FUND,
                expected_return_min=0.055, expected_return_max=0.085,
                risk_level=3, minimum_investment=5000000,
                management_fee=0.015,
                opportunity_levels=[OpportunityLevel.GOOD, OpportunityLevel.VERY_GOOD],
                investment_potential_threshold=60.0
            ),
            NHInvestmentProduct(
                product_id="NH_EQUITY_001",
                product_name="NH 프리미엄주식투자",
                product_category=NHInvestmentProductCategory.EQUITY_INVESTMENT,
                expected_return_min=0.08, expected_return_max=0.15,
                risk_level=4, minimum_investment=20000000,
                management_fee=0.025,
                opportunity_levels=[OpportunityLevel.VERY_GOOD],
                investment_potential_threshold=85.0
            )
        ]
    
    def get_personalized_recommendations(self, 
                                       opportunity_assessment: OpportunityAssessment,
                                       max_loan_products: int = 3,
                                       max_investment_products: int = 3) -> NHRecommendationResult:
        """
        Get personalized loan and investment recommendations based on opportunity assessment.
        
        Args:
            opportunity_assessment: Business opportunity assessment
            max_loan_products: Maximum number of loan products to recommend
            max_investment_products: Maximum number of investment products to recommend
            
        Returns:
            Complete recommendation result
        """
        logger.info(f"Generating NH Bank recommendations for {opportunity_assessment.business_id}")
        
        result = NHRecommendationResult(
            business_id=opportunity_assessment.business_id,
            opportunity_assessment=opportunity_assessment,
            api_response_timestamp=datetime.now().isoformat()
        )
        
        try:
            # Get loan recommendations if needed
            if opportunity_assessment.loan_necessity > 0:
                result.recommended_loans, result.optimal_loan_amount, result.loan_rationale = (
                    self._get_loan_recommendations(opportunity_assessment, max_loan_products)
                )
            
            # Get investment recommendations if potential exists
            if opportunity_assessment.investment_potential > 0:
                result.recommended_investments, result.optimal_investment_amount, result.investment_rationale = (
                    self._get_investment_recommendations(opportunity_assessment, max_investment_products)
                )
            
            result.api_call_success = True
            
        except Exception as e:
            logger.error(f"Failed to generate NH Bank recommendations: {e}")
            result.api_call_success = False
            result.api_error_message = str(e)
        
        return result
    
    def _get_loan_recommendations(self, 
                                 assessment: OpportunityAssessment, 
                                 max_products: int) -> Tuple[List[NHLoanProduct], float, str]:
        """Get loan product recommendations."""
        
        # Filter products by opportunity level
        suitable_products = [
            product for product in self.loan_products
            if assessment.opportunity_level in product.opportunity_levels
        ]
        
        # Estimate monthly revenue from assessment data
        monthly_revenue = self._estimate_monthly_revenue_from_assessment(assessment)
        
        # Filter by financial criteria
        qualified_products = []
        for product in suitable_products:
            if (monthly_revenue >= product.min_monthly_revenue and
                assessment.loan_necessity >= product.loan_amount_min and
                assessment.loan_necessity <= product.loan_amount_max):
                qualified_products.append(product)
        
        # Sort by suitability (lower interest rate = higher priority)
        qualified_products.sort(key=lambda p: p.interest_rate_min)
        
        # Select top products
        recommended_products = qualified_products[:max_products]
        
        # Calculate optimal loan amount
        optimal_amount = min(assessment.loan_necessity, 
                           max([p.loan_amount_max for p in recommended_products], default=0))
        
        # Generate rationale
        if assessment.opportunity_level == OpportunityLevel.VERY_HIGH_RISK:
            rationale = "긴급 현금흐름 안정화를 위한 즉시 자금 지원이 필요합니다."
        elif assessment.opportunity_level == OpportunityLevel.HIGH_RISK:
            rationale = "사업 안정화를 통해 적정 수준까지 개선하기 위한 자금 지원이 권장됩니다."
        else:
            rationale = "성장 기회 활용을 위한 전략적 자금 조달을 검토할 수 있습니다."
        
        return recommended_products, optimal_amount, rationale
    
    def _get_investment_recommendations(self, 
                                      assessment: OpportunityAssessment, 
                                      max_products: int) -> Tuple[List[NHInvestmentProduct], float, str]:
        """Get investment product recommendations."""
        
        # Filter products by opportunity level and investment potential
        suitable_products = [
            product for product in self.investment_products
            if (assessment.opportunity_level in product.opportunity_levels and
                assessment.investment_potential >= product.investment_potential_threshold)
        ]
        
        # Estimate available investment amount based on business performance
        monthly_revenue = self._estimate_monthly_revenue_from_assessment(assessment)
        available_investment = monthly_revenue * (assessment.investment_potential / 100) * 3  # 3 months worth
        
        # Filter by minimum investment requirements
        qualified_products = [
            product for product in suitable_products
            if available_investment >= product.minimum_investment
        ]
        
        # Sort by expected return (descending) but consider risk level
        qualified_products.sort(key=lambda p: (p.expected_return_max - p.risk_level * 0.01), reverse=True)
        
        # Select top products
        recommended_products = qualified_products[:max_products]
        
        # Calculate optimal investment amount
        optimal_amount = available_investment * 0.7  # Conservative approach - 70% of available funds
        
        # Generate rationale
        if assessment.opportunity_level == OpportunityLevel.VERY_GOOD:
            rationale = "뛰어난 사업 성과를 바탕으로 고수익 투자 기회를 활용하실 수 있습니다."
        elif assessment.opportunity_level == OpportunityLevel.GOOD:
            rationale = "안정적인 성장세를 기반으로 포트폴리오 다변화 투자를 권장합니다."
        else:
            rationale = "여유 자금 활용을 통한 안전한 자산 증식 기회를 제공합니다."
        
        return recommended_products, optimal_amount, rationale
    
    def _estimate_monthly_revenue_from_assessment(self, assessment: OpportunityAssessment) -> float:
        """Estimate monthly revenue from opportunity assessment."""
        
        # Look for revenue indicators in assessment data
        if hasattr(assessment, 'revenue_forecast') and assessment.revenue_forecast is not None:
            try:
                recent_revenue = assessment.revenue_forecast['ensemble_pred'].iloc[-1]
                return max(5000000, recent_revenue)  # Minimum 5M KRW
            except:
                pass
        
        # Use component analysis to estimate revenue
        revenue_comp = next((c for c in assessment.components if c.name == 'revenue_change'), None)
        if revenue_comp:
            # Higher percentile suggests higher revenue
            base_revenue = 10000000  # 10M KRW base
            revenue_multiplier = 1 + (revenue_comp.percentile / 100)
            return base_revenue * revenue_multiplier
        
        # Default estimation based on opportunity score
        base_revenue = 8000000  # 8M KRW default
        score_multiplier = 1 + (assessment.opportunity_score / 200)  # Score-based adjustment
        return base_revenue * score_multiplier
    
    def call_nh_api(self, endpoint: str, method: str = 'GET', 
                   data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make API call to NH Bank API (placeholder for actual implementation).
        
        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            data: Request data for POST requests
            
        Returns:
            API response data
        """
        # Placeholder implementation - replace with actual API integration
        logger.info(f"Mock NH Bank API call: {method} {endpoint}")
        
        # Simulate API response
        return {
            "status": "success",
            "data": [],
            "timestamp": datetime.now().isoformat(),
            "mock_response": True
        }
    
    def refresh_product_catalog(self) -> bool:
        """
        Refresh loan and investment product catalog from NH Bank API.
        
        Returns:
            True if refresh was successful
        """
        try:
            # Placeholder for actual API integration
            logger.info("Refreshing NH Bank product catalog")
            
            # In actual implementation, this would:
            # 1. Call NH Bank product catalog API
            # 2. Parse response and update self.loan_products and self.investment_products
            # 3. Cache the results for performance
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh NH Bank product catalog: {e}")
            return False


def main():
    """Main function for testing NH Bank API connector."""
    print("\n=== NH BANK API CONNECTOR TEST ===")
    
    # Initialize connector
    connector = NHBankAPIConnector()
    print(f"NH Bank API Connector initialized")
    print(f"Loan products available: {len(connector.loan_products)}")
    print(f"Investment products available: {len(connector.investment_products)}")
    
    # Create mock opportunity assessment for testing
    from ..risk_scoring.risk_calculator import OpportunityAssessment, RiskComponent
    
    mock_components = [
        RiskComponent("revenue_change", "매출변화율", 45.0, 0.3, 13.5, 60.0, "보통", "매출 변화율 보통 수준"),
        RiskComponent("volatility", "변동성", 35.0, 0.2, 7.0, 75.0, "양호", "변동성 관리 양호"),
        RiskComponent("trend", "트렌드", 25.0, 0.2, 5.0, 80.0, "우수", "성장 트렌드 우수"),
        RiskComponent("seasonality_deviation", "계절성이탈", 40.0, 0.15, 6.0, 55.0, "보통", "계절성 패턴 보통"),
        RiskComponent("industry_comparison", "업종비교", 30.0, 0.15, 4.5, 70.0, "양호", "업종 평균 대비 양호")
    ]
    
    mock_assessment = OpportunityAssessment(
        business_id="test_business_001",
        opportunity_score=65.0,
        opportunity_level=OpportunityLevel.GOOD,
        confidence_score=0.85,
        assessment_timestamp=datetime.now().isoformat(),
        recommended_action="growth_investment",
        loan_necessity=0.0,
        investment_potential=75.0,
        components=mock_components,
        component_scores={comp.name: comp.score for comp in mock_components},
        key_opportunity_factors=["성장 트렌드 우수", "업종 대비 우위"],
        stability_factors=["변동성 관리 양호"],
        recommendations=["성장투자 기회 활용", "포트폴리오 다변화 검토"]
    )
    
    print(f"\n=== RECOMMENDATION TEST ===")
    print(f"Test business: {mock_assessment.business_id}")
    print(f"Opportunity score: {mock_assessment.opportunity_score}")
    print(f"Opportunity level: {mock_assessment.opportunity_level.korean}")
    print(f"Investment potential: {mock_assessment.investment_potential}")
    
    # Get recommendations
    recommendations = connector.get_personalized_recommendations(mock_assessment)
    
    print(f"\n=== RECOMMENDATION RESULTS ===")
    print(f"API call success: {recommendations.api_call_success}")
    
    if recommendations.recommended_loans:
        print(f"\nLoan Recommendations ({len(recommendations.recommended_loans)}):")
        for loan in recommendations.recommended_loans:
            print(f"  - {loan.product_name}")
            print(f"    금리: {loan.interest_rate_min*100:.1f}% ~ {loan.interest_rate_max*100:.1f}%")
            print(f"    한도: {loan.loan_amount_min:,} ~ {loan.loan_amount_max:,} KRW")
    
    if recommendations.recommended_investments:
        print(f"\nInvestment Recommendations ({len(recommendations.recommended_investments)}):")
        for investment in recommendations.recommended_investments:
            print(f"  - {investment.product_name}")
            print(f"    예상수익률: {investment.expected_return_min*100:.1f}% ~ {investment.expected_return_max*100:.1f}%")
            print(f"    위험등급: {investment.risk_level}/5")
            print(f"    최소투자금: {investment.minimum_investment:,} KRW")
        
        if recommendations.optimal_investment_amount > 0:
            print(f"\n권장 투자금액: {recommendations.optimal_investment_amount:,.0f} KRW")
            print(f"투자 근거: {recommendations.investment_rationale}")
    
    print("\n=== NH BANK API CONNECTOR READY ===")


if __name__ == "__main__":
    main()