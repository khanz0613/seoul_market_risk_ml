"""
Risk-Neutralizing Loan Calculator for Seoul Market Risk ML System
Calculates optimal loan amounts to reduce business risk scores to safe levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
import math
warnings.filterwarnings('ignore')

# Internal imports
from ..utils.config_loader import load_config
from ..risk_scoring.risk_calculator import SeoulRiskScoreCalculator, RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)


class LoanProductType(Enum):
    """Types of loan products."""
    OPERATING_CAPITAL = ("operating_capital", "운영자금대출", "일반적인 운영자금 지원")
    EMERGENCY_FUND = ("emergency_fund", "긴급자금대출", "급작스러운 자금 부족 해결")
    GROWTH_CAPITAL = ("growth_capital", "성장자금대출", "사업 확장 및 성장 지원") 
    STABILIZATION_FUND = ("stabilization_fund", "안정화자금", "매출 변동성 완화")
    CASH_FLOW_SUPPORT = ("cash_flow_support", "현금흐름지원대출", "현금 흐름 개선")
    
    def __init__(self, code: str, korean: str, description: str):
        self.code = code
        self.korean = korean
        self.description = description


class BusinessType(Enum):
    """Business types with operating fund ratios."""
    RESTAURANT = ("restaurant", "음식점", 2.5)
    RETAIL = ("retail", "소매업", 3.0)
    SERVICE = ("service", "서비스업", 1.5)
    MANUFACTURING = ("manufacturing", "제조업", 4.0)
    WHOLESALE = ("wholesale", "도매업", 2.8)
    ACCOMMODATION = ("accommodation", "숙박업", 3.5)
    OTHER = ("other", "기타", 2.0)
    
    def __init__(self, code: str, korean: str, ratio: float):
        self.code = code
        self.korean = korean
        self.operating_fund_ratio = ratio


@dataclass
class LoanProduct:
    """Individual loan product details."""
    product_type: LoanProductType
    loan_amount: float
    interest_rate: float
    loan_term_months: int
    monthly_payment: float
    total_interest: float
    risk_reduction_impact: float
    suitability_score: float
    requirements: List[str]
    benefits: List[str]


@dataclass
class RepaymentPlan:
    """Loan repayment analysis."""
    loan_amount: float
    monthly_payment: float
    loan_term_months: int
    total_repayment: float
    total_interest: float
    monthly_cash_flow_impact: float
    repayment_capacity_ratio: float  # Payment / Monthly Revenue
    risk_after_loan: float
    break_even_months: int


@dataclass
class LoanRecommendation:
    """Complete loan recommendation with risk analysis."""
    business_id: str
    current_risk_score: float
    target_risk_score: float
    required_loan_amount: float
    recommended_products: List[LoanProduct]
    repayment_analysis: RepaymentPlan
    recommendation_timestamp: str
    
    # Business context
    monthly_revenue: float
    business_type: str
    risk_factors: List[str]
    
    # Alternative scenarios
    alternative_amounts: Dict[float, float]  # {loan_amount: projected_risk_score}
    sensitivity_analysis: Dict[str, float]
    
    # Warnings and considerations
    warnings: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)


class SeoulLoanCalculator:
    """
    Risk-Neutralizing Loan Calculator for Seoul Market Risk ML System.
    
    Implements the core formula from handover report:
    - Target Risk Score: 15 (안전구간)
    - Business Type Ratios: 음식점(2.5), 소매업(3.0), 서비스업(1.5), 제조업(4.0)  
    - Maximum Loan: 연매출 한도
    - Risk Reduction Focus: 위험점수를 안전구간으로 낮추는 최적 대출 계산
    """
    
    def __init__(self, risk_calculator: Optional[SeoulRiskScoreCalculator] = None,
                 config_path: Optional[str] = None):
        
        self.config = load_config(config_path)
        self.loan_config = self.config.get('loan_calculation', {})
        
        # Risk calculator for integration
        self.risk_calculator = risk_calculator
        
        # Core parameters from handover report
        self.target_risk_score = 15.0  # 안전구간 목표
        self.max_loan_ratio = 12.0     # 최대 연매출 배수
        
        # Business type operating fund ratios (from handover report)
        self.operating_fund_ratios = {
            '음식점': 2.5,
            '소매업': 3.0,  
            '서비스업': 1.5,
            '제조업': 4.0,
            '도매업': 2.8,
            '숙박업': 3.5,
            '기타': 2.0
        }
        
        # Interest rates by product type (annual rates)
        self.interest_rates = {
            LoanProductType.OPERATING_CAPITAL: 0.045,    # 4.5%
            LoanProductType.EMERGENCY_FUND: 0.055,       # 5.5%
            LoanProductType.GROWTH_CAPITAL: 0.040,       # 4.0%
            LoanProductType.STABILIZATION_FUND: 0.048,   # 4.8%
            LoanProductType.CASH_FLOW_SUPPORT: 0.052     # 5.2%
        }
        
        # Loan terms by product type (months)
        self.loan_terms = {
            LoanProductType.OPERATING_CAPITAL: 36,       # 3년
            LoanProductType.EMERGENCY_FUND: 12,          # 1년
            LoanProductType.GROWTH_CAPITAL: 60,          # 5년
            LoanProductType.STABILIZATION_FUND: 24,      # 2년
            LoanProductType.CASH_FLOW_SUPPORT: 18        # 1.5년
        }
        
        # Risk impact factors (how much each product type reduces risk)
        self.risk_reduction_factors = {
            LoanProductType.OPERATING_CAPITAL: 0.7,      # 70% of calculated impact
            LoanProductType.EMERGENCY_FUND: 0.9,         # 90% immediate impact
            LoanProductType.GROWTH_CAPITAL: 0.5,         # 50% longer-term impact
            LoanProductType.STABILIZATION_FUND: 0.8,     # 80% volatility reduction
            LoanProductType.CASH_FLOW_SUPPORT: 0.75      # 75% cash flow improvement
        }
        
        logger.info("Loan Calculator initialized with risk-neutralizing methodology")
    
    def calculate_required_loan(self, current_risk_score: float, 
                               monthly_revenue: float, 
                               business_type: str) -> float:
        """
        Core loan calculation formula from handover report.
        
        Args:
            current_risk_score: Current business risk score (0-100)
            monthly_revenue: Monthly revenue in KRW
            business_type: Business type (Korean)
            
        Returns:
            Required loan amount to reach target risk score
        """
        # Risk reduction needed
        risk_reduction_needed = max(0, current_risk_score - self.target_risk_score)
        
        if risk_reduction_needed <= 0:
            return 0.0  # Already in safe zone
        
        # Get operating fund ratio for business type
        ratio = self.operating_fund_ratios.get(business_type, 2.0)
        
        # Calculate required loan
        required_loan = risk_reduction_needed * monthly_revenue * ratio / 100.0
        
        # Apply maximum loan limit (annual revenue)
        max_loan = monthly_revenue * self.max_loan_ratio
        
        return min(required_loan, max_loan)
    
    def generate_loan_recommendation(self, business_data: pd.DataFrame,
                                   business_id: str = "unknown",
                                   business_type: str = "기타",
                                   requested_amount: Optional[float] = None) -> LoanRecommendation:
        """
        Generate comprehensive loan recommendation with risk analysis.
        
        Args:
            business_data: Historical business data for risk assessment
            business_id: Business identifier
            business_type: Type of business (Korean)
            requested_amount: Specific loan amount requested (optional)
            
        Returns:
            Complete loan recommendation
        """
        start_time = datetime.now()
        logger.info(f"Generating loan recommendation for {business_id}")
        
        try:
            # Step 1: Calculate current risk assessment
            if self.risk_calculator:
                risk_assessment = self.risk_calculator.calculate_risk_score(
                    business_data=business_data,
                    business_id=business_id,
                    include_predictions=True,
                    include_explanations=False
                )
                current_risk_score = risk_assessment.risk_score
                risk_factors = risk_assessment.key_risk_factors
            else:
                # Fallback: simplified risk estimation
                current_risk_score = self._estimate_risk_from_data(business_data)
                risk_factors = ["위험평가 시스템 연동 필요"]
            
            # Step 2: Calculate monthly revenue
            monthly_revenue = self._calculate_monthly_revenue(business_data)
            
            if monthly_revenue <= 0:
                raise ValueError("Monthly revenue must be positive")
            
            # Step 3: Calculate required loan amount
            if requested_amount is not None:
                required_loan = requested_amount
            else:
                required_loan = self.calculate_required_loan(
                    current_risk_score=current_risk_score,
                    monthly_revenue=monthly_revenue,
                    business_type=business_type
                )
            
            # Step 4: Generate loan products
            recommended_products = self._generate_loan_products(
                required_loan, monthly_revenue, business_type, current_risk_score
            )
            
            # Step 5: Perform repayment analysis
            best_product = recommended_products[0] if recommended_products else None
            repayment_analysis = self._analyze_repayment_capacity(
                loan_amount=required_loan,
                monthly_revenue=monthly_revenue,
                current_risk_score=current_risk_score,
                product=best_product
            )
            
            # Step 6: Alternative scenarios
            alternative_amounts = self._calculate_alternative_scenarios(
                current_risk_score, monthly_revenue, business_type
            )
            
            # Step 7: Sensitivity analysis  
            sensitivity_analysis = self._perform_sensitivity_analysis(
                required_loan, monthly_revenue, business_type, current_risk_score
            )
            
            # Step 8: Generate warnings and recommendations
            warnings, success_factors, monitoring = self._generate_recommendations(
                current_risk_score, required_loan, monthly_revenue, repayment_analysis
            )
            
            # Create comprehensive recommendation
            recommendation = LoanRecommendation(
                business_id=business_id,
                current_risk_score=current_risk_score,
                target_risk_score=self.target_risk_score,
                required_loan_amount=required_loan,
                recommended_products=recommended_products,
                repayment_analysis=repayment_analysis,
                recommendation_timestamp=datetime.now().isoformat(),
                monthly_revenue=monthly_revenue,
                business_type=business_type,
                risk_factors=risk_factors[:3] if risk_factors else [],
                alternative_amounts=alternative_amounts,
                sensitivity_analysis=sensitivity_analysis,
                warnings=warnings,
                success_factors=success_factors,
                monitoring_recommendations=monitoring
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Loan recommendation completed for {business_id}: {required_loan:,.0f}원 in {processing_time:.2f}s")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Loan recommendation failed for {business_id}: {e}")
            return self._create_default_recommendation(business_id, str(e))
    
    def _calculate_monthly_revenue(self, business_data: pd.DataFrame) -> float:
        """Calculate average monthly revenue from business data."""
        # Identify revenue column
        revenue_col = None
        for col in ['monthly_revenue', 'revenue', 'sales', 'y']:
            if col in business_data.columns and not business_data[col].isna().all():
                revenue_col = col
                break
        
        if revenue_col is None:
            raise ValueError("No revenue column found in business data")
        
        # Calculate recent average (last 6 months or available data)
        recent_data = business_data.tail(6) if len(business_data) > 6 else business_data
        monthly_revenue = recent_data[revenue_col].mean()
        
        return max(0, monthly_revenue)
    
    def _estimate_risk_from_data(self, business_data: pd.DataFrame) -> float:
        """Simplified risk estimation when risk calculator not available."""
        try:
            revenue_col = None
            for col in ['monthly_revenue', 'revenue', 'sales', 'y']:
                if col in business_data.columns:
                    revenue_col = col
                    break
            
            if revenue_col is None or len(business_data) < 3:
                return 50.0  # Default moderate risk
            
            # Simple volatility-based risk estimate
            revenues = business_data[revenue_col].dropna()
            cv = revenues.std() / revenues.mean() if revenues.mean() > 0 else 0
            
            # Convert coefficient of variation to 0-100 risk score
            risk_score = min(100, cv * 200)  # Scale CV to risk score
            
            return risk_score
            
        except Exception:
            return 50.0  # Default moderate risk
    
    def _generate_loan_products(self, required_loan: float, monthly_revenue: float,
                               business_type: str, current_risk_score: float) -> List[LoanProduct]:
        """Generate suitable loan products based on requirements."""
        products = []
        
        # Determine primary loan needs based on risk level
        if current_risk_score >= 75:
            # High risk - emergency funding
            primary_types = [LoanProductType.EMERGENCY_FUND, LoanProductType.STABILIZATION_FUND]
        elif current_risk_score >= 55:
            # Medium-high risk - stabilization
            primary_types = [LoanProductType.STABILIZATION_FUND, LoanProductType.CASH_FLOW_SUPPORT]
        else:
            # Lower risk - general support
            primary_types = [LoanProductType.OPERATING_CAPITAL, LoanProductType.GROWTH_CAPITAL]
        
        # Generate products for each relevant type
        for product_type in primary_types:
            try:
                product = self._create_loan_product(
                    product_type, required_loan, monthly_revenue, business_type, current_risk_score
                )
                if product:
                    products.append(product)
            except Exception as e:
                logger.warning(f"Failed to create product {product_type}: {e}")
        
        # Add operating capital as backup option
        if not any(p.product_type == LoanProductType.OPERATING_CAPITAL for p in products):
            try:
                operating_product = self._create_loan_product(
                    LoanProductType.OPERATING_CAPITAL, required_loan, monthly_revenue, 
                    business_type, current_risk_score
                )
                if operating_product:
                    products.append(operating_product)
            except Exception:
                pass
        
        # Sort by suitability score
        products.sort(key=lambda x: x.suitability_score, reverse=True)
        
        return products[:3]  # Return top 3 products
    
    def _create_loan_product(self, product_type: LoanProductType, required_loan: float,
                            monthly_revenue: float, business_type: str, 
                            current_risk_score: float) -> Optional[LoanProduct]:
        """Create individual loan product with terms and analysis."""
        try:
            # Adjust loan amount based on product type
            if product_type == LoanProductType.EMERGENCY_FUND:
                # Emergency funds are typically smaller, faster
                loan_amount = min(required_loan, monthly_revenue * 6)
            elif product_type == LoanProductType.GROWTH_CAPITAL:
                # Growth capital can be larger
                loan_amount = min(required_loan * 1.2, monthly_revenue * 15)
            else:
                loan_amount = required_loan
            
            # Get product terms
            interest_rate = self.interest_rates.get(product_type, 0.05)
            loan_term = self.loan_terms.get(product_type, 36)
            
            # Calculate monthly payment (simple interest calculation)
            monthly_interest_rate = interest_rate / 12
            if monthly_interest_rate > 0:
                monthly_payment = loan_amount * (
                    monthly_interest_rate * (1 + monthly_interest_rate) ** loan_term
                ) / (
                    (1 + monthly_interest_rate) ** loan_term - 1
                )
            else:
                monthly_payment = loan_amount / loan_term
            
            total_interest = (monthly_payment * loan_term) - loan_amount
            
            # Calculate risk reduction impact
            risk_reduction_factor = self.risk_reduction_factors.get(product_type, 0.7)
            potential_risk_reduction = (current_risk_score - self.target_risk_score) * risk_reduction_factor
            risk_reduction_impact = max(0, current_risk_score - potential_risk_reduction)
            
            # Calculate suitability score
            suitability_score = self._calculate_suitability_score(
                product_type, loan_amount, monthly_payment, monthly_revenue, current_risk_score
            )
            
            # Generate requirements and benefits
            requirements = self._get_product_requirements(product_type, business_type)
            benefits = self._get_product_benefits(product_type, current_risk_score)
            
            return LoanProduct(
                product_type=product_type,
                loan_amount=loan_amount,
                interest_rate=interest_rate,
                loan_term_months=loan_term,
                monthly_payment=monthly_payment,
                total_interest=total_interest,
                risk_reduction_impact=risk_reduction_impact,
                suitability_score=suitability_score,
                requirements=requirements,
                benefits=benefits
            )
            
        except Exception as e:
            logger.warning(f"Failed to create loan product {product_type}: {e}")
            return None
    
    def _calculate_suitability_score(self, product_type: LoanProductType, 
                                   loan_amount: float, monthly_payment: float,
                                   monthly_revenue: float, current_risk_score: float) -> float:
        """Calculate suitability score (0-100) for loan product."""
        factors = []
        
        # Factor 1: Payment affordability (30%)
        payment_ratio = monthly_payment / monthly_revenue if monthly_revenue > 0 else 1.0
        if payment_ratio <= 0.15:  # 15% of revenue
            affordability_score = 100
        elif payment_ratio <= 0.25:  # 25% of revenue
            affordability_score = 80
        elif payment_ratio <= 0.35:  # 35% of revenue
            affordability_score = 60
        else:
            affordability_score = 30
        
        factors.append(affordability_score * 0.3)
        
        # Factor 2: Risk reduction effectiveness (40%)
        if product_type in [LoanProductType.EMERGENCY_FUND, LoanProductType.STABILIZATION_FUND]:
            if current_risk_score >= 70:
                effectiveness_score = 100
            else:
                effectiveness_score = 70
        else:
            effectiveness_score = 60
        
        factors.append(effectiveness_score * 0.4)
        
        # Factor 3: Term appropriateness (20%)
        loan_term = self.loan_terms.get(product_type, 36)
        if current_risk_score >= 70 and loan_term <= 18:  # Short term for high risk
            term_score = 100
        elif current_risk_score < 70 and loan_term >= 24:  # Longer term for lower risk
            term_score = 100
        else:
            term_score = 70
        
        factors.append(term_score * 0.2)
        
        # Factor 4: Interest rate competitiveness (10%)
        interest_rate = self.interest_rates.get(product_type, 0.05)
        if interest_rate <= 0.045:
            rate_score = 100
        elif interest_rate <= 0.055:
            rate_score = 80
        else:
            rate_score = 60
        
        factors.append(rate_score * 0.1)
        
        return sum(factors)
    
    def _get_product_requirements(self, product_type: LoanProductType, business_type: str) -> List[str]:
        """Get requirements for loan product."""
        base_requirements = [
            "사업자등록증",
            "최근 6개월 매출 증빙서류",
            "재무제표 또는 소득신고서"
        ]
        
        if product_type == LoanProductType.EMERGENCY_FUND:
            base_requirements.extend([
                "긴급자금 필요 사유서",
                "현금흐름 계획서"
            ])
        elif product_type == LoanProductType.GROWTH_CAPITAL:
            base_requirements.extend([
                "사업계획서",
                "성장계획 및 자금운용계획서"
            ])
        elif product_type == LoanProductType.STABILIZATION_FUND:
            base_requirements.extend([
                "매출 변동 현황 분석서",
                "안정화 방안 계획서"
            ])
        
        return base_requirements
    
    def _get_product_benefits(self, product_type: LoanProductType, current_risk_score: float) -> List[str]:
        """Get benefits for loan product."""
        benefits = []
        
        if product_type == LoanProductType.EMERGENCY_FUND:
            benefits = [
                "신속한 자금 지원 (3일 내)",
                "단기 현금흐름 개선",
                "경영 안정성 향상"
            ]
        elif product_type == LoanProductType.OPERATING_CAPITAL:
            benefits = [
                "운영자금 안정적 확보",
                "매출 기반 확대 지원",
                "경쟁력 강화"
            ]
        elif product_type == LoanProductType.GROWTH_CAPITAL:
            benefits = [
                "사업 확장 자금 지원",
                "장기 성장 동력 확보",
                "시장 점유율 확대"
            ]
        elif product_type == LoanProductType.STABILIZATION_FUND:
            benefits = [
                "매출 변동성 완화",
                "위험도 체계적 관리",
                "안정적 운영 기반 구축"
            ]
        elif product_type == LoanProductType.CASH_FLOW_SUPPORT:
            benefits = [
                "현금흐름 개선",
                "운영 효율성 증대",
                "재무 건전성 강화"
            ]
        
        # Add risk-specific benefits
        if current_risk_score >= 70:
            benefits.append("고위험 상황 조기 해결")
        elif current_risk_score >= 50:
            benefits.append("중위험 단계 안정화")
        else:
            benefits.append("예방적 리스크 관리")
        
        return benefits
    
    def _analyze_repayment_capacity(self, loan_amount: float, monthly_revenue: float,
                                  current_risk_score: float, 
                                  product: Optional[LoanProduct] = None) -> RepaymentPlan:
        """Analyze repayment capacity and impact."""
        if product:
            monthly_payment = product.monthly_payment
            loan_term = product.loan_term_months
            total_interest = product.total_interest
        else:
            # Default terms
            annual_rate = 0.05
            loan_term = 36
            monthly_rate = annual_rate / 12
            monthly_payment = loan_amount * (
                monthly_rate * (1 + monthly_rate) ** loan_term
            ) / (
                (1 + monthly_rate) ** loan_term - 1
            )
            total_interest = (monthly_payment * loan_term) - loan_amount
        
        total_repayment = loan_amount + total_interest
        
        # Calculate cash flow impact
        monthly_cash_flow_impact = -monthly_payment  # Negative impact
        repayment_capacity_ratio = monthly_payment / monthly_revenue if monthly_revenue > 0 else 1.0
        
        # Estimate risk after loan (simplified)
        risk_reduction = min(20, (current_risk_score - self.target_risk_score) * 0.7)
        risk_after_loan = max(self.target_risk_score, current_risk_score - risk_reduction)
        
        # Calculate break-even months (when risk reduction benefits outweigh costs)
        monthly_revenue_increase = monthly_revenue * 0.02  # Assume 2% improvement from stability
        if monthly_revenue_increase > 0:
            break_even_months = int(loan_amount / monthly_revenue_increase)
        else:
            break_even_months = loan_term
        
        return RepaymentPlan(
            loan_amount=loan_amount,
            monthly_payment=monthly_payment,
            loan_term_months=loan_term,
            total_repayment=total_repayment,
            total_interest=total_interest,
            monthly_cash_flow_impact=monthly_cash_flow_impact,
            repayment_capacity_ratio=repayment_capacity_ratio,
            risk_after_loan=risk_after_loan,
            break_even_months=break_even_months
        )
    
    def _calculate_alternative_scenarios(self, current_risk_score: float,
                                       monthly_revenue: float, 
                                       business_type: str) -> Dict[float, float]:
        """Calculate alternative loan amounts and their risk impacts."""
        base_amount = self.calculate_required_loan(current_risk_score, monthly_revenue, business_type)
        
        scenarios = {}
        
        # Different loan amounts (50%, 75%, 100%, 125%, 150% of required)
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        for multiplier in multipliers:
            loan_amount = base_amount * multiplier
            
            # Estimate risk reduction (simplified linear model)
            max_reduction = current_risk_score - self.target_risk_score
            actual_reduction = min(max_reduction, max_reduction * multiplier * 0.8)
            projected_risk = max(self.target_risk_score, current_risk_score - actual_reduction)
            
            scenarios[loan_amount] = projected_risk
        
        return scenarios
    
    def _perform_sensitivity_analysis(self, required_loan: float, monthly_revenue: float,
                                     business_type: str, current_risk_score: float) -> Dict[str, float]:
        """Perform sensitivity analysis on key variables."""
        sensitivity = {}
        
        # Revenue sensitivity (±20%)
        revenue_high = self.calculate_required_loan(current_risk_score, monthly_revenue * 1.2, business_type)
        revenue_low = self.calculate_required_loan(current_risk_score, monthly_revenue * 0.8, business_type)
        sensitivity['revenue_+20%'] = (revenue_high - required_loan) / required_loan * 100
        sensitivity['revenue_-20%'] = (revenue_low - required_loan) / required_loan * 100
        
        # Risk score sensitivity (±10 points)
        risk_high = self.calculate_required_loan(current_risk_score + 10, monthly_revenue, business_type)
        risk_low = self.calculate_required_loan(max(0, current_risk_score - 10), monthly_revenue, business_type)
        sensitivity['risk_+10pts'] = (risk_high - required_loan) / required_loan * 100 if required_loan > 0 else 0
        sensitivity['risk_-10pts'] = (risk_low - required_loan) / required_loan * 100 if required_loan > 0 else 0
        
        return sensitivity
    
    def _generate_recommendations(self, current_risk_score: float, required_loan: float,
                                monthly_revenue: float, repayment_analysis: RepaymentPlan) -> Tuple[List[str], List[str], List[str]]:
        """Generate warnings, success factors, and monitoring recommendations."""
        warnings = []
        success_factors = []
        monitoring = []
        
        # Warnings
        if repayment_analysis.repayment_capacity_ratio > 0.3:
            warnings.append("월 상환금이 매출 대비 30% 초과 - 상환 부담 높음")
        
        if current_risk_score >= 75:
            warnings.append("현재 위험도가 매우 높음 - 대출만으로는 근본적 해결 어려움")
        
        if required_loan > monthly_revenue * 10:
            warnings.append("대출 규모가 매우 큼 - 신중한 검토 필요")
        
        # Success factors
        if repayment_analysis.repayment_capacity_ratio <= 0.2:
            success_factors.append("상환 여력 충분 - 안정적 상환 가능")
        
        if repayment_analysis.break_even_months <= 24:
            success_factors.append("투자 회수 기간 양호 - 2년 내 효과 기대")
        
        success_factors.extend([
            "체계적인 위험 관리 접근",
            "데이터 기반 대출 규모 산정",
            "맞춤형 상품 추천"
        ])
        
        # Monitoring recommendations
        monitoring.extend([
            "월별 매출 및 현금흐름 모니터링",
            "분기별 위험점수 재평가",
            "상환계획 대비 실행 현황 추적"
        ])
        
        if current_risk_score >= 50:
            monitoring.append("고위험 요소 집중 관찰")
        
        return warnings, success_factors, monitoring
    
    def _create_default_recommendation(self, business_id: str, error_message: str) -> LoanRecommendation:
        """Create default recommendation when calculation fails."""
        return LoanRecommendation(
            business_id=business_id,
            current_risk_score=50.0,
            target_risk_score=self.target_risk_score,
            required_loan_amount=0.0,
            recommended_products=[],
            repayment_analysis=RepaymentPlan(
                loan_amount=0.0,
                monthly_payment=0.0,
                loan_term_months=0,
                total_repayment=0.0,
                total_interest=0.0,
                monthly_cash_flow_impact=0.0,
                repayment_capacity_ratio=0.0,
                risk_after_loan=50.0,
                break_even_months=0
            ),
            recommendation_timestamp=datetime.now().isoformat(),
            monthly_revenue=0.0,
            business_type="알 수 없음",
            risk_factors=[],
            alternative_amounts={},
            sensitivity_analysis={},
            warnings=[f"대출 계산 오류: {error_message}"],
            success_factors=[],
            monitoring_recommendations=["데이터 보완 후 재평가 필요"]
        )
    
    def batch_calculate_loans(self, business_data_list: List[Tuple[str, pd.DataFrame, str]]) -> List[LoanRecommendation]:
        """
        Calculate loan recommendations for multiple businesses.
        
        Args:
            business_data_list: List of (business_id, data, business_type) tuples
            
        Returns:
            List of loan recommendations
        """
        logger.info(f"Calculating loan recommendations for {len(business_data_list)} businesses")
        
        results = []
        for business_id, business_data, business_type in business_data_list:
            try:
                recommendation = self.generate_loan_recommendation(
                    business_data=business_data,
                    business_id=business_id,
                    business_type=business_type
                )
                results.append(recommendation)
            except Exception as e:
                logger.error(f"Failed to calculate loan for {business_id}: {e}")
                results.append(self._create_default_recommendation(business_id, str(e)))
        
        return results
    
    def export_recommendation_json(self, recommendation: LoanRecommendation) -> str:
        """Export loan recommendation to JSON format."""
        rec_dict = {
            'business_id': recommendation.business_id,
            'risk_analysis': {
                'current_risk_score': recommendation.current_risk_score,
                'target_risk_score': recommendation.target_risk_score,
                'risk_factors': recommendation.risk_factors
            },
            'loan_calculation': {
                'required_amount': recommendation.required_loan_amount,
                'monthly_revenue': recommendation.monthly_revenue,
                'business_type': recommendation.business_type
            },
            'recommended_products': [
                {
                    'type': prod.product_type.korean,
                    'amount': prod.loan_amount,
                    'interest_rate': prod.interest_rate * 100,
                    'term_months': prod.loan_term_months,
                    'monthly_payment': prod.monthly_payment,
                    'suitability_score': prod.suitability_score,
                    'requirements': prod.requirements,
                    'benefits': prod.benefits
                }
                for prod in recommendation.recommended_products
            ],
            'repayment_analysis': {
                'monthly_payment': recommendation.repayment_analysis.monthly_payment,
                'total_repayment': recommendation.repayment_analysis.total_repayment,
                'repayment_capacity_ratio': recommendation.repayment_analysis.repayment_capacity_ratio,
                'risk_after_loan': recommendation.repayment_analysis.risk_after_loan,
                'break_even_months': recommendation.repayment_analysis.break_even_months
            },
            'alternative_scenarios': recommendation.alternative_amounts,
            'sensitivity_analysis': recommendation.sensitivity_analysis,
            'recommendations': {
                'warnings': recommendation.warnings,
                'success_factors': recommendation.success_factors,
                'monitoring': recommendation.monitoring_recommendations
            },
            'timestamp': recommendation.recommendation_timestamp
        }
        
        return json.dumps(rec_dict, ensure_ascii=False, indent=2)


def main():
    """Main function for testing loan calculator."""
    print("\n=== LOAN CALCULATOR TEST ===")
    
    # Initialize calculator
    calculator = SeoulLoanCalculator()
    print(f"Loan Calculator initialized")
    print(f"Target risk score: {calculator.target_risk_score}")
    print(f"Operating fund ratios: {calculator.operating_fund_ratios}")
    
    # Test core loan calculation
    current_risk = 68.5
    monthly_revenue = 15000000  # 1,500만원
    business_type = "음식점"
    
    required_loan = calculator.calculate_required_loan(
        current_risk_score=current_risk,
        monthly_revenue=monthly_revenue,
        business_type=business_type
    )
    
    print(f"\n=== CORE CALCULATION TEST ===")
    print(f"Current risk score: {current_risk}")
    print(f"Monthly revenue: {monthly_revenue:,}원")
    print(f"Business type: {business_type}")
    print(f"Required loan: {required_loan:,.0f}원")
    
    # Test with sample business data
    sample_data = pd.DataFrame({
        'ds': pd.date_range('2023-01-01', periods=12, freq='M'),
        'monthly_revenue': [monthly_revenue + np.random.normal(0, 1000000) for _ in range(12)]
    })
    
    # Generate comprehensive recommendation
    recommendation = calculator.generate_loan_recommendation(
        business_data=sample_data,
        business_id="test_restaurant_001", 
        business_type=business_type
    )
    
    print(f"\n=== LOAN RECOMMENDATION ===")
    print(f"Business: {recommendation.business_id}")
    print(f"Risk: {recommendation.current_risk_score:.1f} → {recommendation.target_risk_score:.1f}")
    print(f"Required loan: {recommendation.required_loan_amount:,.0f}원")
    
    if recommendation.recommended_products:
        best_product = recommendation.recommended_products[0]
        print(f"\nBest product: {best_product.product_type.korean}")
        print(f"  Amount: {best_product.loan_amount:,.0f}원")
        print(f"  Rate: {best_product.interest_rate*100:.1f}%")
        print(f"  Term: {best_product.loan_term_months}개월")
        print(f"  Monthly payment: {best_product.monthly_payment:,.0f}원")
        print(f"  Suitability: {best_product.suitability_score:.1f}점")
    
    print(f"\nRepayment analysis:")
    rep = recommendation.repayment_analysis
    print(f"  Repayment ratio: {rep.repayment_capacity_ratio*100:.1f}%")
    print(f"  Risk after loan: {rep.risk_after_loan:.1f}")
    print(f"  Break-even: {rep.break_even_months}개월")
    
    if recommendation.warnings:
        print(f"\nWarnings: {len(recommendation.warnings)}")
        for warning in recommendation.warnings[:2]:
            print(f"  - {warning}")
    
    print("\n=== LOAN CALCULATOR READY ===")


if __name__ == "__main__":
    main()