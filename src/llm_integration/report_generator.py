"""
LLM Auto-Reporting System for Seoul Market Risk ML System
Automated business risk report generation with dual format structure.
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
import re
warnings.filterwarnings('ignore')

# Internal imports
from ..utils.config_loader import load_config
from ..risk_scoring.risk_calculator import OpportunityAssessment, OpportunityLevel
from ..risk_scoring.changepoint_detection import ChangePointSummary, ChangePoint, ChangeType, AlertLevel
from ..loan_calculation.loan_calculator import LoanRecommendation, LoanProduct
from ..nh_bank_integration.nh_api_connector import NHBankAPIConnector, NHRecommendationResult

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of generated reports."""
    SUMMARY = ("summary", "간단 요약", 200)
    DETAILED = ("detailed", "상세 분석", 1000)
    EXECUTIVE = ("executive", "경영진 보고서", 1500)
    TECHNICAL = ("technical", "기술 분석", 800)
    
    def __init__(self, code: str, korean: str, max_chars: int):
        self.code = code
        self.korean = korean
        self.max_chars = max_chars


class ReportFormat(Enum):
    """Output formats for reports."""
    KOREAN = "korean"
    ENGLISH = "english"
    BILINGUAL = "bilingual"


@dataclass
class ReportSection:
    """Individual report section."""
    title: str
    content: str
    priority: int  # 1=highest, 5=lowest
    section_type: str  # risk, trend, loan, recommendation, etc.


@dataclass
class BusinessReport:
    """Complete business analysis report."""
    business_id: str
    report_type: ReportType
    report_format: ReportFormat
    generation_timestamp: str
    
    # Report content
    executive_summary: str
    detailed_analysis: str
    sections: List[ReportSection]
    
    # Key metrics
    key_insights: List[str]
    critical_alerts: List[str]
    recommendations: List[str]
    
    # Supporting data
    opportunity_data: Optional[OpportunityAssessment] = None
    change_data: Optional[ChangePointSummary] = None
    loan_data: Optional[LoanRecommendation] = None
    nh_recommendations: Optional[NHRecommendationResult] = None
    
    # Metadata
    confidence_level: float = 0.0
    data_quality_score: float = 0.0


class SeoulReportGenerator:
    """
    Proactive Financial Services Report Generator for Seoul Market Risk ML System.
    
    Generates comprehensive business reports with opportunity-focused approach:
    - 간단 요약 (200자): Key opportunity insights with financial recommendations
    - 상세 분석 (1000자): Full opportunity analysis with proactive financial services
    
    Integrates opportunity scoring, NH Bank API, and proactive financial recommendations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.report_config = self.config.get('llm_integration', {})
        
        # Report templates
        self.templates = self._initialize_templates()
        
        # Opportunity level mappings for emojis and descriptions
        self.opportunity_emoji_map = {
            1: "🚨",  # Very High Risk
            2: "🔴",  # High Risk
            3: "🟡",  # Moderate
            4: "🟢",  # Good
            5: "💚",  # Very Good
        }
        
        self.change_emoji_map = {
            ChangeType.SUDDEN_INCREASE: "📈",
            ChangeType.SUDDEN_DECREASE: "📉",
            ChangeType.VOLATILITY_INCREASE: "⚡",
            ChangeType.TREND_CHANGE: "🔄",
            ChangeType.LEVEL_SHIFT: "📊"
        }
        
        self.alert_emoji_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🆘"
        }
        
        # Industry benchmarks for opportunity comparison
        self.industry_benchmarks = {
            "음식점": {"기회_기준": 60, "평균_기회도": 45},
            "소매업": {"기회_기준": 65, "평균_기회도": 50}, 
            "서비스업": {"기회_기준": 70, "평균_기회도": 55},
            "제조업": {"기회_기준": 65, "평균_기회도": 40},
            "기타": {"기회_기준": 60, "평균_기회도": 45}
        }
        
        # NH Bank API connector
        try:
            self.nh_connector = NHBankAPIConnector()
        except Exception as e:
            logger.warning(f"NH Bank API connector initialization failed: {e}")
            self.nh_connector = None
            
        logger.info("Report Generator initialized with proactive financial services methodology")
    
    def generate_comprehensive_report(self, 
                                    business_id: str,
                                    opportunity_assessment: Optional[OpportunityAssessment] = None,
                                    change_summary: Optional[ChangePointSummary] = None,
                                    loan_recommendation: Optional[LoanRecommendation] = None,
                                    report_type: ReportType = ReportType.DETAILED,
                                    report_format: ReportFormat = ReportFormat.KOREAN) -> BusinessReport:
        """
        Generate comprehensive business report with proactive financial services.
        
        Args:
            business_id: Business identifier
            opportunity_assessment: Opportunity analysis results
            change_summary: Change point detection results
            loan_recommendation: Loan calculation results
            report_type: Type of report to generate
            report_format: Output language format
            
        Returns:
            Complete business report with financial recommendations
        """
        start_time = datetime.now()
        logger.info(f"Generating {report_type.korean} report for {business_id}")
        
        try:
            # Step 0: Get NH Bank recommendations if opportunity assessment is available
            nh_recommendations = None
            if opportunity_assessment and self.nh_connector:
                try:
                    nh_recommendations = self.nh_connector.get_personalized_recommendations(opportunity_assessment)
                except Exception as e:
                    logger.warning(f"NH Bank recommendations failed: {e}")
            
            # Step 1: Generate executive summary
            executive_summary = self._generate_executive_summary(
                business_id, opportunity_assessment, change_summary, loan_recommendation, nh_recommendations
            )
            
            # Step 2: Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(
                business_id, opportunity_assessment, change_summary, loan_recommendation, report_type, nh_recommendations
            )
            
            # Step 3: Generate individual sections
            sections = self._generate_report_sections(
                opportunity_assessment, change_summary, loan_recommendation, nh_recommendations
            )
            
            # Step 4: Extract key insights and recommendations
            key_insights = self._extract_key_insights(opportunity_assessment, change_summary, loan_recommendation, nh_recommendations)
            critical_alerts = self._extract_critical_alerts(opportunity_assessment, change_summary)
            recommendations = self._generate_recommendations(opportunity_assessment, change_summary, loan_recommendation, nh_recommendations)
            
            # Step 5: Calculate report quality metrics
            confidence_level = self._calculate_confidence_level(opportunity_assessment, change_summary, loan_recommendation)
            data_quality_score = self._calculate_data_quality(opportunity_assessment, change_summary)
            
            # Create comprehensive report
            report = BusinessReport(
                business_id=business_id,
                report_type=report_type,
                report_format=report_format,
                generation_timestamp=datetime.now().isoformat(),
                executive_summary=executive_summary,
                detailed_analysis=detailed_analysis,
                sections=sections,
                key_insights=key_insights,
                critical_alerts=critical_alerts,
                recommendations=recommendations,
                opportunity_data=opportunity_assessment,
                change_data=change_summary,
                loan_data=loan_recommendation,
                nh_recommendations=nh_recommendations,
                confidence_level=confidence_level,
                data_quality_score=data_quality_score
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Report generation completed for {business_id} in {processing_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed for {business_id}: {e}")
            return self._create_error_report(business_id, str(e), report_type, report_format)
    
    def _generate_executive_summary(self, business_id: str,
                                  opportunity_assessment: Optional[OpportunityAssessment],
                                  change_summary: Optional[ChangePointSummary],
                                  loan_recommendation: Optional[LoanRecommendation],
                                  nh_recommendations: Optional[NHRecommendationResult]) -> str:
        """Generate 200-character executive summary with opportunity and financial service indicators."""
        summary_parts = []
        
        # Opportunity assessment summary
        if opportunity_assessment:
            opportunity_emoji = self.opportunity_emoji_map.get(opportunity_assessment.opportunity_level.level, "⚪")
            summary_parts.append(
                f"{opportunity_emoji} 기회도: {opportunity_assessment.opportunity_level.korean} ({opportunity_assessment.opportunity_score:.0f}점)"
            )
            
            # Add financial recommendation
            if opportunity_assessment.recommended_action == "emergency_loan":
                summary_parts.append("🆘 긴급대출 필요")
            elif opportunity_assessment.recommended_action == "stabilization_loan":
                summary_parts.append("💳 안정화대출 권장")
            elif opportunity_assessment.recommended_action == "growth_investment":
                summary_parts.append("📈 성장투자 기회")
            elif opportunity_assessment.recommended_action == "high_yield_investment":
                summary_parts.append("💎 고수익투자 추천")
        
        # Change detection summary
        if change_summary and change_summary.total_changes > 0:
            if change_summary.critical_changes > 0:
                summary_parts.append(f"🚨 {change_summary.critical_changes}개 위험 신호")
            else:
                summary_parts.append(f"📊 {change_summary.total_changes}개 변화점")
        
        # NH Bank specific recommendations
        if nh_recommendations and nh_recommendations.api_call_success:
            if nh_recommendations.recommended_loans:
                summary_parts.append(f"🏦 NH대출 {len(nh_recommendations.recommended_loans)}건")
            if nh_recommendations.recommended_investments:
                summary_parts.append(f"💼 NH투자 {len(nh_recommendations.recommended_investments)}건")
        
        # Loan recommendation summary
        if opportunity_assessment and opportunity_assessment.loan_necessity > 0:
            loan_amount_m = opportunity_assessment.loan_necessity / 1000000  # 백만원 단위
            summary_parts.append(f"💰 필요자금: {loan_amount_m:.0f}백만원")
        
        # Stability indicator
        if change_summary:
            if change_summary.stability_score >= 70:
                summary_parts.append("✅ 안정적")
            elif change_summary.stability_score >= 50:
                summary_parts.append("⚠️ 보통")
            else:
                summary_parts.append("🔄 불안정")
        
        # Combine and limit to 200 characters
        summary = " | ".join(summary_parts)
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        return f"🎯 {summary}"
    
    def _generate_detailed_analysis(self, business_id: str,
                                   risk_assessment: Optional[RiskAssessment],
                                   change_summary: Optional[ChangePointSummary], 
                                   loan_recommendation: Optional[LoanRecommendation],
                                   report_type: ReportType) -> str:
        """Generate detailed analysis (1000+ characters) with comprehensive insights."""
        analysis_sections = []
        
        # Risk Analysis Section
        if risk_assessment:
            risk_section = self._generate_risk_analysis_section(risk_assessment)
            analysis_sections.append(risk_section)
        
        # Trend Analysis Section
        if change_summary:
            trend_section = self._generate_trend_analysis_section(change_summary)
            analysis_sections.append(trend_section)
        
        # Financial Recommendation Section
        if loan_recommendation:
            financial_section = self._generate_financial_section(loan_recommendation)
            analysis_sections.append(financial_section)
        
        # Industry Benchmarking Section
        if risk_assessment and loan_recommendation:
            benchmark_section = self._generate_benchmark_section(risk_assessment, loan_recommendation)
            analysis_sections.append(benchmark_section)
        
        # Combine sections
        detailed_analysis = "\n\n".join(analysis_sections)
        
        # Ensure length constraints
        max_chars = report_type.max_chars
        if len(detailed_analysis) > max_chars:
            detailed_analysis = detailed_analysis[:max_chars-3] + "..."
        
        return detailed_analysis
    
    def _generate_risk_analysis_section(self, risk_assessment: RiskAssessment) -> str:
        """Generate risk analysis section."""
        risk_emoji = self.risk_emoji_map.get(risk_assessment.risk_level.level, "⚪")
        
        section = f"{risk_emoji} **위험도 분석**\n"
        section += f"현재 위험점수: {risk_assessment.risk_score:.1f}점 ({risk_assessment.risk_level.korean})\n"
        
        # Top risk factors
        high_risk_components = [comp for comp in risk_assessment.components if comp.contribution >= 15]
        if high_risk_components:
            section += f"주요 위험요인: "
            factors = [f"{comp.korean_name}({comp.score:.0f}점)" for comp in high_risk_components[:2]]
            section += ", ".join(factors) + "\n"
        
        # Confidence indicator
        confidence_desc = "높음" if risk_assessment.confidence_score >= 0.8 else "보통" if risk_assessment.confidence_score >= 0.6 else "낮음"
        section += f"분석 신뢰도: {confidence_desc} ({risk_assessment.confidence_score:.2f})"
        
        return section
    
    def _generate_trend_analysis_section(self, change_summary: ChangePointSummary) -> str:
        """Generate trend analysis section."""
        section = "📈 **트렌드 분석**\n"
        
        # Overall stability
        if change_summary.stability_score >= 70:
            section += f"✅ 매출 패턴 안정적 (안정도: {change_summary.stability_score:.0f}점)\n"
        elif change_summary.stability_score >= 50:
            section += f"⚠️ 매출 패턴 보통 (안정도: {change_summary.stability_score:.0f}점)\n"
        else:
            section += f"🔄 매출 패턴 불안정 (안정도: {change_summary.stability_score:.0f}점)\n"
        
        # Critical changes
        if change_summary.critical_changes > 0:
            section += f"🚨 위험 신호 {change_summary.critical_changes}건 감지"
            
            # Most recent critical change
            critical_changes = [cp for cp in change_summary.change_points 
                             if cp.alert_level.level >= AlertLevel.WARNING.level]
            if critical_changes:
                latest_change = critical_changes[-1]
                change_emoji = self.change_emoji_map.get(latest_change.change_type, "📊")
                section += f"\n최근: {change_emoji} {latest_change.description}"
        else:
            section += "변동성 관리 양호"
        
        # Volatility trend
        if change_summary.volatility_trend == "increasing":
            section += "\n📊 변동성 증가 추세"
        elif change_summary.volatility_trend == "decreasing":
            section += "\n📉 변동성 감소 추세"
        
        return section
    
    def _generate_financial_section(self, loan_recommendation: LoanRecommendation) -> str:
        """Generate financial recommendation section."""
        section = "💰 **자금 지원 방안**\n"
        
        if loan_recommendation.required_loan_amount <= 0:
            section += "✅ 현재 자금 상황 안정적, 추가 대출 불필요"
            return section
        
        # Loan recommendation
        loan_amount_m = loan_recommendation.required_loan_amount / 1000000
        section += f"권장 대출 규모: {loan_amount_m:.0f}백만원\n"
        section += f"목표: 위험도 {loan_recommendation.current_risk_score:.0f}점 → {loan_recommendation.target_risk_score:.0f}점 개선\n"
        
        # Best product recommendation
        if loan_recommendation.recommended_products:
            best_product = loan_recommendation.recommended_products[0]
            section += f"추천 상품: {best_product.product_type.korean}"
            section += f" (금리 {best_product.interest_rate*100:.1f}%, {best_product.loan_term_months}개월)"
        
        # Repayment capacity
        repayment_ratio = loan_recommendation.repayment_analysis.repayment_capacity_ratio
        if repayment_ratio <= 0.2:
            section += "\n💚 상환여력 충분"
        elif repayment_ratio <= 0.3:
            section += "\n🟡 상환여력 보통"
        else:
            section += "\n🔴 상환부담 주의"
        
        return section
    
    def _generate_benchmark_section(self, risk_assessment: RiskAssessment,
                                   loan_recommendation: LoanRecommendation) -> str:
        """Generate industry benchmarking section."""
        section = "📊 **업종 비교**\n"
        
        business_type = loan_recommendation.business_type
        benchmark = self.industry_benchmarks.get(business_type, self.industry_benchmarks["기타"])
        
        current_risk = risk_assessment.risk_score
        industry_avg = benchmark["평균_위험도"]
        safety_threshold = benchmark["안전_기준"]
        
        if current_risk <= safety_threshold:
            section += f"✅ 업종 안전 기준({safety_threshold}점) 대비 양호"
        elif current_risk <= industry_avg:
            section += f"🟡 업종 평균({industry_avg}점) 대비 보통 수준"
        else:
            section += f"🔴 업종 평균({industry_avg}점) 대비 위험 수준"
        
        # Industry position
        if current_risk < industry_avg * 0.8:
            section += "\n상위 20% 안정성"
        elif current_risk < industry_avg * 1.2:
            section += "\n평균적 위험 수준"
        else:
            section += "\n하위 20% 주의 필요"
        
        return section
    
    def _generate_report_sections(self, risk_assessment: Optional[RiskAssessment],
                                change_summary: Optional[ChangePointSummary],
                                loan_recommendation: Optional[LoanRecommendation]) -> List[ReportSection]:
        """Generate individual report sections by category."""
        sections = []
        
        # Risk Components Section
        if risk_assessment:
            risk_content = "위험도 구성요소 분석:\n"
            for comp in risk_assessment.components:
                status_emoji = "🔴" if comp.score >= 70 else "🟡" if comp.score >= 50 else "🟢"
                risk_content += f"{status_emoji} {comp.korean_name}: {comp.score:.0f}점 ({comp.status})\n"
            
            sections.append(ReportSection(
                title="위험도 구성요소",
                content=risk_content,
                priority=1,
                section_type="risk"
            ))
        
        # Change Detection Section
        if change_summary and change_summary.change_points:
            change_content = "감지된 변화점:\n"
            for i, change in enumerate(change_summary.change_points[:3], 1):  # Top 3 changes
                alert_emoji = self.alert_emoji_map.get(change.alert_level, "ℹ️")
                change_content += f"{i}. {alert_emoji} {change.timestamp}: {change.description}\n"
            
            sections.append(ReportSection(
                title="매출 패턴 변화",
                content=change_content,
                priority=2,
                section_type="trend"
            ))
        
        # Loan Products Section
        if loan_recommendation and loan_recommendation.recommended_products:
            loan_content = "추천 금융상품:\n"
            for i, product in enumerate(loan_recommendation.recommended_products[:2], 1):
                loan_content += f"{i}. {product.product_type.korean}\n"
                loan_content += f"   금액: {product.loan_amount/1000000:.0f}백만원, "
                loan_content += f"금리: {product.interest_rate*100:.1f}%, "
                loan_content += f"기간: {product.loan_term_months}개월\n"
            
            sections.append(ReportSection(
                title="추천 금융상품",
                content=loan_content,
                priority=3,
                section_type="loan"
            ))
        
        return sections
    
    def _extract_key_insights(self, risk_assessment: Optional[RiskAssessment],
                            change_summary: Optional[ChangePointSummary],
                            loan_recommendation: Optional[LoanRecommendation]) -> List[str]:
        """Extract key business insights."""
        insights = []
        
        # Risk insights
        if risk_assessment:
            if risk_assessment.risk_score <= 35:
                insights.append("경영 안정성이 우수한 수준입니다")
            elif risk_assessment.risk_score >= 75:
                insights.append("즉시 위험 관리 조치가 필요합니다")
            
            # Component insights
            high_risk_components = [comp for comp in risk_assessment.components if comp.score >= 70]
            if high_risk_components:
                comp_names = [comp.korean_name for comp in high_risk_components[:2]]
                insights.append(f"{', '.join(comp_names)} 개선이 시급합니다")
        
        # Change detection insights
        if change_summary:
            if change_summary.stability_score >= 80:
                insights.append("매출 패턴이 매우 안정적입니다")
            elif change_summary.critical_changes >= 2:
                insights.append("최근 매출 변동이 크게 증가했습니다")
        
        # Financial insights
        if loan_recommendation and loan_recommendation.required_loan_amount > 0:
            if loan_recommendation.repayment_analysis.repayment_capacity_ratio <= 0.2:
                insights.append("추천 대출의 상환 여력이 충분합니다")
            
            if loan_recommendation.repayment_analysis.break_even_months <= 24:
                insights.append("투자 회수 기간이 양호합니다")
        
        return insights[:5]  # Top 5 insights
    
    def _extract_critical_alerts(self, risk_assessment: Optional[RiskAssessment],
                                change_summary: Optional[ChangePointSummary]) -> List[str]:
        """Extract critical alerts requiring immediate attention."""
        alerts = []
        
        # Risk-based alerts
        if risk_assessment:
            if risk_assessment.risk_score >= 85:
                alerts.append("🚨 매우 높은 위험도 - 즉시 대응 필요")
            elif risk_assessment.risk_score >= 75:
                alerts.append("⚠️ 높은 위험도 - 조기 개선 권장")
        
        # Change-based alerts
        if change_summary:
            emergency_changes = [cp for cp in change_summary.change_points 
                               if cp.alert_level == AlertLevel.EMERGENCY]
            if emergency_changes:
                alerts.append("🆘 긴급 매출 변화 감지 - 즉시 점검 필요")
            
            critical_changes = [cp for cp in change_summary.change_points
                              if cp.alert_level == AlertLevel.CRITICAL]
            if len(critical_changes) >= 2:
                alerts.append("🚨 다수 위험 신호 - 종합 점검 권장")
        
        return alerts
    
    def _generate_recommendations(self, risk_assessment: Optional[RiskAssessment],
                                change_summary: Optional[ChangePointSummary],
                                loan_recommendation: Optional[LoanRecommendation]) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        # Risk-based recommendations
        if risk_assessment:
            # Use existing recommendations from risk assessment
            if hasattr(risk_assessment, 'recommendations'):
                recommendations.extend(risk_assessment.recommendations[:3])
        
        # Change-based recommendations
        if change_summary:
            if change_summary.volatility_trend == "increasing":
                recommendations.append("매출 변동성 완화를 위한 안정화 전략 수립")
            
            if change_summary.stability_score < 50:
                recommendations.append("매출 패턴 분석 및 예측력 향상 방안 검토")
        
        # Financial recommendations
        if loan_recommendation:
            if loan_recommendation.required_loan_amount > 0:
                recommendations.append("추천 금융상품을 통한 자금 안정성 확보")
            
            # Use existing recommendations from loan assessment
            if hasattr(loan_recommendation, 'success_factors'):
                recommendations.extend(loan_recommendation.success_factors[:2])
        
        # General recommendations
        recommendations.append("정기적인 위험도 모니터링 및 평가 체계 구축")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _calculate_confidence_level(self, risk_assessment: Optional[RiskAssessment],
                                   change_summary: Optional[ChangePointSummary],
                                   loan_recommendation: Optional[LoanRecommendation]) -> float:
        """Calculate overall report confidence level."""
        confidence_factors = []
        
        # Risk assessment confidence
        if risk_assessment and hasattr(risk_assessment, 'confidence_score'):
            confidence_factors.append(risk_assessment.confidence_score)
        
        # Change detection confidence
        if change_summary and change_summary.total_changes > 0:
            # Higher confidence with more data and fewer critical changes
            detection_confidence = min(1.0, 0.5 + (0.5 * (10 - change_summary.critical_changes)) / 10)
            confidence_factors.append(detection_confidence)
        
        # Data availability factor
        components_available = sum([
            1 if risk_assessment else 0,
            1 if change_summary else 0,
            1 if loan_recommendation else 0
        ])
        availability_confidence = components_available / 3.0
        confidence_factors.append(availability_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_data_quality(self, risk_assessment: Optional[RiskAssessment],
                               change_summary: Optional[ChangePointSummary]) -> float:
        """Calculate data quality score."""
        quality_factors = []
        
        # Risk data quality
        if risk_assessment:
            # Based on number of components and their consistency
            component_scores = [comp.score for comp in risk_assessment.components]
            if len(component_scores) >= 5:  # All components available
                consistency = 1.0 - (np.std(component_scores) / 100.0)  # Lower std = higher quality
                quality_factors.append(max(0.5, consistency))
            else:
                quality_factors.append(0.6)  # Partial data
        
        # Change detection data quality
        if change_summary:
            # Based on analysis period length and stability
            if hasattr(change_summary, 'analysis_period'):
                quality_factors.append(0.8)  # Has full analysis period
            else:
                quality_factors.append(0.6)
        
        return np.mean(quality_factors) if quality_factors else 0.7
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize report templates."""
        return {
            'summary_template': """
🎯 {business_id} 위험도 분석 요약
{risk_emoji} 위험도: {risk_level} ({risk_score}점)
{trend_analysis}
{financial_recommendation}
{stability_indicator}
            """.strip(),
            
            'detailed_template': """
📊 **{business_id} 종합 위험도 분석 보고서**

{risk_section}

{trend_section}

{financial_section}

{benchmark_section}

**핵심 제언**
{recommendations}
            """.strip(),
            
            'executive_template': """
# {business_id} 경영진 위험도 분석 보고서

## 요약
{executive_summary}

## 주요 발견사항
{key_insights}

## 위험 경보
{critical_alerts}

## 권고사항
{recommendations}

## 상세 분석
{detailed_analysis}
            """.strip()
        }
    
    def _create_error_report(self, business_id: str, error_message: str,
                           report_type: ReportType, report_format: ReportFormat) -> BusinessReport:
        """Create error report when generation fails."""
        return BusinessReport(
            business_id=business_id,
            report_type=report_type,
            report_format=report_format,
            generation_timestamp=datetime.now().isoformat(),
            executive_summary=f"⚠️ 분석 오류: {business_id} 보고서 생성 실패",
            detailed_analysis=f"보고서 생성 중 오류가 발생했습니다: {error_message}\n\n데이터 품질을 점검하고 다시 시도해주세요.",
            sections=[],
            key_insights=["데이터 보완이 필요합니다"],
            critical_alerts=[f"보고서 생성 오류: {error_message}"],
            recommendations=["데이터 품질 점검 및 시스템 상태 확인"],
            confidence_level=0.0,
            data_quality_score=0.0
        )
    
    def export_report_html(self, report: BusinessReport) -> str:
        """Export report to HTML format."""
        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{report.business_id} 위험도 분석 보고서</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 20px; padding: 15px; border-left: 4px solid #007bff; }}
        .summary {{ background: #e7f3ff; padding: 15px; border-radius: 5px; font-size: 18px; }}
        .alert {{ background: #fff3cd; border-left-color: #ffc107; }}
        .critical {{ background: #f8d7da; border-left-color: #dc3545; }}
        .insight {{ background: #d1ecf1; border-left-color: #17a2b8; }}
        ul {{ padding-left: 20px; }}
        .timestamp {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏢 {report.business_id} 위험도 분석 보고서</h1>
        <p class="timestamp">생성일시: {report.generation_timestamp}</p>
        <p>보고서 유형: {report.report_type.korean} | 신뢰도: {report.confidence_level:.1%}</p>
    </div>
    
    <div class="section summary">
        <h2>📋 요약</h2>
        <p>{report.executive_summary}</p>
    </div>
    
    <div class="section">
        <h2>🔍 상세 분석</h2>
        <div style="white-space: pre-line;">{report.detailed_analysis}</div>
    </div>
    
    <div class="section insight">
        <h2>💡 핵심 인사이트</h2>
        <ul>
        {''.join(f'<li>{insight}</li>' for insight in report.key_insights)}
        </ul>
    </div>
    
    {'<div class="section critical"><h2>🚨 위험 경보</h2><ul>' + ''.join(f'<li>{alert}</li>' for alert in report.critical_alerts) + '</ul></div>' if report.critical_alerts else ''}
    
    <div class="section">
        <h2>📝 권고사항</h2>
        <ul>
        {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
        </ul>
    </div>
    
    <div class="section">
        <h2>📊 섹션별 상세</h2>
        {''.join(f'<h3>{section.title}</h3><div style="white-space: pre-line; margin-bottom: 15px;">{section.content}</div>' for section in report.sections)}
    </div>
</body>
</html>
        """
        return html_template
    
    def export_report_json(self, report: BusinessReport) -> str:
        """Export report to JSON format."""
        report_dict = {
            'business_id': report.business_id,
            'report_metadata': {
                'type': report.report_type.korean,
                'format': report.report_format.value,
                'generation_timestamp': report.generation_timestamp,
                'confidence_level': report.confidence_level,
                'data_quality_score': report.data_quality_score
            },
            'executive_summary': report.executive_summary,
            'detailed_analysis': report.detailed_analysis,
            'key_insights': report.key_insights,
            'critical_alerts': report.critical_alerts,
            'recommendations': report.recommendations,
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'priority': section.priority,
                    'type': section.section_type
                }
                for section in report.sections
            ]
        }
        
        # Add underlying data if available
        if report.risk_data:
            report_dict['risk_score'] = report.risk_data.risk_score
            report_dict['risk_level'] = report.risk_data.risk_level.korean
        
        if report.change_data:
            report_dict['stability_score'] = report.change_data.stability_score
            report_dict['total_changes'] = report.change_data.total_changes
        
        if report.loan_data:
            report_dict['loan_recommendation'] = report.loan_data.required_loan_amount
        
        return json.dumps(report_dict, ensure_ascii=False, indent=2)


def main():
    """Main function for testing report generator."""
    print("\n=== REPORT GENERATOR TEST ===")
    
    # Initialize generator
    generator = SeoulReportGenerator()
    print(f"Report Generator initialized")
    print(f"Available templates: {list(generator.templates.keys())}")
    
    # Create sample data for testing
    from ..risk_scoring.risk_calculator import RiskAssessment, RiskLevel, RiskComponent
    from ..loan_calculation.loan_calculator import LoanRecommendation, RepaymentPlan
    
    # Sample risk assessment
    sample_components = [
        RiskComponent("revenue_change", "매출변화율", 72.5, 0.3, 21.75, 72.5, "주의", "매출 변화율이 높음"),
        RiskComponent("volatility", "변동성", 58.0, 0.2, 11.6, 58.0, "보통", "매출 변동성이 보통"),
        RiskComponent("trend", "트렌드", 45.0, 0.2, 9.0, 45.0, "보통", "매출 트렌드가 보통"),
        RiskComponent("seasonality", "계절성이탈", 38.2, 0.15, 5.73, 38.2, "보통", "계절성 패턴 보통"),
        RiskComponent("industry", "업종비교", 52.0, 0.15, 7.8, 52.0, "보통", "업종 평균 수준")
    ]
    
    sample_risk = RiskAssessment(
        business_id="test_business_001",
        risk_score=67.5,
        risk_level=RiskLevel.WARNING,
        confidence_score=0.85,
        assessment_timestamp=datetime.now().isoformat(),
        components=sample_components,
        component_scores={comp.name: comp.score for comp in sample_components},
        key_risk_factors=["매출변화율이 높아 주의 필요", "변동성 관리 필요"],
        recommendations=["매출 안정화 전략 수립", "변동성 완화 방안 모색"]
    )
    
    # Sample loan recommendation
    sample_loan = LoanRecommendation(
        business_id="test_business_001",
        current_risk_score=67.5,
        target_risk_score=15.0,
        required_loan_amount=50000000,  # 5천만원
        recommended_products=[],
        repayment_analysis=RepaymentPlan(
            loan_amount=50000000,
            monthly_payment=1500000,
            loan_term_months=36,
            total_repayment=54000000,
            total_interest=4000000,
            monthly_cash_flow_impact=-1500000,
            repayment_capacity_ratio=0.15,
            risk_after_loan=25.0,
            break_even_months=18
        ),
        recommendation_timestamp=datetime.now().isoformat(),
        monthly_revenue=10000000,
        business_type="음식점",
        risk_factors=["매출 변화율 높음"],
        alternative_amounts={},
        sensitivity_analysis={}
    )
    
    # Generate test report
    report = generator.generate_comprehensive_report(
        business_id="test_business_001",
        risk_assessment=sample_risk,
        change_summary=None,
        loan_recommendation=sample_loan,
        report_type=ReportType.DETAILED
    )
    
    print(f"\n=== GENERATED REPORT ===")
    print(f"Business: {report.business_id}")
    print(f"Type: {report.report_type.korean}")
    print(f"Confidence: {report.confidence_level:.1%}")
    print(f"Data Quality: {report.data_quality_score:.1%}")
    
    print(f"\n=== EXECUTIVE SUMMARY ===")
    print(report.executive_summary)
    
    print(f"\n=== KEY INSIGHTS ===")
    for i, insight in enumerate(report.key_insights, 1):
        print(f"{i}. {insight}")
    
    print(f"\n=== RECOMMENDATIONS ===")
    for i, rec in enumerate(report.recommendations[:3], 1):
        print(f"{i}. {rec}")
    
    if report.critical_alerts:
        print(f"\n=== CRITICAL ALERTS ===")
        for alert in report.critical_alerts:
            print(f"  {alert}")
    
    print("\n=== REPORT GENERATOR READY ===")


if __name__ == "__main__":
    main()