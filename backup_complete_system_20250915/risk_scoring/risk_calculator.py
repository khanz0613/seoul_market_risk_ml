"""
Risk Score Calculator for Seoul Market Risk ML System
Real-time business risk assessment with 5-level classification and explainability.
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

# ML libraries for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available - explanations will be simplified")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Internal imports
from ..utils.config_loader import load_config, get_data_paths
from ..feature_engineering.feature_engine import SeoulFeatureEngine
from ..models.model_orchestrator import SeoulModelOrchestrator, PredictionRequest

logger = logging.getLogger(__name__)


class OpportunityLevel(Enum):
    """Financial opportunity level classifications."""
    VERY_HIGH_RISK = (1, "매우위험", "Very High Risk", "#FF0000")     # 0-20 points
    HIGH_RISK = (2, "위험군", "High Risk", "#FF6600")              # 21-40 points  
    MODERATE = (3, "적정", "Moderate", "#FFAA00")                 # 41-60 points
    GOOD = (4, "좋음", "Good", "#66BB00")                        # 61-80 points
    VERY_GOOD = (5, "매우좋음", "Very Good", "#00AA00")            # 81-100 points
    
    def __init__(self, level: int, korean: str, english: str, color: str):
        self.level = level
        self.korean = korean  
        self.english = english
        self.color = color


@dataclass
class RiskComponent:
    """Individual risk component with score and contribution."""
    name: str
    korean_name: str
    score: float
    weight: float
    contribution: float
    percentile: float
    status: str
    explanation: str


@dataclass 
class OpportunityAssessment:
    """Complete opportunity assessment result."""
    business_id: str
    opportunity_score: float
    opportunity_level: OpportunityLevel
    confidence_score: float
    assessment_timestamp: str
    
    # Financial service recommendations
    recommended_action: str  # "loan", "investment", "monitoring"
    loan_necessity: float    # How much loan needed (0 if not needed)
    investment_potential: float  # Investment suitability score (0-100)
    
    # Component analysis
    components: List[RiskComponent]
    component_scores: Dict[str, float]
    
    # Predictions and trends
    revenue_forecast: Optional[pd.DataFrame] = None
    trend_analysis: Optional[Dict[str, Any]] = None
    
    # Explanations and recommendations
    key_opportunity_factors: List[str] = field(default_factory=list)
    stability_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # SHAP explanations (if available)
    shap_values: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None


class SeoulOpportunityScoreCalculator:
    """
    Proactive Financial Opportunity Calculator for Seoul Market Risk ML System.
    
    Implements opportunity-based scoring for proactive financial services:
    - 매출변화율 (30%): Revenue growth patterns and stability
    - 변동성 (20%): Revenue volatility as risk/opportunity indicator
    - 트렌드 (20%): Growth trend strength and sustainability
    - 계절성이탈 (15%): Seasonal predictability and management
    - 업종비교 (15%): Competitive positioning and industry performance
    
    Score interpretation:
    - 0-20점 (매우위험): 선제적 긴급대출 필요
    - 21-40점 (위험군): 안정화 대출 추천
    - 41-60점 (적정): 모니터링 지속
    - 61-80점 (좋음): 성장투자 기회
    - 81-100점 (매우좋음): 고수익 투자 추천
    """
    
    def __init__(self, feature_engine: Optional[SeoulFeatureEngine] = None,
                 model_orchestrator: Optional[SeoulModelOrchestrator] = None,
                 config_path: Optional[str] = None):
        
        self.config = load_config(config_path)
        self.risk_config = self.config['risk_scoring']
        self.data_paths = get_data_paths(self.config)
        
        # Core engines
        self.feature_engine = feature_engine or SeoulFeatureEngine()
        self.model_orchestrator = model_orchestrator
        
        # Risk scoring weights (from handover report)
        self.component_weights = {
            'revenue_change': 0.30,      # 매출변화율
            'volatility': 0.20,          # 변동성  
            'trend': 0.20,               # 트렌드
            'seasonality_deviation': 0.15, # 계절성이탈
            'industry_comparison': 0.15   # 업종비교
        }
        
        # Opportunity level thresholds
        self.opportunity_thresholds = {
            OpportunityLevel.VERY_HIGH_RISK: (0, 20),
            OpportunityLevel.HIGH_RISK: (21, 40),
            OpportunityLevel.MODERATE: (41, 60),
            OpportunityLevel.GOOD: (61, 80),
            OpportunityLevel.VERY_GOOD: (81, 100)
        }
        
        # Industry benchmarks (loaded from config or calculated)
        self.industry_benchmarks = {}
        self.percentile_cache = {}
        
        # Explanation models (for SHAP)
        self.explanation_model = None
        self.feature_names = []
        
        # Initialize explanation model if SHAP available
        if SHAP_AVAILABLE:
            self._initialize_explanation_model()
        
        logger.info("Opportunity Score Calculator initialized for proactive financial services")
    
    def calculate_opportunity_score(self, business_data: pd.DataFrame, 
                                  business_id: str = "unknown",
                                  include_predictions: bool = True,
                                  include_explanations: bool = True) -> OpportunityAssessment:
        """
        Calculate comprehensive opportunity score for a business.
        
        Args:
            business_data: Historical business data
            business_id: Business identifier
            include_predictions: Whether to include revenue forecasts
            include_explanations: Whether to include SHAP explanations
            
        Returns:
            Complete opportunity assessment with financial recommendations
        """
        start_time = time.time()
        logger.info(f"Calculating opportunity score for business {business_id}")
        
        try:
            # Step 1: Calculate feature components using feature engine
            feature_scores = self._calculate_feature_components(business_data)
            
            # Step 2: Calculate individual component scores
            components = self._calculate_component_scores(feature_scores, business_data)
            
            # Step 3: Calculate weighted opportunity score
            opportunity_score = self._calculate_weighted_opportunity_score(components)
            
            # Step 4: Determine opportunity level and confidence
            opportunity_level = self._determine_opportunity_level(opportunity_score)
            confidence_score = self._calculate_confidence_score(components, business_data)
            
            # Step 5: Determine recommended financial action
            recommended_action, loan_necessity, investment_potential = self._determine_financial_recommendation(
                opportunity_score, components, business_data
            )
            
            # Step 5: Generate predictions if requested
            revenue_forecast = None
            trend_analysis = None
            if include_predictions and self.model_orchestrator:
                revenue_forecast, trend_analysis = self._generate_predictions(business_data)
            
            # Step 6: Generate explanations and recommendations
            key_factors, stability_factors, recommendations = self._generate_insights(components, business_data, opportunity_level)
            
            # Step 7: SHAP explanations if available and requested
            shap_values = None
            feature_importance = None
            if include_explanations and SHAP_AVAILABLE and self.explanation_model:
                shap_values, feature_importance = self._generate_shap_explanations(feature_scores)
            
            # Create assessment result
            assessment = OpportunityAssessment(
                business_id=business_id,
                opportunity_score=opportunity_score,
                opportunity_level=opportunity_level,
                confidence_score=confidence_score,
                assessment_timestamp=datetime.now().isoformat(),
                recommended_action=recommended_action,
                loan_necessity=loan_necessity,
                investment_potential=investment_potential,
                components=components,
                component_scores={comp.name: comp.score for comp in components},
                revenue_forecast=revenue_forecast,
                trend_analysis=trend_analysis,
                key_opportunity_factors=key_factors,
                stability_factors=stability_factors,
                recommendations=recommendations,
                shap_values=shap_values,
                feature_importance=feature_importance
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Opportunity assessment completed for {business_id}: {opportunity_score:.1f} ({opportunity_level.korean}) → {recommended_action} in {processing_time:.2f}s")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Opportunity calculation failed for {business_id}: {e}")
            # Return moderate default assessment
            return self._create_default_assessment(business_id, str(e))
    
    def _calculate_feature_components(self, business_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate raw feature components using feature engine."""
        try:
            # Use feature engine to calculate all components
            features = self.feature_engine.calculate_all_features(business_data)
            
            # Extract key risk-related features
            feature_scores = {
                'revenue_change_rate': features.get('revenue_change_score', 50.0),
                'volatility_score': features.get('volatility_score', 50.0),
                'trend_strength': features.get('trend_score', 50.0),
                'seasonality_deviation': features.get('seasonality_deviation_score', 50.0),
                'industry_percentile': features.get('industry_comparison_score', 50.0)
            }
            
            return feature_scores
            
        except Exception as e:
            logger.warning(f"Feature calculation failed, using defaults: {e}")
            return {
                'revenue_change_rate': 50.0,
                'volatility_score': 50.0, 
                'trend_strength': 50.0,
                'seasonality_deviation': 50.0,
                'industry_percentile': 50.0
            }
    
    def _calculate_component_scores(self, feature_scores: Dict[str, float], 
                                  business_data: pd.DataFrame) -> List[RiskComponent]:
        """Calculate normalized component scores with explanations."""
        components = []
        
        # Component 1: Revenue Change Rate (30%)
        revenue_change_raw = feature_scores.get('revenue_change_rate', 50.0)
        revenue_change_score = self._normalize_score(revenue_change_raw, invert=True)  # Higher change = higher risk
        revenue_explanation = self._explain_revenue_change(revenue_change_raw, business_data)
        
        components.append(RiskComponent(
            name='revenue_change',
            korean_name='매출변화율', 
            score=revenue_change_score,
            weight=self.component_weights['revenue_change'],
            contribution=revenue_change_score * self.component_weights['revenue_change'],
            percentile=revenue_change_raw,
            status=self._get_component_status(revenue_change_score),
            explanation=revenue_explanation
        ))
        
        # Component 2: Volatility (20%)
        volatility_raw = feature_scores.get('volatility_score', 50.0)
        volatility_score = self._normalize_score(volatility_raw)  # Higher volatility = higher risk
        volatility_explanation = self._explain_volatility(volatility_raw, business_data)
        
        components.append(RiskComponent(
            name='volatility',
            korean_name='변동성',
            score=volatility_score,
            weight=self.component_weights['volatility'],
            contribution=volatility_score * self.component_weights['volatility'],
            percentile=volatility_raw,
            status=self._get_component_status(volatility_score),
            explanation=volatility_explanation
        ))
        
        # Component 3: Trend Strength (20%)
        trend_raw = feature_scores.get('trend_strength', 50.0)
        trend_score = self._normalize_score(trend_raw, invert=True)  # Weaker trend = higher risk
        trend_explanation = self._explain_trend(trend_raw, business_data)
        
        components.append(RiskComponent(
            name='trend',
            korean_name='트렌드',
            score=trend_score,
            weight=self.component_weights['trend'],
            contribution=trend_score * self.component_weights['trend'],
            percentile=trend_raw,
            status=self._get_component_status(trend_score),
            explanation=trend_explanation
        ))
        
        # Component 4: Seasonality Deviation (15%)
        seasonality_raw = feature_scores.get('seasonality_deviation', 50.0)
        seasonality_score = self._normalize_score(seasonality_raw)  # Higher deviation = higher risk
        seasonality_explanation = self._explain_seasonality(seasonality_raw, business_data)
        
        components.append(RiskComponent(
            name='seasonality_deviation',
            korean_name='계절성이탈',
            score=seasonality_score,
            weight=self.component_weights['seasonality_deviation'],
            contribution=seasonality_score * self.component_weights['seasonality_deviation'],
            percentile=seasonality_raw,
            status=self._get_component_status(seasonality_score),
            explanation=seasonality_explanation
        ))
        
        # Component 5: Industry Comparison (15%)
        industry_raw = feature_scores.get('industry_percentile', 50.0)
        industry_score = self._normalize_score(industry_raw, invert=True)  # Lower percentile = higher risk
        industry_explanation = self._explain_industry_comparison(industry_raw, business_data)
        
        components.append(RiskComponent(
            name='industry_comparison',
            korean_name='업종비교',
            score=industry_score,
            weight=self.component_weights['industry_comparison'],
            contribution=industry_score * self.component_weights['industry_comparison'],
            percentile=industry_raw,
            status=self._get_component_status(industry_score),
            explanation=industry_explanation
        ))
        
        return components
    
    def _calculate_weighted_opportunity_score(self, components: List[RiskComponent]) -> float:
        """
        Calculate final weighted opportunity score.
        Converts risk-based components to opportunity-based scoring.
        """
        # Calculate traditional risk score
        traditional_risk_score = sum(comp.contribution for comp in components)
        
        # Convert risk score to opportunity score
        # High risk components become low opportunity, but with different interpretation
        opportunity_score = self._convert_risk_to_opportunity_score(traditional_risk_score, components)
        
        # Ensure score is within 0-100 range
        opportunity_score = max(0.0, min(100.0, opportunity_score))
        
        return opportunity_score
    
    def _convert_risk_to_opportunity_score(self, risk_score: float, components: List[RiskComponent]) -> float:
        """
        Convert traditional risk score to opportunity score with contextual interpretation.
        
        Args:
            risk_score: Traditional risk score (0-100, higher = more risky)
            components: Component breakdown for context
            
        Returns:
            Opportunity score (0-100, higher = better opportunity)
        """
        # Base conversion: invert the risk score for opportunity perspective
        base_opportunity = 100 - risk_score
        
        # Apply contextual adjustments based on component analysis
        adjustments = []
        
        # Revenue change analysis - high growth = opportunity boost
        revenue_comp = next((c for c in components if c.name == 'revenue_change'), None)
        if revenue_comp:
            if revenue_comp.percentile > 70:  # Strong revenue growth
                adjustments.append(10)
            elif revenue_comp.percentile < 30:  # Revenue decline
                adjustments.append(-15)
        
        # Trend analysis - positive trends boost opportunity
        trend_comp = next((c for c in components if c.name == 'trend'), None)
        if trend_comp:
            if trend_comp.percentile > 75:  # Strong upward trend
                adjustments.append(15)
            elif trend_comp.percentile < 25:  # Declining trend
                adjustments.append(-10)
        
        # Industry comparison - better than average = opportunity
        industry_comp = next((c for c in components if c.name == 'industry_comparison'), None)
        if industry_comp:
            if industry_comp.percentile > 80:  # Industry leader
                adjustments.append(12)
            elif industry_comp.percentile < 20:  # Industry laggard
                adjustments.append(-8)
        
        # Apply adjustments with damping to prevent extreme swings
        total_adjustment = sum(adjustments)
        damped_adjustment = total_adjustment * 0.3  # Damp the adjustment by 70%
        
        final_score = base_opportunity + damped_adjustment
        
        return final_score
    
    def _determine_opportunity_level(self, opportunity_score: float) -> OpportunityLevel:
        """Determine opportunity level based on score thresholds."""
        for opportunity_level, (min_score, max_score) in self.opportunity_thresholds.items():
            if min_score <= opportunity_score <= max_score:
                return opportunity_level
        
        # Default to moderate if score is outside thresholds
        return OpportunityLevel.MODERATE
    
    def _determine_financial_recommendation(self, opportunity_score: float, 
                                           components: List[RiskComponent],
                                           business_data: pd.DataFrame) -> Tuple[str, float, float]:
        """
        Determine financial service recommendation based on opportunity score.
        
        Args:
            opportunity_score: Calculated opportunity score (0-100)
            components: Component breakdown for context
            business_data: Historical business data for loan calculation
            
        Returns:
            Tuple of (recommended_action, loan_necessity, investment_potential)
        """
        # Calculate monthly revenue for loan calculations
        try:
            monthly_revenue = self._estimate_monthly_revenue(business_data)
        except:
            monthly_revenue = 10000000  # Default 10M KRW
        
        if opportunity_score <= 20:
            # 매우위험: 긴급 대출 필요
            recommended_action = "emergency_loan"
            # 위험군(40점)까지 올리기 위한 필요 대출 계산
            target_score = 40
            score_gap = target_score - opportunity_score
            loan_necessity = monthly_revenue * (score_gap / 100.0) * 3.0  # 3배수 적용
            investment_potential = 0.0
            
        elif opportunity_score <= 40:
            # 위험군: 안정화 대출 추천
            recommended_action = "stabilization_loan"
            # 적정(60점)까지 올리기 위한 필요 대출 계산
            target_score = 60
            score_gap = target_score - opportunity_score
            loan_necessity = monthly_revenue * (score_gap / 100.0) * 2.0  # 2배수 적용
            investment_potential = 10.0
            
        elif opportunity_score <= 60:
            # 적정: 모니터링 지속
            recommended_action = "monitoring"
            loan_necessity = 0.0
            investment_potential = 30.0
            
        elif opportunity_score <= 80:
            # 좋음: 투자 기회 제공
            recommended_action = "growth_investment"
            loan_necessity = 0.0
            investment_potential = 75.0
            
        else:
            # 매우좋음: 고수익 투자 추천
            recommended_action = "high_yield_investment"
            loan_necessity = 0.0
            investment_potential = 95.0
        
        # 컴포넌트 분석 기반 세부 조정
        loan_necessity, investment_potential = self._adjust_recommendations_by_components(
            loan_necessity, investment_potential, components
        )
        
        return recommended_action, loan_necessity, investment_potential
    
    def _estimate_monthly_revenue(self, business_data: pd.DataFrame) -> float:
        """Estimate monthly revenue from business data."""
        revenue_columns = ['monthly_revenue', 'revenue', 'sales', 'y']
        
        for col in revenue_columns:
            if col in business_data.columns and not business_data[col].isna().all():
                recent_revenue = business_data[col].dropna().tail(6).mean()  # Last 6 months average
                return max(1000000, recent_revenue)  # Minimum 1M KRW
        
        # If no revenue data found, return reasonable default
        return 10000000  # 10M KRW default
    
    def _adjust_recommendations_by_components(self, loan_necessity: float, investment_potential: float,
                                            components: List[RiskComponent]) -> Tuple[float, float]:
        """Adjust financial recommendations based on component analysis."""
        
        # Revenue trend adjustment
        trend_comp = next((c for c in components if c.name == 'trend'), None)
        if trend_comp:
            if trend_comp.percentile > 80:  # Strong growth trend
                loan_necessity *= 0.8  # Less loan needed
                investment_potential += 15  # More investment potential
            elif trend_comp.percentile < 20:  # Declining trend
                loan_necessity *= 1.3  # More loan needed
                investment_potential = max(0, investment_potential - 20)  # Less investment potential
        
        # Volatility adjustment
        volatility_comp = next((c for c in components if c.name == 'volatility'), None)
        if volatility_comp:
            if volatility_comp.score > 80:  # High volatility
                loan_necessity *= 1.2  # More stabilization needed
                investment_potential *= 0.7  # Riskier for investment
        
        # Industry position adjustment
        industry_comp = next((c for c in components if c.name == 'industry_comparison'), None)
        if industry_comp:
            if industry_comp.percentile > 85:  # Industry leader
                investment_potential += 10  # Premium investment opportunity
            elif industry_comp.percentile < 15:  # Industry laggard
                loan_necessity *= 1.15  # May need more support
        
        # Ensure reasonable bounds
        loan_necessity = max(0, loan_necessity)
        investment_potential = max(0, min(100, investment_potential))
        
        return loan_necessity, investment_potential
    
    def _calculate_confidence_score(self, components: List[RiskComponent], 
                                  business_data: pd.DataFrame) -> float:
        """Calculate confidence in the risk assessment."""
        confidence_factors = []
        
        # Factor 1: Data quality (amount of historical data)
        data_quality_score = min(1.0, len(business_data) / 12.0)  # 12 quarters = full confidence
        confidence_factors.append(data_quality_score)
        
        # Factor 2: Component consistency (low variance between components = higher confidence)
        component_scores = [comp.score for comp in components]
        if len(component_scores) > 1:
            consistency_score = 1.0 - (np.std(component_scores) / 100.0)  # Normalize by max std
            confidence_factors.append(max(0.0, consistency_score))
        
        # Factor 3: Extreme value detection (scores near middle = higher confidence)
        extreme_penalty = 0.0
        for comp in components:
            if comp.score < 10 or comp.score > 90:  # Very extreme scores
                extreme_penalty += 0.1
        
        extreme_confidence = max(0.0, 1.0 - extreme_penalty)
        confidence_factors.append(extreme_confidence)
        
        # Calculate weighted confidence score
        confidence_score = np.mean(confidence_factors)
        return min(1.0, max(0.0, confidence_score))
    
    def _generate_predictions(self, business_data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Generate revenue forecasts and trend analysis."""
        if not self.model_orchestrator:
            return None, None
        
        try:
            # Create future data for next 4 quarters
            last_date = business_data['ds'].max() if 'ds' in business_data.columns else datetime.now()
            future_dates = pd.date_range(start=last_date, periods=5, freq='Q')[1:]  # Skip first (current)
            
            future_data = pd.DataFrame({
                'ds': future_dates,
                'y': np.nan  # Will be predicted
            })
            
            # Request prediction
            request = PredictionRequest(future_data=future_data)
            result = self.model_orchestrator.predict(request)
            
            if not result.predictions.empty:
                # Trend analysis
                trend_analysis = {
                    'forecast_direction': 'increasing' if result.predictions['ensemble_pred'].diff().mean() > 0 else 'decreasing',
                    'forecast_volatility': result.predictions['ensemble_pred'].std(),
                    'confidence': result.confidence_score,
                    'model_used': result.model_used
                }
                
                return result.predictions, trend_analysis
            else:
                return None, None
                
        except Exception as e:
            logger.warning(f"Prediction generation failed: {e}")
            return None, None
    
    def _generate_insights(self, components: List[RiskComponent], 
                         business_data: pd.DataFrame,
                         opportunity_level: OpportunityLevel) -> Tuple[List[str], List[str], List[str]]:
        """Generate key opportunity factors, stability factors, and recommendations."""
        key_opportunity_factors = []
        stability_factors = []
        recommendations = []
        
        # Identify key opportunity factors based on opportunity level
        if opportunity_level in [OpportunityLevel.GOOD, OpportunityLevel.VERY_GOOD]:
            # Focus on growth and investment factors
            growth_components = [comp for comp in components 
                               if comp.name in ['trend', 'revenue_change', 'industry_comparison'] 
                               and comp.percentile > 60]
            for comp in growth_components:
                key_opportunity_factors.append(f"✅ {comp.korean_name}: {comp.explanation} (성장 기회)")
        else:
            # Focus on risk mitigation factors
            high_concern_components = [comp for comp in components if comp.score > 70]
            for comp in high_concern_components:
                key_opportunity_factors.append(f"⚠️ {comp.korean_name}: {comp.explanation} (개선 필요)")
        
        # Identify stability factors (consistent, reliable components)
        stable_components = [comp for comp in components 
                           if comp.name in ['volatility', 'seasonality_deviation'] 
                           and comp.score < 50]
        for comp in stable_components:
            stability_factors.append(f"🟢 {comp.korean_name}: {comp.explanation} (안정 요소)")
        
        # Generate recommendations based on opportunity level and components
        recommendations = self._generate_opportunity_recommendations(components, business_data, opportunity_level)
        
        return key_opportunity_factors, stability_factors, recommendations
    
    def _generate_opportunity_recommendations(self, components: List[RiskComponent], 
                                            business_data: pd.DataFrame,
                                            opportunity_level: OpportunityLevel) -> List[str]:
        """Generate specific business recommendations based on opportunity level."""
        recommendations = []
        
        # Level-specific recommendations
        if opportunity_level == OpportunityLevel.VERY_HIGH_RISK:
            recommendations.extend([
                "💰 긴급 운영자금 대출을 통한 현금흐름 안정화",
                "📊 주요 위험요인에 대한 즉시 개선 조치 필요",
                "🔍 전문 경영 컨설팅을 통한 사업 구조 점검",
                "⚡ 단기 비용 절감을 통한 손익분기점 달성"
            ])
        elif opportunity_level == OpportunityLevel.HIGH_RISK:
            recommendations.extend([
                "💳 안정화 대출을 통한 적정 수준으로의 개선",
                "📈 매출 다각화 전략 수립 및 실행",
                "🛡️ 변동성 완화를 위한 고정비용 관리",
                "🎯 업종 평균 수준 달성을 위한 벤치마킹"
            ])
        elif opportunity_level == OpportunityLevel.MODERATE:
            recommendations.extend([
                "📊 정기적인 재무상태 모니터링 체계 구축",
                "📈 성장 동력 발굴을 위한 시장 기회 탐색",
                "💡 운영 효율성 개선을 통한 수익성 향상",
                "🔄 계절성 대응 전략 수립"
            ])
        elif opportunity_level == OpportunityLevel.GOOD:
            recommendations.extend([
                "🚀 성장투자 기회 활용을 통한 사업 확장",
                "💼 포트폴리오 다변화 투자 검토",
                "🏆 시장 리더십 강화를 위한 혁신 투자",
                "📊 데이터 기반 의사결정 시스템 도입"
            ])
        else:  # VERY_GOOD
            recommendations.extend([
                "💎 고수익 투자상품을 통한 자산 증식",
                "🌟 프리미엄 투자 기회 우선 접근권 활용",
                "🔥 신사업 영역 진출을 위한 전략적 투자",
                "🏅 업계 선도 기업으로서의 ESG 투자 검토"
            ])
        
        # Component-specific recommendations
        for comp in sorted(components, key=lambda x: x.contribution, reverse=True)[:2]:
            if comp.score > 70:  # High concern components
                if comp.name == 'revenue_change':
                    recommendations.append("💰 매출 안정성 개선을 위한 고객 다변화 전략")
                elif comp.name == 'volatility':
                    recommendations.append("📉 매출 변동성 완화를 위한 구조적 개선")
                elif comp.name == 'trend':
                    recommendations.append("📈 성장 트렌드 회복을 위한 마케팅 강화")
            elif comp.score < 30:  # Strong performance components
                if comp.name == 'industry_comparison':
                    recommendations.append("🏆 업계 우위를 활용한 시장 점유율 확대")
                elif comp.name == 'trend':
                    recommendations.append("🚀 강력한 성장세를 기반으로 한 적극적 확장")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_recommendations(self, components: List[RiskComponent], 
                                business_data: pd.DataFrame) -> List[str]:
        """Generate specific business recommendations."""
        recommendations = []
        
        # Sort components by contribution (highest risk first)
        sorted_components = sorted(components, key=lambda x: x.contribution, reverse=True)
        
        for comp in sorted_components[:3]:  # Top 3 risk factors
            if comp.name == 'revenue_change':
                if comp.score > 70:
                    recommendations.append("매출 안정성 개선을 위한 다각화 전략 검토")
                    recommendations.append("계절적 변동에 대비한 현금 흐름 관리")
                    
            elif comp.name == 'volatility':
                if comp.score > 70:
                    recommendations.append("매출 변동성 감소를 위한 고정 고객 확보")
                    recommendations.append("예측 가능한 수익원 개발")
                    
            elif comp.name == 'trend':
                if comp.score > 70:
                    recommendations.append("성장 동력 회복을 위한 마케팅 전략 강화")
                    recommendations.append("신규 시장 개척 또는 상품 개발")
                    
            elif comp.name == 'seasonality_deviation':
                if comp.score > 70:
                    recommendations.append("계절성 패턴 분석 및 대응 전략 수립")
                    recommendations.append("비수기 대비 비용 구조 최적화")
                    
            elif comp.name == 'industry_comparison':
                if comp.score > 70:
                    recommendations.append("업종 평균 대비 경쟁력 강화 방안 모색")
                    recommendations.append("업계 모범 사례 벤치마킹")
        
        # Add general recommendations
        if len([comp for comp in components if comp.score > 70]) >= 3:
            recommendations.append("종합적인 사업 구조 개선 계획 수립 권장")
            recommendations.append("전문 경영 컨설팅 서비스 이용 검토")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _normalize_score(self, raw_score: float, invert: bool = False) -> float:
        """
        Normalize score to 0-100 risk scale.
        
        Args:
            raw_score: Raw score (usually percentile 0-100)
            invert: Whether to invert score (higher raw = lower risk)
            
        Returns:
            Normalized risk score (0=low risk, 100=high risk)
        """
        if invert:
            # Higher raw score = lower risk (e.g., better industry percentile)
            normalized = 100.0 - raw_score
        else:
            # Higher raw score = higher risk (e.g., higher volatility)
            normalized = raw_score
        
        return max(0.0, min(100.0, normalized))
    
    def _get_component_status(self, score: float) -> str:
        """Get status description for component score."""
        if score <= 30:
            return "양호"
        elif score <= 50:
            return "보통"
        elif score <= 70:
            return "주의"
        else:
            return "위험"
    
    def _explain_revenue_change(self, raw_score: float, business_data: pd.DataFrame) -> str:
        """Generate explanation for revenue change component."""
        if raw_score > 80:
            return "매출 변화율이 매우 높아 불안정성 우려"
        elif raw_score > 60:
            return "매출 변화율이 높아 주의 필요"
        elif raw_score > 40:
            return "매출 변화율이 보통 수준"
        else:
            return "매출 변화율이 안정적"
    
    def _explain_volatility(self, raw_score: float, business_data: pd.DataFrame) -> str:
        """Generate explanation for volatility component.""" 
        if raw_score > 80:
            return "매출 변동성이 매우 높음"
        elif raw_score > 60:
            return "매출 변동성이 높음"
        elif raw_score > 40:
            return "매출 변동성이 보통 수준"
        else:
            return "매출 변동성이 낮아 안정적"
    
    def _explain_trend(self, raw_score: float, business_data: pd.DataFrame) -> str:
        """Generate explanation for trend component."""
        if raw_score > 80:
            return "매출 상승 트렌드가 강함"
        elif raw_score > 60:
            return "매출 상승 트렌드가 양호"
        elif raw_score > 40:
            return "매출 트렌드가 보통"
        else:
            return "매출 하향 트렌드 또는 트렌드 약함"
    
    def _explain_seasonality(self, raw_score: float, business_data: pd.DataFrame) -> str:
        """Generate explanation for seasonality deviation component."""
        if raw_score > 80:
            return "계절성 패턴에서 크게 벗어남"
        elif raw_score > 60:
            return "계절성 패턴에서 다소 벗어남"
        elif raw_score > 40:
            return "계절성 패턴이 보통 수준"
        else:
            return "계절성 패턴이 예측 가능함"
    
    def _explain_industry_comparison(self, raw_score: float, business_data: pd.DataFrame) -> str:
        """Generate explanation for industry comparison component."""
        if raw_score > 80:
            return f"업종 내 상위 {100-raw_score:.0f}% 수준"
        elif raw_score > 60:
            return f"업종 평균 이상 (상위 {100-raw_score:.0f}%)"
        elif raw_score > 40:
            return "업종 평균 수준"
        else:
            return f"업종 평균 이하 (하위 {raw_score:.0f}%)"
    
    def _initialize_explanation_model(self) -> None:
        """Initialize SHAP explanation model."""
        if not SHAP_AVAILABLE:
            return
        
        try:
            # Simple explanation model using synthetic data
            self.feature_names = ['revenue_change', 'volatility', 'trend', 'seasonality', 'industry_position']
            
            # Create synthetic training data for explanation model
            n_samples = 1000
            X_train = np.random.rand(n_samples, len(self.feature_names)) * 100
            
            # Synthetic risk scores based on weighted combination
            y_train = (X_train[:, 0] * 0.3 +  # revenue_change
                      X_train[:, 1] * 0.2 +  # volatility
                      (100 - X_train[:, 2]) * 0.2 +  # trend (inverted)
                      X_train[:, 3] * 0.15 +  # seasonality
                      (100 - X_train[:, 4]) * 0.15)  # industry (inverted)
            
            # Train simple explanation model
            self.explanation_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.explanation_model.fit(X_train, y_train)
            
            logger.info("SHAP explanation model initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize explanation model: {e}")
            self.explanation_model = None
    
    def _generate_shap_explanations(self, feature_scores: Dict[str, float]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate SHAP explanations for risk assessment."""
        if not SHAP_AVAILABLE or not self.explanation_model:
            return None, None
        
        try:
            # Prepare feature vector
            feature_vector = np.array([[
                feature_scores.get('revenue_change_rate', 50.0),
                feature_scores.get('volatility_score', 50.0), 
                feature_scores.get('trend_strength', 50.0),
                feature_scores.get('seasonality_deviation', 50.0),
                feature_scores.get('industry_percentile', 50.0)
            ]])
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(self.explanation_model)
            shap_values = explainer.shap_values(feature_vector)
            
            # Convert to dictionary
            shap_dict = {name: float(value) for name, value in zip(self.feature_names, shap_values[0])}
            
            # Feature importance (absolute SHAP values)
            importance_dict = {name: abs(value) for name, value in shap_dict.items()}
            
            return shap_dict, importance_dict
            
        except Exception as e:
            logger.warning(f"SHAP explanation generation failed: {e}")
            return None, None
    
    def _create_default_assessment(self, business_id: str, error_message: str) -> OpportunityAssessment:
        """Create default moderate assessment when calculation fails."""
        default_components = []
        
        for name, korean_name in [
            ('revenue_change', '매출변화율'),
            ('volatility', '변동성'),
            ('trend', '트렌드'),
            ('seasonality_deviation', '계절성이탈'),
            ('industry_comparison', '업종비교')
        ]:
            component = RiskComponent(
                name=name,
                korean_name=korean_name,
                score=50.0,
                weight=self.component_weights.get(name, 0.2),
                contribution=50.0 * self.component_weights.get(name, 0.2),
                percentile=50.0,
                status="불명",
                explanation="데이터 부족으로 평가 불가"
            )
            default_components.append(component)
        
        return OpportunityAssessment(
            business_id=business_id,
            opportunity_score=50.0,
            opportunity_level=OpportunityLevel.MODERATE,
            confidence_score=0.0,
            assessment_timestamp=datetime.now().isoformat(),
            recommended_action="monitoring",
            loan_necessity=0.0,
            investment_potential=0.0,
            components=default_components,
            component_scores={comp.name: comp.score for comp in default_components},
            key_opportunity_factors=[f"평가 오류: {error_message}"],
            stability_factors=[],
            recommendations=["데이터 품질 개선 후 재평가 권장"]
        )
    
    def batch_calculate_opportunity_scores(self, business_data_list: List[Tuple[str, pd.DataFrame]],
                                         include_predictions: bool = False,
                                         include_explanations: bool = False) -> List[OpportunityAssessment]:
        """
        Calculate opportunity scores for multiple businesses efficiently.
        
        Args:
            business_data_list: List of (business_id, data) tuples
            include_predictions: Whether to include forecasts
            include_explanations: Whether to include SHAP explanations
            
        Returns:
            List of opportunity assessments
        """
        logger.info(f"Calculating opportunity scores for {len(business_data_list)} businesses")
        
        results = []
        for business_id, business_data in business_data_list:
            try:
                assessment = self.calculate_opportunity_score(
                    business_data=business_data,
                    business_id=business_id,
                    include_predictions=include_predictions,
                    include_explanations=include_explanations
                )
                results.append(assessment)
            except Exception as e:
                logger.error(f"Failed to calculate opportunity for {business_id}: {e}")
                results.append(self._create_default_assessment(business_id, str(e)))
        
        return results
    
    def export_opportunity_assessment_json(self, assessment: OpportunityAssessment) -> str:
        """Export opportunity assessment to JSON format."""
        assessment_dict = {
            'business_id': assessment.business_id,
            'opportunity_score': assessment.opportunity_score,
            'opportunity_level': {
                'level': assessment.opportunity_level.level,
                'korean': assessment.opportunity_level.korean,
                'english': assessment.opportunity_level.english,
                'color': assessment.opportunity_level.color
            },
            'confidence_score': assessment.confidence_score,
            'assessment_timestamp': assessment.assessment_timestamp,
            'financial_recommendations': {
                'recommended_action': assessment.recommended_action,
                'loan_necessity': assessment.loan_necessity,
                'investment_potential': assessment.investment_potential
            },
            'components': [
                {
                    'name': comp.name,
                    'korean_name': comp.korean_name,
                    'score': comp.score,
                    'weight': comp.weight,
                    'contribution': comp.contribution,
                    'percentile': comp.percentile,
                    'status': comp.status,
                    'explanation': comp.explanation
                }
                for comp in assessment.components
            ],
            'key_opportunity_factors': assessment.key_opportunity_factors,
            'stability_factors': assessment.stability_factors,
            'recommendations': assessment.recommendations
        }
        
        if assessment.shap_values:
            assessment_dict['shap_values'] = assessment.shap_values
            assessment_dict['feature_importance'] = assessment.feature_importance
        
        if assessment.revenue_forecast is not None:
            assessment_dict['revenue_forecast'] = assessment.revenue_forecast.to_dict('records')
            
        if assessment.trend_analysis:
            assessment_dict['trend_analysis'] = assessment.trend_analysis
        
        return json.dumps(assessment_dict, ensure_ascii=False, indent=2)


def main():
    """Main function for testing opportunity score calculator."""
    print("\n=== OPPORTUNITY SCORE CALCULATOR TEST ===")
    
    # Initialize calculator
    calculator = SeoulOpportunityScoreCalculator()
    print(f"Opportunity Score Calculator initialized")
    print(f"Component weights: {calculator.component_weights}")
    print(f"Opportunity thresholds: {[(level.korean, thresholds) for level, thresholds in calculator.opportunity_thresholds.items()]}")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'ds': pd.date_range('2023-01-01', periods=12, freq='Q'),
        'monthly_revenue': [1000000 + np.random.normal(0, 100000) for _ in range(12)],
        'monthly_transactions': [1000 + np.random.normal(0, 50) for _ in range(12)]
    })
    
    print(f"\nSample business data: {len(sample_data)} quarters")
    
    # Calculate opportunity assessment
    assessment = calculator.calculate_opportunity_score(
        business_data=sample_data,
        business_id="test_business_001",
        include_predictions=False,
        include_explanations=True
    )
    
    print(f"\n=== OPPORTUNITY ASSESSMENT RESULT ===")
    print(f"Business ID: {assessment.business_id}")
    print(f"Opportunity Score: {assessment.opportunity_score:.1f}")
    print(f"Opportunity Level: {assessment.opportunity_level.korean} ({assessment.opportunity_level.english})")
    print(f"Confidence: {assessment.confidence_score:.2f}")
    print(f"Recommended Action: {assessment.recommended_action}")
    if assessment.loan_necessity > 0:
        print(f"Loan Necessity: {assessment.loan_necessity:,.0f} KRW")
    if assessment.investment_potential > 0:
        print(f"Investment Potential: {assessment.investment_potential:.1f}/100")
    
    print(f"\nComponent Scores:")
    for comp in assessment.components:
        print(f"  {comp.korean_name}: {comp.score:.1f} (기여도: {comp.contribution:.1f})")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(assessment.recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    print("\n=== OPPORTUNITY SCORE CALCULATOR READY ===")


if __name__ == "__main__":
    import time
    main()