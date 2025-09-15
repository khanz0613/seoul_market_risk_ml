"""
CUSUM + Bayesian Change Point Detection for Seoul Market Risk ML System
Real-time revenue pattern monitoring with critical threshold detection.
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
from scipy import stats
from scipy.signal import find_peaks
from collections import deque
warnings.filterwarnings('ignore')

# Statistical libraries
try:
    import ruptures as rpt  # For advanced change point detection
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    warnings.warn("ruptures not available - using simplified change point detection")

# Internal imports
from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of detected changes."""
    SUDDEN_INCREASE = ("sudden_increase", "급격한 상승", "#00AA00")
    SUDDEN_DECREASE = ("sudden_decrease", "급격한 하락", "#FF0000")
    VOLATILITY_INCREASE = ("volatility_increase", "변동성 증가", "#FF6600")
    TREND_CHANGE = ("trend_change", "트렌드 변화", "#0066FF")
    LEVEL_SHIFT = ("level_shift", "수준 변화", "#AA00AA")
    
    def __init__(self, code: str, korean: str, color: str):
        self.code = code
        self.korean = korean
        self.color = color


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = (1, "정보", "#0066FF")
    WARNING = (2, "주의", "#FFAA00") 
    CRITICAL = (3, "위험", "#FF0000")
    EMERGENCY = (4, "긴급", "#AA0000")
    
    def __init__(self, level: int, korean: str, color: str):
        self.level = level
        self.korean = korean
        self.color = color


@dataclass
class ChangePoint:
    """Detected change point with metadata."""
    timestamp: str
    index: int
    change_type: ChangeType
    alert_level: AlertLevel
    confidence: float
    magnitude: float
    
    # Detection method information
    cusum_statistic: float
    bayesian_probability: float
    threshold_triggered: str
    
    # Context information
    before_value: float
    after_value: float
    percentage_change: float
    
    # Additional metadata
    detection_method: str
    description: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ChangePointSummary:
    """Summary of change point analysis."""
    business_id: str
    analysis_period: Tuple[str, str]
    total_changes: int
    critical_changes: int
    analysis_timestamp: str
    
    # Detected changes by type
    changes_by_type: Dict[str, int]
    changes_by_alert: Dict[str, int]
    
    # All change points
    change_points: List[ChangePoint]
    
    # Overall assessment
    stability_score: float  # 0-100 (higher = more stable)
    volatility_trend: str   # increasing, decreasing, stable
    recent_patterns: List[str]


class SeoulChangePointDetector:
    """
    Advanced Change Point Detection System for Seoul Market Revenue Monitoring.
    
    Combines multiple methods:
    1. CUSUM (Cumulative Sum) for systematic drift detection
    2. Bayesian Change Point Detection for probabilistic inference
    3. Threshold-based rules for critical business scenarios
    4. Statistical tests for trend and volatility changes
    
    Critical Thresholds (from handover report):
    - 급격한 상승: 3주 연속 +20% 또는 1주 +35%
    - 급격한 하락: 2주 연속 -15% 또는 1주 -25%
    - 변동성 증가: 최근 4주 표준편차 > 과거 12주 평균×1.5
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.detection_config = self.config.get('changepoint_detection', {})
        
        # Critical threshold configuration (from handover report)
        self.thresholds = {
            # Sudden increase thresholds
            'sudden_increase_weekly': 0.35,      # 1주 +35%
            'sudden_increase_consecutive': 0.20,  # 3주 연속 +20%
            'consecutive_increase_periods': 3,
            
            # Sudden decrease thresholds  
            'sudden_decrease_weekly': -0.25,     # 1주 -25%
            'sudden_decrease_consecutive': -0.15, # 2주 연속 -15%
            'consecutive_decrease_periods': 2,
            
            # Volatility increase threshold
            'volatility_multiplier': 1.5,        # 1.5배 증가
            'volatility_recent_window': 4,       # 최근 4주
            'volatility_baseline_window': 12     # 과거 12주 평균
        }
        
        # CUSUM parameters
        self.cusum_config = {
            'drift': self.detection_config.get('cusum_drift', 1.0),
            'threshold': self.detection_config.get('cusum_threshold', 5.0),
            'reset_threshold': self.detection_config.get('cusum_reset', 3.0)
        }
        
        # Bayesian parameters
        self.bayesian_config = {
            'prior_probability': self.detection_config.get('bayesian_prior', 0.1),
            'confidence_threshold': self.detection_config.get('bayesian_confidence', 0.8)
        }
        
        # Detection history for trend analysis
        self.detection_history = deque(maxlen=1000)
        self.stability_scores = deque(maxlen=100)
        
        logger.info("Change Point Detector initialized with CUSUM + Bayesian methodology")
    
    def detect_changes(self, revenue_data: pd.DataFrame, 
                      business_id: str = "unknown",
                      include_bayesian: bool = True,
                      include_statistical: bool = True) -> ChangePointSummary:
        """
        Detect change points using multiple methods.
        
        Args:
            revenue_data: Time series data with 'ds' and revenue columns
            business_id: Business identifier
            include_bayesian: Whether to include Bayesian detection
            include_statistical: Whether to include statistical tests
            
        Returns:
            Comprehensive change point analysis
        """
        start_time = datetime.now()
        logger.info(f"Analyzing change points for {business_id} over {len(revenue_data)} periods")
        
        try:
            # Prepare data
            clean_data = self._prepare_data(revenue_data)
            if len(clean_data) < 10:
                logger.warning(f"Insufficient data for {business_id}: {len(clean_data)} periods")
                return self._create_empty_summary(business_id)
            
            # Initialize results
            all_changes = []
            
            # Method 1: Critical threshold detection (primary method)
            threshold_changes = self._detect_threshold_changes(clean_data)
            all_changes.extend(threshold_changes)
            
            # Method 2: CUSUM detection
            cusum_changes = self._detect_cusum_changes(clean_data)
            all_changes.extend(cusum_changes)
            
            # Method 3: Bayesian detection (if enabled)
            if include_bayesian:
                bayesian_changes = self._detect_bayesian_changes(clean_data)
                all_changes.extend(bayesian_changes)
            
            # Method 4: Statistical tests (if enabled)
            if include_statistical:
                statistical_changes = self._detect_statistical_changes(clean_data)
                all_changes.extend(statistical_changes)
            
            # Consolidate and rank changes
            consolidated_changes = self._consolidate_changes(all_changes, clean_data)
            
            # Calculate summary metrics
            summary = self._create_summary(business_id, clean_data, consolidated_changes, start_time)
            
            # Update detection history
            self._update_detection_history(summary)
            
            logger.info(f"Change point analysis completed for {business_id}: {len(consolidated_changes)} changes detected")
            return summary
            
        except Exception as e:
            logger.error(f"Change point detection failed for {business_id}: {e}")
            return self._create_empty_summary(business_id, str(e))
    
    def _prepare_data(self, revenue_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean revenue data for analysis."""
        # Ensure required columns exist
        if 'ds' not in revenue_data.columns:
            revenue_data = revenue_data.reset_index()
            if 'ds' not in revenue_data.columns:
                revenue_data['ds'] = pd.date_range(start='2023-01-01', periods=len(revenue_data), freq='W')
        
        # Identify revenue column
        revenue_col = None
        for col in ['monthly_revenue', 'revenue', 'sales', 'y']:
            if col in revenue_data.columns:
                revenue_col = col
                break
        
        if revenue_col is None:
            raise ValueError("No revenue column found in data")
        
        # Clean and prepare data
        clean_data = revenue_data[['ds', revenue_col]].copy()
        clean_data = clean_data.rename(columns={revenue_col: 'revenue'})
        clean_data = clean_data.sort_values('ds')
        clean_data = clean_data.dropna()
        
        # Calculate percentage changes
        clean_data['pct_change'] = clean_data['revenue'].pct_change()
        clean_data['abs_change'] = clean_data['revenue'].diff()
        
        # Calculate rolling statistics
        clean_data['rolling_mean'] = clean_data['revenue'].rolling(window=4, min_periods=2).mean()
        clean_data['rolling_std'] = clean_data['revenue'].rolling(window=4, min_periods=2).std()
        
        return clean_data.reset_index(drop=True)
    
    def _detect_threshold_changes(self, data: pd.DataFrame) -> List[ChangePoint]:
        """Detect changes based on critical business thresholds."""
        changes = []
        
        if len(data) < 3:
            return changes
        
        pct_changes = data['pct_change'].fillna(0).values
        timestamps = data['ds'].astype(str).values
        revenue_values = data['revenue'].values
        
        # Check for sudden single-period changes
        for i in range(1, len(pct_changes)):
            pct_change = pct_changes[i]
            
            # Sudden increase: 1주 +35%
            if pct_change >= self.thresholds['sudden_increase_weekly']:
                change = ChangePoint(
                    timestamp=timestamps[i],
                    index=i,
                    change_type=ChangeType.SUDDEN_INCREASE,
                    alert_level=AlertLevel.CRITICAL,
                    confidence=0.95,
                    magnitude=pct_change,
                    cusum_statistic=0.0,
                    bayesian_probability=0.0,
                    threshold_triggered="single_period_increase",
                    before_value=revenue_values[i-1],
                    after_value=revenue_values[i],
                    percentage_change=pct_change * 100,
                    detection_method="threshold",
                    description=f"매출이 1주 동안 {pct_change*100:.1f}% 급증",
                    recommendations=["급격한 매출 증가 원인 분석", "지속 가능성 검토", "운영 역량 확인"]
                )
                changes.append(change)
            
            # Sudden decrease: 1주 -25%
            elif pct_change <= self.thresholds['sudden_decrease_weekly']:
                change = ChangePoint(
                    timestamp=timestamps[i],
                    index=i,
                    change_type=ChangeType.SUDDEN_DECREASE,
                    alert_level=AlertLevel.EMERGENCY,
                    confidence=0.95,
                    magnitude=abs(pct_change),
                    cusum_statistic=0.0,
                    bayesian_probability=0.0,
                    threshold_triggered="single_period_decrease",
                    before_value=revenue_values[i-1],
                    after_value=revenue_values[i],
                    percentage_change=pct_change * 100,
                    detection_method="threshold",
                    description=f"매출이 1주 동안 {abs(pct_change)*100:.1f}% 급감",
                    recommendations=["긴급 매출 회복 계획 수립", "원인 분석 및 대응", "현금 흐름 점검"]
                )
                changes.append(change)
        
        # Check for consecutive period changes
        changes.extend(self._detect_consecutive_changes(data))
        
        # Check for volatility increases
        changes.extend(self._detect_volatility_changes(data))
        
        return changes
    
    def _detect_consecutive_changes(self, data: pd.DataFrame) -> List[ChangePoint]:
        """Detect consecutive period changes."""
        changes = []
        
        if len(data) < 4:
            return changes
        
        pct_changes = data['pct_change'].fillna(0).values
        timestamps = data['ds'].astype(str).values
        revenue_values = data['revenue'].values
        
        # Check for consecutive increases: 3주 연속 +20%
        consecutive_increases = 0
        for i in range(1, len(pct_changes)):
            if pct_changes[i] >= self.thresholds['sudden_increase_consecutive']:
                consecutive_increases += 1
                
                if consecutive_increases >= self.thresholds['consecutive_increase_periods']:
                    # Calculate average change over consecutive periods
                    avg_change = np.mean(pct_changes[i-consecutive_increases+1:i+1])
                    
                    change = ChangePoint(
                        timestamp=timestamps[i],
                        index=i,
                        change_type=ChangeType.SUDDEN_INCREASE,
                        alert_level=AlertLevel.WARNING,
                        confidence=0.85,
                        magnitude=avg_change,
                        cusum_statistic=0.0,
                        bayesian_probability=0.0,
                        threshold_triggered="consecutive_increase",
                        before_value=revenue_values[i-consecutive_increases],
                        after_value=revenue_values[i],
                        percentage_change=avg_change * 100,
                        detection_method="threshold_consecutive",
                        description=f"{consecutive_increases}주 연속 평균 {avg_change*100:.1f}% 증가",
                        recommendations=["증가 추세 지속성 분석", "시장 포화도 점검", "확장 계획 검토"]
                    )
                    changes.append(change)
                    consecutive_increases = 0  # Reset after detection
            else:
                consecutive_increases = 0
        
        # Check for consecutive decreases: 2주 연속 -15%
        consecutive_decreases = 0
        for i in range(1, len(pct_changes)):
            if pct_changes[i] <= self.thresholds['sudden_decrease_consecutive']:
                consecutive_decreases += 1
                
                if consecutive_decreases >= self.thresholds['consecutive_decrease_periods']:
                    avg_change = np.mean(pct_changes[i-consecutive_decreases+1:i+1])
                    
                    change = ChangePoint(
                        timestamp=timestamps[i],
                        index=i,
                        change_type=ChangeType.SUDDEN_DECREASE,
                        alert_level=AlertLevel.CRITICAL,
                        confidence=0.90,
                        magnitude=abs(avg_change),
                        cusum_statistic=0.0,
                        bayesian_probability=0.0,
                        threshold_triggered="consecutive_decrease",
                        before_value=revenue_values[i-consecutive_decreases],
                        after_value=revenue_values[i],
                        percentage_change=avg_change * 100,
                        detection_method="threshold_consecutive",
                        description=f"{consecutive_decreases}주 연속 평균 {abs(avg_change)*100:.1f}% 감소",
                        recommendations=["즉시 매출 회복 조치", "시장 변화 분석", "경쟁사 동향 파악"]
                    )
                    changes.append(change)
                    consecutive_decreases = 0
            else:
                consecutive_decreases = 0
        
        return changes
    
    def _detect_volatility_changes(self, data: pd.DataFrame) -> List[ChangePoint]:
        """Detect volatility increases: 최근 4주 표준편차 > 과거 12주 평균×1.5"""
        changes = []
        
        if len(data) < 16:  # Need at least 16 periods (4 recent + 12 baseline)
            return changes
        
        recent_window = self.thresholds['volatility_recent_window']
        baseline_window = self.thresholds['volatility_baseline_window']
        multiplier = self.thresholds['volatility_multiplier']
        
        timestamps = data['ds'].astype(str).values
        revenue_values = data['revenue'].values
        
        # Calculate volatility at each point where we have enough data
        for i in range(baseline_window + recent_window, len(data)):
            # Baseline volatility (past 12 periods before recent window)
            baseline_start = i - recent_window - baseline_window
            baseline_end = i - recent_window
            baseline_std = np.std(revenue_values[baseline_start:baseline_end])
            
            # Recent volatility (last 4 periods)
            recent_start = i - recent_window
            recent_end = i
            recent_std = np.std(revenue_values[recent_start:recent_end])
            
            # Check if recent volatility exceeds threshold
            if baseline_std > 0 and recent_std > baseline_std * multiplier:
                volatility_ratio = recent_std / baseline_std
                
                change = ChangePoint(
                    timestamp=timestamps[i-1],
                    index=i-1,
                    change_type=ChangeType.VOLATILITY_INCREASE,
                    alert_level=AlertLevel.WARNING,
                    confidence=0.80,
                    magnitude=volatility_ratio,
                    cusum_statistic=0.0,
                    bayesian_probability=0.0,
                    threshold_triggered="volatility_increase",
                    before_value=baseline_std,
                    after_value=recent_std,
                    percentage_change=(volatility_ratio - 1) * 100,
                    detection_method="threshold_volatility",
                    description=f"변동성이 {volatility_ratio:.1f}배 증가 (기준: {multiplier:.1f}배)",
                    recommendations=["매출 안정화 방안 모색", "변동성 원인 분석", "위험 관리 강화"]
                )
                changes.append(change)
        
        return changes
    
    def _detect_cusum_changes(self, data: pd.DataFrame) -> List[ChangePoint]:
        """Detect changes using CUSUM algorithm."""
        changes = []
        
        if len(data) < 5:
            return changes
        
        # Prepare data for CUSUM
        values = data['revenue'].values
        timestamps = data['ds'].astype(str).values
        
        # Calculate CUSUM statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return changes
        
        # Standardized values
        standardized = (values - mean_val) / std_val
        
        # CUSUM parameters
        drift = self.cusum_config['drift']
        threshold = self.cusum_config['threshold']
        
        # Initialize CUSUM statistics
        cusum_pos = 0  # Positive CUSUM (detects increases)
        cusum_neg = 0  # Negative CUSUM (detects decreases)
        
        for i in range(1, len(standardized)):
            # Update CUSUM statistics
            cusum_pos = max(0, cusum_pos + standardized[i] - drift)
            cusum_neg = max(0, cusum_neg - standardized[i] - drift)
            
            # Check for change points
            if cusum_pos > threshold:
                change = ChangePoint(
                    timestamp=timestamps[i],
                    index=i,
                    change_type=ChangeType.LEVEL_SHIFT,
                    alert_level=AlertLevel.INFO,
                    confidence=0.70,
                    magnitude=cusum_pos,
                    cusum_statistic=cusum_pos,
                    bayesian_probability=0.0,
                    threshold_triggered="cusum_positive",
                    before_value=mean_val,
                    after_value=values[i],
                    percentage_change=(values[i] - mean_val) / mean_val * 100 if mean_val > 0 else 0,
                    detection_method="cusum",
                    description=f"CUSUM 알고리즘이 상향 수준 변화 감지",
                    recommendations=["수준 변화 지속성 모니터링", "트렌드 분석"]
                )
                changes.append(change)
                cusum_pos = 0  # Reset after detection
            
            if cusum_neg > threshold:
                change = ChangePoint(
                    timestamp=timestamps[i],
                    index=i,
                    change_type=ChangeType.LEVEL_SHIFT,
                    alert_level=AlertLevel.WARNING,
                    confidence=0.70,
                    magnitude=cusum_neg,
                    cusum_statistic=cusum_neg,
                    bayesian_probability=0.0,
                    threshold_triggered="cusum_negative",
                    before_value=mean_val,
                    after_value=values[i],
                    percentage_change=(values[i] - mean_val) / mean_val * 100 if mean_val > 0 else 0,
                    detection_method="cusum",
                    description=f"CUSUM 알고리즘이 하향 수준 변화 감지",
                    recommendations=["하향 추세 원인 분석", "회복 방안 검토"]
                )
                changes.append(change)
                cusum_neg = 0
        
        return changes
    
    def _detect_bayesian_changes(self, data: pd.DataFrame) -> List[ChangePoint]:
        """Detect changes using Bayesian change point detection."""
        changes = []
        
        if len(data) < 10:
            return changes
        
        values = data['revenue'].values
        timestamps = data['ds'].astype(str).values
        
        # Simple Bayesian change point detection
        # Calculate likelihood of change at each point
        confidence_threshold = self.bayesian_config['confidence_threshold']
        
        for i in range(5, len(values) - 5):  # Need buffer on both sides
            # Split data into before and after segments
            before_segment = values[:i]
            after_segment = values[i:]
            
            # Calculate segment statistics
            mean_before = np.mean(before_segment)
            std_before = np.std(before_segment)
            mean_after = np.mean(after_segment)
            std_after = np.std(after_segment)
            
            if std_before == 0 or std_after == 0:
                continue
            
            # Simple Bayesian probability calculation
            # This is a simplified approximation - full implementation would use proper Bayesian inference
            mean_diff = abs(mean_after - mean_before)
            pooled_std = np.sqrt((std_before**2 + std_after**2) / 2)
            
            if pooled_std > 0:
                t_statistic = mean_diff / pooled_std
                # Convert to probability (simplified)
                probability = min(0.99, t_statistic / 5.0)
                
                if probability > confidence_threshold:
                    # Determine change type
                    if mean_after > mean_before:
                        change_type = ChangeType.TREND_CHANGE
                        alert_level = AlertLevel.INFO
                    else:
                        change_type = ChangeType.TREND_CHANGE
                        alert_level = AlertLevel.WARNING
                    
                    change = ChangePoint(
                        timestamp=timestamps[i],
                        index=i,
                        change_type=change_type,
                        alert_level=alert_level,
                        confidence=probability,
                        magnitude=mean_diff,
                        cusum_statistic=0.0,
                        bayesian_probability=probability,
                        threshold_triggered="bayesian",
                        before_value=mean_before,
                        after_value=mean_after,
                        percentage_change=(mean_after - mean_before) / mean_before * 100 if mean_before > 0 else 0,
                        detection_method="bayesian",
                        description=f"베이지안 분석이 트렌드 변화 감지 (확률: {probability:.2f})",
                        recommendations=["트렌드 변화 검증", "장기 영향 분석"]
                    )
                    changes.append(change)
        
        return changes
    
    def _detect_statistical_changes(self, data: pd.DataFrame) -> List[ChangePoint]:
        """Detect changes using statistical tests."""
        changes = []
        
        if not RUPTURES_AVAILABLE or len(data) < 10:
            return changes
        
        try:
            values = data['revenue'].values
            timestamps = data['ds'].astype(str).values
            
            # Use ruptures for advanced change point detection
            model = rpt.Pelt(model='rbf').fit(values)
            change_points = model.predict(pen=10)
            
            # Remove the last point (end of series)
            if change_points and change_points[-1] == len(values):
                change_points = change_points[:-1]
            
            for cp_index in change_points:
                if 0 < cp_index < len(values):
                    change = ChangePoint(
                        timestamp=timestamps[cp_index],
                        index=cp_index,
                        change_type=ChangeType.LEVEL_SHIFT,
                        alert_level=AlertLevel.INFO,
                        confidence=0.75,
                        magnitude=abs(values[cp_index] - values[cp_index-1]) if cp_index > 0 else 0,
                        cusum_statistic=0.0,
                        bayesian_probability=0.0,
                        threshold_triggered="statistical",
                        before_value=values[cp_index-1] if cp_index > 0 else values[0],
                        after_value=values[cp_index],
                        percentage_change=(values[cp_index] - values[cp_index-1]) / values[cp_index-1] * 100 if cp_index > 0 and values[cp_index-1] > 0 else 0,
                        detection_method="statistical_ruptures",
                        description="통계적 변화점 감지 (ruptures 알고리즘)",
                        recommendations=["변화점 주변 상황 분석", "구조적 변화 여부 확인"]
                    )
                    changes.append(change)
            
        except Exception as e:
            logger.warning(f"Statistical change point detection failed: {e}")
        
        return changes
    
    def _consolidate_changes(self, all_changes: List[ChangePoint], data: pd.DataFrame) -> List[ChangePoint]:
        """Consolidate overlapping changes and remove duplicates."""
        if not all_changes:
            return []
        
        # Sort by index
        all_changes.sort(key=lambda x: x.index)
        
        consolidated = []
        
        for change in all_changes:
            # Check if this change is too close to an existing one
            is_duplicate = False
            
            for existing in consolidated:
                # If changes are within 2 periods of each other, consider consolidation
                if abs(change.index - existing.index) <= 2:
                    # Keep the one with higher confidence or more critical alert level
                    if (change.alert_level.level > existing.alert_level.level or 
                        (change.alert_level.level == existing.alert_level.level and 
                         change.confidence > existing.confidence)):
                        # Replace existing with current
                        consolidated.remove(existing)
                        consolidated.append(change)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                consolidated.append(change)
        
        # Sort by timestamp for final output
        consolidated.sort(key=lambda x: x.timestamp)
        
        return consolidated
    
    def _create_summary(self, business_id: str, data: pd.DataFrame, 
                       changes: List[ChangePoint], start_time: datetime) -> ChangePointSummary:
        """Create comprehensive summary of change point analysis."""
        # Analysis period
        if not data.empty:
            period_start = str(data['ds'].min())
            period_end = str(data['ds'].max())
        else:
            period_start = period_end = "unknown"
        
        # Count changes by type and alert level
        changes_by_type = {}
        changes_by_alert = {}
        
        for change in changes:
            # By type
            type_key = change.change_type.code
            changes_by_type[type_key] = changes_by_type.get(type_key, 0) + 1
            
            # By alert level
            alert_key = change.alert_level.korean
            changes_by_alert[alert_key] = changes_by_alert.get(alert_key, 0) + 1
        
        # Count critical changes (WARNING or higher)
        critical_changes = len([c for c in changes 
                              if c.alert_level.level >= AlertLevel.WARNING.level])
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(data, changes)
        
        # Analyze volatility trend
        volatility_trend = self._analyze_volatility_trend(data)
        
        # Identify recent patterns
        recent_patterns = self._identify_recent_patterns(data, changes)
        
        return ChangePointSummary(
            business_id=business_id,
            analysis_period=(period_start, period_end),
            total_changes=len(changes),
            critical_changes=critical_changes,
            analysis_timestamp=datetime.now().isoformat(),
            changes_by_type=changes_by_type,
            changes_by_alert=changes_by_alert,
            change_points=changes,
            stability_score=stability_score,
            volatility_trend=volatility_trend,
            recent_patterns=recent_patterns
        )
    
    def _calculate_stability_score(self, data: pd.DataFrame, changes: List[ChangePoint]) -> float:
        """Calculate stability score (0-100, higher = more stable)."""
        if data.empty:
            return 50.0
        
        # Factors affecting stability
        factors = []
        
        # Factor 1: Number of changes (fewer changes = more stable)
        change_penalty = min(50, len(changes) * 5)  # 5 points per change, max 50
        factors.append(100 - change_penalty)
        
        # Factor 2: Coefficient of variation (lower CV = more stable)
        revenue_values = data['revenue'].values
        if len(revenue_values) > 1 and np.mean(revenue_values) > 0:
            cv = np.std(revenue_values) / np.mean(revenue_values)
            cv_score = max(0, 100 - cv * 100)  # Convert CV to 0-100 scale
            factors.append(cv_score)
        
        # Factor 3: Critical changes penalty
        critical_changes = len([c for c in changes if c.alert_level.level >= AlertLevel.WARNING.level])
        critical_penalty = min(30, critical_changes * 10)  # 10 points per critical change
        factors.append(100 - critical_penalty)
        
        # Calculate weighted average
        stability_score = np.mean(factors) if factors else 50.0
        return max(0, min(100, stability_score))
    
    def _analyze_volatility_trend(self, data: pd.DataFrame) -> str:
        """Analyze if volatility is increasing, decreasing, or stable."""
        if len(data) < 10:
            return "unknown"
        
        # Calculate rolling volatility
        rolling_std = data['revenue'].rolling(window=5, min_periods=3).std()
        recent_volatility = rolling_std.tail(5).mean()
        earlier_volatility = rolling_std.head(len(rolling_std)//2).mean()
        
        if pd.isna(recent_volatility) or pd.isna(earlier_volatility):
            return "unknown"
        
        ratio = recent_volatility / earlier_volatility if earlier_volatility > 0 else 1.0
        
        if ratio > 1.2:
            return "increasing"
        elif ratio < 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_recent_patterns(self, data: pd.DataFrame, changes: List[ChangePoint]) -> List[str]:
        """Identify patterns in recent changes."""
        patterns = []
        
        if data.empty:
            return patterns
        
        # Analyze recent 4 periods
        recent_data = data.tail(4)
        recent_changes = [c for c in changes if c.index >= len(data) - 4]
        
        # Pattern 1: Consistent growth
        if len(recent_data) >= 3:
            pct_changes = recent_data['pct_change'].fillna(0)
            if all(pc > 0 for pc in pct_changes[1:]):  # All positive changes
                patterns.append("지속적 성장")
            elif all(pc < 0 for pc in pct_changes[1:]):  # All negative changes
                patterns.append("지속적 감소")
        
        # Pattern 2: High frequency of changes
        if len(recent_changes) >= 2:
            patterns.append("최근 변동성 높음")
        
        # Pattern 3: Critical alerts
        critical_recent = [c for c in recent_changes if c.alert_level.level >= AlertLevel.CRITICAL.level]
        if critical_recent:
            patterns.append("최근 위험 신호")
        
        # Pattern 4: Volatility increase detected
        volatility_changes = [c for c in recent_changes if c.change_type == ChangeType.VOLATILITY_INCREASE]
        if volatility_changes:
            patterns.append("변동성 증가")
        
        return patterns[:5]  # Limit to top 5 patterns
    
    def _create_empty_summary(self, business_id: str, error_message: str = None) -> ChangePointSummary:
        """Create empty summary when analysis fails or insufficient data."""
        return ChangePointSummary(
            business_id=business_id,
            analysis_period=("unknown", "unknown"),
            total_changes=0,
            critical_changes=0,
            analysis_timestamp=datetime.now().isoformat(),
            changes_by_type={},
            changes_by_alert={"오류": 1} if error_message else {},
            change_points=[],
            stability_score=50.0,
            volatility_trend="unknown",
            recent_patterns=["데이터 부족"] if not error_message else [f"분석 오류: {error_message}"]
        )
    
    def _update_detection_history(self, summary: ChangePointSummary) -> None:
        """Update detection history for trend analysis."""
        history_entry = {
            'timestamp': summary.analysis_timestamp,
            'business_id': summary.business_id,
            'total_changes': summary.total_changes,
            'critical_changes': summary.critical_changes,
            'stability_score': summary.stability_score
        }
        
        self.detection_history.append(history_entry)
        self.stability_scores.append(summary.stability_score)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        if not self.detection_history:
            return {'message': 'No detection history available'}
        
        recent_history = list(self.detection_history)[-50:]  # Last 50 analyses
        
        # Overall statistics
        total_analyses = len(recent_history)
        total_changes = sum(entry['total_changes'] for entry in recent_history)
        total_critical = sum(entry['critical_changes'] for entry in recent_history)
        avg_stability = np.mean([entry['stability_score'] for entry in recent_history])
        
        # Trend analysis
        stability_trend = "stable"
        if len(self.stability_scores) >= 10:
            recent_avg = np.mean(list(self.stability_scores)[-5:])
            earlier_avg = np.mean(list(self.stability_scores)[-10:-5])
            
            if recent_avg > earlier_avg * 1.1:
                stability_trend = "improving"
            elif recent_avg < earlier_avg * 0.9:
                stability_trend = "declining"
        
        return {
            'monitoring_period': '최근 분석 현황',
            'total_analyses': total_analyses,
            'total_changes_detected': total_changes,
            'critical_changes': total_critical,
            'average_stability_score': avg_stability,
            'stability_trend': stability_trend,
            'alert_rate': (total_critical / total_changes * 100) if total_changes > 0 else 0,
            'configuration': {
                'thresholds': self.thresholds,
                'cusum_config': self.cusum_config,
                'bayesian_config': self.bayesian_config
            }
        }


def main():
    """Main function for testing change point detector."""
    print("\n=== CHANGE POINT DETECTOR TEST ===")
    
    # Initialize detector
    detector = SeoulChangePointDetector()
    print(f"Change Point Detector initialized")
    print(f"Critical thresholds: {detector.thresholds}")
    
    # Generate test data with artificial change points
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=20, freq='W')
    
    # Base revenue with artificial changes
    base_revenue = 1000000
    revenues = [base_revenue]
    
    for i in range(1, 20):
        if i == 5:  # Sudden increase at week 5
            change = revenues[-1] * 1.4  # 40% increase
        elif i == 12:  # Sudden decrease at week 12
            change = revenues[-1] * 0.7  # 30% decrease
        else:
            # Normal variation
            change = revenues[-1] * (1 + np.random.normal(0, 0.05))
        
        revenues.append(max(0, change))
    
    test_data = pd.DataFrame({
        'ds': dates,
        'revenue': revenues
    })
    
    print(f"\nTest data: {len(test_data)} weeks")
    print(f"Revenue range: {test_data['revenue'].min():,.0f} - {test_data['revenue'].max():,.0f}")
    
    # Run change point detection
    summary = detector.detect_changes(
        revenue_data=test_data,
        business_id="test_business",
        include_bayesian=True,
        include_statistical=False  # Skip ruptures if not available
    )
    
    print(f"\n=== DETECTION RESULTS ===")
    print(f"Business ID: {summary.business_id}")
    print(f"Analysis period: {summary.analysis_period[0]} to {summary.analysis_period[1]}")
    print(f"Total changes: {summary.total_changes}")
    print(f"Critical changes: {summary.critical_changes}")
    print(f"Stability score: {summary.stability_score:.1f}")
    print(f"Volatility trend: {summary.volatility_trend}")
    
    if summary.change_points:
        print(f"\nDetected Changes:")
        for i, change in enumerate(summary.change_points, 1):
            print(f"  {i}. {change.timestamp}: {change.change_type.korean}")
            print(f"     Alert: {change.alert_level.korean}, Confidence: {change.confidence:.2f}")
            print(f"     Description: {change.description}")
    
    print(f"\nRecent patterns: {', '.join(summary.recent_patterns)}")
    
    print("\n=== CHANGE POINT DETECTOR READY ===")


if __name__ == "__main__":
    main()