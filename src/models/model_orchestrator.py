"""
Model Orchestrator for Seoul Market Risk ML System
Intelligent model selection and Cold Start fallback system with prediction confidence assessment.
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
import time
warnings.filterwarnings('ignore')

# Internal imports
from ..utils.config_loader import load_config, get_data_paths
from .global_model import SeoulGlobalModel
from .regional_model import SeoulRegionalModel, SeoulRegionalModelManager  
from .local_model import SeoulLocalModel, SeoulLocalModelManager

logger = logging.getLogger(__name__)


class ModelLevel(Enum):
    """Model hierarchy levels."""
    LOCAL = "local"
    REGIONAL = "regional"
    GLOBAL = "global"


class PredictionConfidence(Enum):
    """Prediction confidence levels."""
    HIGH = "high"      # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence  
    LOW = "low"        # <70% confidence


@dataclass
class PredictionRequest:
    """Request structure for model predictions."""
    business_id: str
    region_id: Optional[int] = None
    business_category: Optional[int] = None
    historical_data: List[Dict] = field(default_factory=list)
    prediction_horizon: int = 30
    required_confidence: PredictionConfidence = PredictionConfidence.MEDIUM
    prefer_local: bool = True
    max_prediction_time_seconds: float = 30.0


@dataclass
class PredictionResult:
    """Result structure for model predictions."""
    predictions: pd.DataFrame
    model_used: str
    model_level: ModelLevel
    confidence_score: float
    prediction_timestamp: str
    processing_time_seconds: float
    fallback_chain: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ModelPerformanceMetrics:
    """Performance tracking for models."""
    model_id: str
    model_level: ModelLevel
    prediction_count: int = 0
    average_confidence: float = 0.0
    average_processing_time: float = 0.0
    failure_count: int = 0
    last_used: Optional[str] = None
    performance_trend: List[float] = field(default_factory=list)


class SeoulModelOrchestrator:
    """
    Intelligent Model Orchestrator for Seoul Market Risk ML System.
    
    Provides:
    1. Smart model selection: Local → Regional → Global
    2. Prediction confidence assessment and fallback logic
    3. Performance monitoring and automatic model switching
    4. Cold start handling for new businesses/regions
    5. Real-time model health monitoring
    """
    
    def __init__(self, 
                 global_model: Optional[SeoulGlobalModel] = None,
                 regional_manager: Optional[SeoulRegionalModelManager] = None,
                 local_manager: Optional[SeoulLocalModelManager] = None,
                 config_path: Optional[str] = None):
        
        self.config = load_config(config_path)
        self.orchestrator_config = self.config.get('models', {}).get('orchestrator', {})
        
        # Model hierarchy
        self.global_model = global_model
        self.regional_manager = regional_manager
        self.local_manager = local_manager
        
        # Performance tracking
        self.model_performance: Dict[str, ModelPerformanceMetrics] = {}
        self.prediction_history: List[Dict] = []
        
        # Configuration parameters
        self.confidence_thresholds = {
            PredictionConfidence.HIGH: self.orchestrator_config.get('high_confidence_threshold', 0.9),
            PredictionConfidence.MEDIUM: self.orchestrator_config.get('medium_confidence_threshold', 0.7),
            PredictionConfidence.LOW: self.orchestrator_config.get('low_confidence_threshold', 0.5)
        }
        
        self.max_fallback_attempts = self.orchestrator_config.get('max_fallback_attempts', 3)
        self.performance_window_size = self.orchestrator_config.get('performance_window_size', 100)
        self.auto_retrain_threshold = self.orchestrator_config.get('auto_retrain_threshold', 0.6)
        
        # Model availability cache
        self._model_availability_cache = {}
        self._last_availability_check = None
        self._cache_expiry_minutes = 10
        
        logger.info("Model Orchestrator initialized with intelligent fallback system")
    
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Main prediction method with intelligent model selection.
        
        Args:
            request: Prediction request with data and preferences
            
        Returns:
            PredictionResult with predictions and metadata
        """
        start_time = time.time()
        logger.info(f"Processing prediction request with {len(request.future_data)} data points")
        
        # Initialize result structure
        result = PredictionResult(
            predictions=pd.DataFrame(),
            model_used="none",
            model_level=ModelLevel.GLOBAL,
            confidence_score=0.0,
            prediction_timestamp=datetime.now().isoformat(),
            processing_time_seconds=0.0
        )
        
        try:
            # Determine optimal model selection strategy
            model_strategy = self._determine_model_strategy(request)
            result.fallback_chain.append(f"Strategy: {model_strategy}")
            
            # Execute prediction with fallback chain
            prediction_result = self._execute_prediction_chain(request, model_strategy)
            
            if prediction_result:
                result.predictions = prediction_result['predictions']
                result.model_used = prediction_result['model_used']
                result.model_level = self._get_model_level(prediction_result['model_used'])
                result.confidence_score = self._calculate_prediction_confidence(prediction_result)
                result.metadata = prediction_result.get('metadata', {})
                result.fallback_chain.extend(prediction_result.get('fallback_chain', []))
            else:
                result.warnings.append("All prediction methods failed")
                logger.error("All prediction methods failed")
            
        except Exception as e:
            result.warnings.append(f"Prediction error: {str(e)}")
            logger.error(f"Prediction error: {e}")
        
        # Calculate processing time and update performance metrics
        result.processing_time_seconds = time.time() - start_time
        self._update_performance_metrics(result)
        
        # Log prediction request for monitoring
        self._log_prediction_request(request, result)
        
        return result
    
    def _determine_model_strategy(self, request: PredictionRequest) -> List[str]:
        """
        Determine optimal model selection strategy based on request and performance history.
        
        Args:
            request: Prediction request
            
        Returns:
            Ordered list of model strategies to try
        """
        strategy = []
        
        # Check if specific region/business specified and data availability
        if (request.region_id is not None and request.business_category is not None 
            and self._is_model_available('local', request.region_id, request.business_category)):
            
            if request.prefer_local:
                strategy.append(f"local_{request.region_id}_{request.business_category}")
                strategy.append(f"regional_{request.region_id}")
                strategy.append("global")
            else:
                # User prefers higher-level models
                strategy.append(f"regional_{request.region_id}")
                strategy.append(f"local_{request.region_id}_{request.business_category}")
                strategy.append("global")
        
        elif request.region_id is not None and self._is_model_available('regional', request.region_id):
            strategy.append(f"regional_{request.region_id}")
            strategy.append("global")
        
        else:
            # Fall back to global model
            strategy.append("global")
        
        # Add performance-based adjustments
        strategy = self._adjust_strategy_by_performance(strategy)
        
        logger.debug(f"Model selection strategy: {strategy}")
        return strategy
    
    def _execute_prediction_chain(self, request: PredictionRequest, 
                                strategy: List[str]) -> Optional[Dict[str, Any]]:
        """
        Execute prediction using fallback chain.
        
        Args:
            request: Prediction request
            strategy: Ordered model strategies
            
        Returns:
            Prediction result or None if all failed
        """
        fallback_chain = []
        
        for attempt, model_strategy in enumerate(strategy):
            if attempt >= self.max_fallback_attempts:
                logger.warning(f"Reached maximum fallback attempts ({self.max_fallback_attempts})")
                break
            
            try:
                logger.debug(f"Attempting prediction with strategy: {model_strategy}")
                result = self._execute_single_prediction(request, model_strategy)
                
                if result and self._validate_prediction_result(result):
                    confidence = self._calculate_prediction_confidence(result)
                    
                    # Check if confidence meets requirements
                    if confidence >= self.confidence_thresholds[request.required_confidence]:
                        result['fallback_chain'] = fallback_chain + [f"Success: {model_strategy}"]
                        return result
                    else:
                        fallback_chain.append(f"Low confidence ({confidence:.2f}): {model_strategy}")
                        logger.info(f"Prediction confidence too low: {confidence:.2f} < {self.confidence_thresholds[request.required_confidence]}")
                else:
                    fallback_chain.append(f"Failed: {model_strategy}")
                    logger.warning(f"Prediction validation failed for: {model_strategy}")
                    
            except Exception as e:
                fallback_chain.append(f"Error in {model_strategy}: {str(e)}")
                logger.error(f"Prediction error with {model_strategy}: {e}")
        
        # If all strategies failed, try emergency global fallback
        try:
            logger.warning("All strategies failed, attempting emergency global fallback")
            result = self._execute_single_prediction(request, "global")
            if result:
                result['fallback_chain'] = fallback_chain + ["Emergency global fallback"]
                return result
        except Exception as e:
            logger.error(f"Emergency global fallback failed: {e}")
        
        return None
    
    def _execute_single_prediction(self, request: PredictionRequest, 
                                 model_strategy: str) -> Optional[Dict[str, Any]]:
        """
        Execute prediction using single model strategy.
        
        Args:
            request: Prediction request
            model_strategy: Model strategy (e.g., "local_1_5", "regional_2", "global")
            
        Returns:
            Prediction result or None
        """
        if model_strategy == "global":
            return self._predict_global(request)
        
        elif model_strategy.startswith("regional_"):
            region_id = int(model_strategy.split("_")[1])
            return self._predict_regional(request, region_id)
        
        elif model_strategy.startswith("local_"):
            parts = model_strategy.split("_")
            region_id = int(parts[1]) 
            business_cat = int(parts[2])
            return self._predict_local(request, region_id, business_cat)
        
        else:
            logger.error(f"Unknown model strategy: {model_strategy}")
            return None
    
    def _predict_global(self, request: PredictionRequest) -> Optional[Dict[str, Any]]:
        """Execute prediction using global model."""
        if not self.global_model:
            return None
        
        try:
            result = self.global_model.predict_ensemble(request.future_data)
            if result:
                result['model_used'] = 'global'
                result['model_level'] = 'global'
            return result
        except Exception as e:
            logger.error(f"Global model prediction failed: {e}")
            return None
    
    def _predict_regional(self, request: PredictionRequest, region_id: int) -> Optional[Dict[str, Any]]:
        """Execute prediction using regional model."""
        if not self.regional_manager:
            return None
        
        try:
            result = self.regional_manager.predict_by_region(request.future_data, region_id)
            if result and 'error' not in result:
                result['model_used'] = f'regional_{region_id}'
                result['model_level'] = 'regional'
            return result
        except Exception as e:
            logger.error(f"Regional model prediction failed for region {region_id}: {e}")
            return None
    
    def _predict_local(self, request: PredictionRequest, 
                      region_id: int, business_cat: int) -> Optional[Dict[str, Any]]:
        """Execute prediction using local model."""
        if not self.local_manager:
            return None
        
        try:
            result = self.local_manager.predict_by_local(request.future_data, region_id, business_cat)
            if result and 'error' not in result:
                result['model_used'] = f'local_{region_id}_{business_cat}'
                result['model_level'] = 'local'
            return result
        except Exception as e:
            logger.error(f"Local model prediction failed for [{region_id}, {business_cat}]: {e}")
            return None
    
    def _validate_prediction_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate prediction result quality.
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            True if result is valid and usable
        """
        if not result or 'predictions' not in result:
            return False
        
        predictions = result['predictions']
        
        # Check if predictions DataFrame is valid
        if predictions.empty or 'ensemble_pred' not in predictions.columns:
            return False
        
        # Check for NaN or infinite values
        if predictions['ensemble_pred'].isna().all() or np.isinf(predictions['ensemble_pred']).any():
            return False
        
        # Check for reasonable prediction values (basic sanity check)
        pred_values = predictions['ensemble_pred'].dropna()
        if len(pred_values) == 0:
            return False
        
        # Revenue predictions should be positive and within reasonable bounds
        if (pred_values < 0).any() or (pred_values > 1e12).any():  # 1 trillion won upper bound
            return False
        
        return True
    
    def _calculate_prediction_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate prediction confidence score.
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            Confidence score between 0 and 1
        """
        if not result or 'predictions' not in result:
            return 0.0
        
        predictions = result['predictions']
        confidence_factors = []
        
        # Factor 1: Model ensemble agreement
        if 'prophet_pred' in predictions.columns and 'ensemble_pred' in predictions.columns:
            prophet_pred = predictions['prophet_pred'].dropna()
            ensemble_pred = predictions['ensemble_pred'].dropna()
            
            if len(prophet_pred) > 0 and len(ensemble_pred) > 0:
                # Calculate coefficient of variation (lower = more confident)
                cv = np.std(ensemble_pred) / np.mean(ensemble_pred) if np.mean(ensemble_pred) > 0 else 1.0
                agreement_score = max(0.0, 1.0 - cv)
                confidence_factors.append(agreement_score)
        
        # Factor 2: Historical model performance
        model_used = result.get('model_used', '')
        if model_used in self.model_performance:
            perf = self.model_performance[model_used]
            if len(perf.performance_trend) > 0:
                recent_performance = np.mean(perf.performance_trend[-5:])  # Last 5 predictions
                confidence_factors.append(recent_performance)
        
        # Factor 3: Data quality indicators
        ensemble_pred = predictions['ensemble_pred'].dropna()
        if len(ensemble_pred) > 0:
            # Stability check (less variation = more confident)
            stability_score = 1.0 / (1.0 + np.std(ensemble_pred) / np.mean(ensemble_pred)) if np.mean(ensemble_pred) > 0 else 0.5
            confidence_factors.append(stability_score)
        
        # Factor 4: Model level preference (Local > Regional > Global)
        model_level_scores = {'local': 1.0, 'regional': 0.8, 'global': 0.6}
        model_level = result.get('model_level', 'global')
        confidence_factors.append(model_level_scores.get(model_level, 0.5))
        
        # Calculate weighted average confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    def _is_model_available(self, model_type: str, region_id: Optional[int] = None, 
                          business_cat: Optional[int] = None) -> bool:
        """
        Check if specific model is available and functional.
        
        Args:
            model_type: Type of model ("local", "regional", "global")
            region_id: Region ID for regional/local models
            business_cat: Business category for local models
            
        Returns:
            True if model is available
        """
        # Use cache to avoid repeated expensive checks
        cache_key = f"{model_type}_{region_id}_{business_cat}"
        
        if (self._last_availability_check and 
            (datetime.now() - datetime.fromisoformat(self._last_availability_check)).total_seconds() < self._cache_expiry_minutes * 60):
            return self._model_availability_cache.get(cache_key, False)
        
        # Check actual model availability
        available = False
        
        try:
            if model_type == "global":
                available = self.global_model is not None
            
            elif model_type == "regional" and region_id is not None:
                available = (self.regional_manager is not None and 
                           region_id in self.regional_manager.regional_models)
            
            elif model_type == "local" and region_id is not None and business_cat is not None:
                available = (self.local_manager is not None and
                           self.local_manager.get_model(region_id, business_cat) is not None and
                           self.local_manager.get_model(region_id, business_cat).is_trained)
            
        except Exception as e:
            logger.warning(f"Error checking model availability {cache_key}: {e}")
            available = False
        
        # Update cache
        self._model_availability_cache[cache_key] = available
        self._last_availability_check = datetime.now().isoformat()
        
        return available
    
    def _adjust_strategy_by_performance(self, strategy: List[str]) -> List[str]:
        """
        Adjust model selection strategy based on historical performance.
        
        Args:
            strategy: Initial strategy list
            
        Returns:
            Performance-adjusted strategy list
        """
        if len(strategy) <= 1:
            return strategy
        
        # Score each strategy based on performance metrics
        strategy_scores = []
        for model_strategy in strategy:
            if model_strategy in self.model_performance:
                perf = self.model_performance[model_strategy]
                # Combine confidence and speed (weighted average)
                score = 0.7 * perf.average_confidence + 0.3 * (1.0 / (1.0 + perf.average_processing_time))
                strategy_scores.append((model_strategy, score))
            else:
                # No performance history - use default score
                default_scores = {'global': 0.6, 'regional': 0.7, 'local': 0.8}
                model_type = model_strategy.split('_')[0]
                score = default_scores.get(model_type, 0.5)
                strategy_scores.append((model_strategy, score))
        
        # Sort by score (descending) and return strategy list
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        adjusted_strategy = [strategy for strategy, _ in strategy_scores]
        
        if adjusted_strategy != strategy:
            logger.debug(f"Strategy adjusted by performance: {strategy} → {adjusted_strategy}")
        
        return adjusted_strategy
    
    def _update_performance_metrics(self, result: PredictionResult) -> None:
        """Update performance metrics for the used model."""
        model_id = result.model_used
        
        if model_id not in self.model_performance:
            self.model_performance[model_id] = ModelPerformanceMetrics(
                model_id=model_id,
                model_level=result.model_level
            )
        
        perf = self.model_performance[model_id]
        perf.prediction_count += 1
        perf.last_used = result.prediction_timestamp
        
        # Update rolling averages
        alpha = 0.1  # Learning rate for exponential moving average
        perf.average_confidence = (1 - alpha) * perf.average_confidence + alpha * result.confidence_score
        perf.average_processing_time = (1 - alpha) * perf.average_processing_time + alpha * result.processing_time_seconds
        
        # Update performance trend (keep last N values)
        perf.performance_trend.append(result.confidence_score)
        if len(perf.performance_trend) > self.performance_window_size:
            perf.performance_trend.pop(0)
        
        # Check for failures
        if result.confidence_score < self.confidence_thresholds[PredictionConfidence.LOW]:
            perf.failure_count += 1
    
    def _log_prediction_request(self, request: PredictionRequest, result: PredictionResult) -> None:
        """Log prediction request for monitoring and analysis."""
        log_entry = {
            'timestamp': result.prediction_timestamp,
            'request': {
                'data_points': len(request.future_data),
                'region_id': request.region_id,
                'business_category': request.business_category,
                'required_confidence': request.required_confidence.value,
                'prefer_local': request.prefer_local
            },
            'result': {
                'model_used': result.model_used,
                'model_level': result.model_level.value,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time_seconds,
                'fallback_chain': result.fallback_chain,
                'warnings': result.warnings
            }
        }
        
        self.prediction_history.append(log_entry)
        
        # Keep only recent history (memory management)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]  # Keep last 500
    
    def _get_model_level(self, model_used: str) -> ModelLevel:
        """Determine model level from model identifier."""
        if model_used.startswith('local_'):
            return ModelLevel.LOCAL
        elif model_used.startswith('regional_'):
            return ModelLevel.REGIONAL
        else:
            return ModelLevel.GLOBAL
    
    def get_model_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive model health and performance report.
        
        Returns:
            Detailed health report
        """
        # Overall system health
        total_predictions = sum(perf.prediction_count for perf in self.model_performance.values())
        total_failures = sum(perf.failure_count for perf in self.model_performance.values())
        overall_success_rate = 1 - (total_failures / total_predictions) if total_predictions > 0 else 0
        
        # Model-level statistics
        model_stats = {}
        for model_id, perf in self.model_performance.items():
            model_stats[model_id] = {
                'prediction_count': perf.prediction_count,
                'average_confidence': perf.average_confidence,
                'average_processing_time': perf.average_processing_time,
                'failure_count': perf.failure_count,
                'success_rate': 1 - (perf.failure_count / perf.prediction_count) if perf.prediction_count > 0 else 0,
                'last_used': perf.last_used,
                'recent_performance': np.mean(perf.performance_trend[-10:]) if perf.performance_trend else 0
            }
        
        # Model availability summary
        availability_summary = {
            'global_available': self._is_model_available('global'),
            'regional_models_available': sum(1 for i in range(6) if self._is_model_available('regional', i)),
            'local_models_available': sum(1 for i in range(6) for j in range(12) 
                                       if self._is_model_available('local', i, j))
        }
        
        # Performance trends
        recent_predictions = [entry for entry in self.prediction_history 
                            if datetime.fromisoformat(entry['timestamp']) > datetime.now() - timedelta(hours=24)]
        
        trend_analysis = {
            'predictions_24h': len(recent_predictions),
            'average_confidence_24h': np.mean([p['result']['confidence_score'] for p in recent_predictions]) if recent_predictions else 0,
            'average_processing_time_24h': np.mean([p['result']['processing_time'] for p in recent_predictions]) if recent_predictions else 0,
            'most_used_model_24h': max((p['result']['model_used'] for p in recent_predictions), 
                                     key=lambda x: sum(1 for p in recent_predictions if p['result']['model_used'] == x),
                                     default='none') if recent_predictions else 'none'
        }
        
        return {
            'system_health': {
                'overall_success_rate': overall_success_rate,
                'total_predictions': total_predictions,
                'total_failures': total_failures
            },
            'model_statistics': model_stats,
            'model_availability': availability_summary,
            'performance_trends': trend_analysis,
            'configuration': {
                'confidence_thresholds': {k.value: v for k, v in self.confidence_thresholds.items()},
                'max_fallback_attempts': self.max_fallback_attempts,
                'auto_retrain_threshold': self.auto_retrain_threshold
            }
        }
    
    def suggest_model_improvements(self) -> List[Dict[str, Any]]:
        """
        Analyze performance and suggest model improvements.
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check models with poor performance
        for model_id, perf in self.model_performance.items():
            if perf.prediction_count > 10:  # Only consider models with sufficient data
                if perf.average_confidence < self.auto_retrain_threshold:
                    suggestions.append({
                        'type': 'retrain_recommended',
                        'model': model_id,
                        'reason': f'Low confidence: {perf.average_confidence:.2f}',
                        'priority': 'high' if perf.average_confidence < 0.5 else 'medium'
                    })
                
                if perf.average_processing_time > 60:  # Over 1 minute
                    suggestions.append({
                        'type': 'performance_optimization',
                        'model': model_id,
                        'reason': f'Slow processing: {perf.average_processing_time:.1f}s',
                        'priority': 'medium'
                    })
        
        # Check for unused models
        for model_type in ['local', 'regional', 'global']:
            if model_type == 'local':
                for i in range(6):
                    for j in range(12):
                        model_id = f'local_{i}_{j}'
                        if (self._is_model_available('local', i, j) and 
                            (model_id not in self.model_performance or 
                             self.model_performance[model_id].prediction_count == 0)):
                            suggestions.append({
                                'type': 'unused_model',
                                'model': model_id,
                                'reason': 'Model available but never used',
                                'priority': 'low'
                            })
        
        return suggestions
    
    def batch_predict(self, requests: List[PredictionRequest]) -> List[PredictionResult]:
        """
        Process multiple prediction requests efficiently.
        
        Args:
            requests: List of prediction requests
            
        Returns:
            List of prediction results
        """
        logger.info(f"Processing batch of {len(requests)} prediction requests")
        
        results = []
        for i, request in enumerate(requests):
            logger.debug(f"Processing batch request {i+1}/{len(requests)}")
            result = self.predict(request)
            results.append(result)
        
        return results
    
    def _mock_predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Mock prediction for testing purposes."""
        # Simulate prediction result
        mock_result = {
            'predictions': {
                'ensemble_pred': [
                    float(np.random.normal(5000000, 500000)) for _ in range(request.prediction_horizon)
                ],
                'prophet_pred': [
                    float(np.random.normal(4800000, 400000)) for _ in range(request.prediction_horizon)
                ]
            },
            'confidence': np.random.uniform(0.7, 0.95),
            'model_used': f"mock_local_{request.region_id}_{request.business_category}",
            'model_level': 'local',
            'business_id': request.business_id
        }
        
        return mock_result


def main():
    """Main function for testing model orchestrator."""
    print("\n=== MODEL ORCHESTRATOR TEST ===")
    
    # Test orchestrator initialization
    orchestrator = SeoulModelOrchestrator()
    print(f"Orchestrator initialized")
    print(f"Confidence thresholds: {orchestrator.confidence_thresholds}")
    print(f"Max fallback attempts: {orchestrator.max_fallback_attempts}")
    
    # Test prediction request structure
    sample_data = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=4, freq='Q'),
        'y': [100000, 120000, 110000, 130000]
    })
    
    request = PredictionRequest(
        future_data=sample_data,
        region_id=1,
        business_category=5,
        required_confidence=PredictionConfidence.MEDIUM,
        prefer_local=True
    )
    
    print(f"\nSample request created:")
    print(f"  Data points: {len(request.future_data)}")
    print(f"  Region: {request.region_id}")
    print(f"  Business: {request.business_category}")
    print(f"  Required confidence: {request.required_confidence}")
    
    # Test model availability check
    global_available = orchestrator._is_model_available('global')
    regional_available = orchestrator._is_model_available('regional', 1)
    local_available = orchestrator._is_model_available('local', 1, 5)
    
    print(f"\nModel availability:")
    print(f"  Global: {global_available}")
    print(f"  Regional (1): {regional_available}")
    print(f"  Local (1,5): {local_available}")
    
    print("\n=== MODEL ORCHESTRATOR READY ===")


if __name__ == "__main__":
    main()