"""
Local Model for Seoul Market Risk ML System
Automated system for managing 72 local models (6 regions × 12 business categories).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import joblib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import time
import multiprocessing as mp
warnings.filterwarnings('ignore')

# Time series forecasting libraries
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import lightgbm as lgb

# ML libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor

# Internal imports
from ..utils.config_loader import load_config, get_data_paths
from .global_model import SeoulGlobalModel
from .regional_model import SeoulRegionalModel, SeoulRegionalModelManager

logger = logging.getLogger(__name__)


@dataclass
class LocalModelConfig:
    """Configuration for local model training."""
    region_id: int
    business_category: int
    min_data_threshold: int = 8  # Minimum quarters for local training
    fallback_to_regional: bool = True
    fallback_to_global: bool = True
    enable_parallel_training: bool = True
    max_training_time_minutes: int = 10  # Max time per local model


class SeoulLocalModel:
    """
    Individual Local Model for specific region-business combination.
    
    Lightweight model optimized for specific regional business patterns:
    - Focuses on dominant patterns for the region-business combination  
    - Fast training and prediction for real-time use
    - Automatic fallback to Regional/Global models when needed
    """
    
    def __init__(self, region_id: int, business_category: int, 
                 regional_model: Optional[SeoulRegionalModel] = None,
                 global_model: Optional[SeoulGlobalModel] = None, 
                 config: Optional[LocalModelConfig] = None):
        
        self.region_id = region_id
        self.business_category = business_category  
        self.config = config or LocalModelConfig(region_id, business_category)
        
        # Model hierarchy for fallback
        self.regional_model = regional_model
        self.global_model = global_model
        
        # Local models (simplified ensemble)
        self.prophet_model = None
        self.lightgbm_model = None  # Skip ARIMA for locals to improve speed
        self.ensemble_weights = {'prophet': 0.7, 'lightgbm': 0.3}
        
        # Model metadata
        self.training_data = None
        self.model_performance = {}
        self.is_trained = False
        self.training_timestamp = None
        self.data_sufficiency = False
        
        # Feature configuration - streamlined for local models
        self.feature_columns = []
        self.target_column = 'monthly_revenue'
        
        logger.debug(f"Local Model initialized: Region {region_id}, Business {business_category}")
    
    def prepare_local_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset for specific region-business combination.
        
        Args:
            df: Input DataFrame with sales data
            
        Returns:
            Prepared local time series dataset
        """
        # Filter for specific region and business category
        if 'region_cluster' in df.columns and 'business_cluster' in df.columns:
            local_data = df[
                (df['region_cluster'] == self.region_id) & 
                (df['business_cluster'] == self.business_category)
            ].copy()
        else:
            # Fallback: deterministic sampling
            logger.warning(f"No clustering columns, using sample for Region {self.region_id}, Business {self.business_category}")
            total_combinations = 72
            combination_id = self.region_id * 12 + self.business_category
            sample_size = len(df) // total_combinations
            start_idx = combination_id * sample_size
            end_idx = min((combination_id + 1) * sample_size, len(df))
            local_data = df.iloc[start_idx:end_idx].copy()
        
        if len(local_data) < self.config.min_data_threshold:
            logger.info(f"Insufficient data for Local Model [{self.region_id}, {self.business_category}]: {len(local_data)} records")
            self.data_sufficiency = False
            return pd.DataFrame()
        
        self.data_sufficiency = True
        
        # Aggregate to quarterly data for the local segment
        local_series = local_data.groupby('quarter_code').agg({
            'monthly_revenue': ['sum', 'mean', 'count'],
            'monthly_transactions': ['sum', 'mean']
        }).round(2)
        
        # Flatten column names  
        local_series.columns = [f'{col[0]}_{col[1]}' for col in local_series.columns]
        local_series = local_series.reset_index()
        
        # Create date column
        local_series['year'] = local_series['quarter_code'].astype(str).str[:4].astype(int)
        local_series['quarter'] = local_series['quarter_code'].astype(str).str[4:].astype(int)
        local_series['ds'] = pd.to_datetime(
            local_series['year'].astype(str) + '-' + 
            ((local_series['quarter'] - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
        )
        
        # Target variable
        local_series['y'] = local_series['monthly_revenue_sum']
        
        # Simplified features for local models
        local_series['revenue_per_transaction'] = (
            local_series['monthly_revenue_sum'] / local_series['monthly_transactions_sum']
        )
        
        # Time features
        local_series['year_num'] = local_series['ds'].dt.year
        local_series['quarter_num'] = local_series['ds'].dt.quarter
        local_series['is_peak_quarter'] = local_series['quarter_num'].isin([4, 1])  # Simplified seasonality
        
        # Sort by date
        local_series = local_series.sort_values('ds').reset_index(drop=True)
        
        logger.debug(f"Local dataset prepared [{self.region_id}, {self.business_category}]: {len(local_series)} quarters")
        return local_series
    
    def train_local_models(self, df: pd.DataFrame, 
                         train_test_split: float = 0.8) -> Dict[str, Any]:
        """
        Train local model ensemble with time constraints.
        
        Args:
            df: Input dataset  
            train_test_split: Ratio for train/test split
            
        Returns:
            Training results and performance metrics
        """
        start_time = time.time()
        logger.debug(f"Training Local Model [{self.region_id}, {self.business_category}]...")
        
        # Prepare local dataset
        local_data = self.prepare_local_dataset(df)
        
        if local_data.empty or not self.data_sufficiency:
            return {'error': 'insufficient_data', 'fallback_required': True}
        
        self.training_data = local_data.copy()
        
        # Split data
        split_idx = max(1, int(len(local_data) * train_test_split))
        train_data = local_data[:split_idx].copy()
        test_data = local_data[split_idx:].copy() if split_idx < len(local_data) else pd.DataFrame()
        
        # Train models with timeout protection
        results = {'training_duration': 0, 'models_trained': []}
        
        try:
            # Train Prophet (fast, essential)
            if time.time() - start_time < self.config.max_training_time_minutes * 60 * 0.6:  # 60% of time budget
                self.prophet_model = self._train_local_prophet(train_data)
                if self.prophet_model:
                    results['models_trained'].append('prophet')
            
            # Train LightGBM (if time remains)  
            if time.time() - start_time < self.config.max_training_time_minutes * 60 * 0.9:  # 90% of time budget
                self.lightgbm_model = self._train_local_lightgbm(train_data)
                if self.lightgbm_model:
                    results['models_trained'].append('lightgbm')
            
            # Evaluate if test data available
            if not test_data.empty:
                results.update(self._evaluate_local_models(test_data))
            
            # Update training status
            self.is_trained = len(results['models_trained']) > 0
            self.training_timestamp = datetime.now().isoformat()
            results['training_duration'] = time.time() - start_time
            
            if self.is_trained:
                logger.debug(f"Local Model [{self.region_id}, {self.business_category}] trained successfully in {results['training_duration']:.1f}s")
            else:
                logger.warning(f"Local Model [{self.region_id}, {self.business_category}] training failed")
            
        except Exception as e:
            logger.error(f"Local Model [{self.region_id}, {self.business_category}] training error: {e}")
            results['error'] = str(e)
            self.is_trained = False
        
        return results
    
    def _train_local_prophet(self, train_data: pd.DataFrame) -> Optional[Prophet]:
        """Train streamlined Prophet model for local patterns."""
        try:
            # Simplified Prophet configuration for speed
            prophet_model = Prophet(
                yearly_seasonality=False,       # Skip yearly for local models
                quarterly_seasonality=True,     # Keep quarterly
                daily_seasonality=False,
                weekly_seasonality=False,
                seasonality_mode='additive',    # Simpler mode
                changepoint_prior_scale=0.01,   # Conservative
                seasonality_prior_scale=5,      # Moderate
                uncertainty_samples=0,          # Skip uncertainty for speed
                mcmc_samples=0
            )
            
            # Minimal feature set for speed
            prophet_data = train_data[['ds', 'y']].copy()
            
            # Add only key features
            if 'is_peak_quarter' in train_data.columns:
                prophet_model.add_regressor('is_peak_quarter')
                prophet_data['is_peak_quarter'] = train_data['is_peak_quarter']
            
            # Fit with timeout protection
            prophet_model.fit(prophet_data)
            return prophet_model
            
        except Exception as e:
            logger.warning(f"Local Prophet training failed [{self.region_id}, {self.business_category}]: {e}")
            return None
    
    def _train_local_lightgbm(self, train_data: pd.DataFrame) -> Optional[lgb.LGBMRegressor]:
        """Train fast LightGBM model for local non-linear patterns."""
        try:
            if len(train_data) < 4:  # Need minimum data
                return None
                
            # Prepare features
            X_train = pd.DataFrame()
            
            # Basic time features
            X_train['quarter'] = train_data['ds'].dt.quarter
            X_train['year'] = train_data['ds'].dt.year
            
            # Lag features (minimal)
            y_series = train_data['y']
            if len(y_series) > 1:
                X_train['y_lag_1'] = y_series.shift(1)
            if len(y_series) > 2:
                X_train['y_rolling_mean_2'] = y_series.rolling(window=2).mean()
            
            # Business features
            if 'revenue_per_transaction' in train_data.columns:
                X_train['revenue_per_transaction'] = train_data['revenue_per_transaction']
            
            # Remove NaN and prepare target
            valid_indices = X_train.dropna().index
            X_train = X_train.loc[valid_indices]
            y_train = train_data.loc[valid_indices, 'y']
            
            if len(X_train) < 3:
                return None
            
            self.feature_columns = X_train.columns.tolist()
            
            # Fast LightGBM configuration
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse', 
                'boosting_type': 'gbdt',
                'num_leaves': min(15, 2 ** int(np.log2(len(X_train)))),  # Adaptive
                'learning_rate': 0.1,  # Faster learning
                'feature_fraction': 1.0,  # Use all features
                'bagging_fraction': 1.0,  # No bagging for speed
                'verbose': -1,
                'min_child_samples': max(2, len(X_train) // 5),
                'num_iterations': min(50, max(10, len(X_train) * 2))  # Adaptive iterations
            }
            
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                callbacks=[lgb.early_stopping(stopping_rounds=5), lgb.log_evaluation(0)]
            )
            
            return lgb_model
            
        except Exception as e:
            logger.warning(f"Local LightGBM training failed [{self.region_id}, {self.business_category}]: {e}")
            return None
    
    def predict_with_fallback(self, future_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate predictions with hierarchical fallback.
        
        Priority: Local → Regional → Global
        
        Args:
            future_data: Future data for prediction
            
        Returns:
            Prediction results with fallback information
        """
        try:
            # Try local prediction first
            if self.is_trained:
                local_predictions = self._predict_local_ensemble(future_data)
                
                if local_predictions and 'predictions' in local_predictions:
                    pred_df = local_predictions['predictions']
                    if not pred_df.empty and not pred_df['ensemble_pred'].isna().all():
                        local_predictions['used_fallback'] = False
                        local_predictions['prediction_source'] = f'local_{self.region_id}_{self.business_category}'
                        return local_predictions
            
            # Fallback to regional model
            if self.config.fallback_to_regional and self.regional_model:
                logger.debug(f"Using regional fallback for Local Model [{self.region_id}, {self.business_category}]")
                regional_predictions = self.regional_model.predict_with_fallback(future_data)
                if regional_predictions:
                    regional_predictions['used_fallback'] = True
                    regional_predictions['prediction_source'] = f'regional_{self.region_id}_fallback'
                    regional_predictions['original_local'] = (self.region_id, self.business_category)
                    return regional_predictions
            
            # Final fallback to global model
            if self.config.fallback_to_global and self.global_model:
                logger.debug(f"Using global fallback for Local Model [{self.region_id}, {self.business_category}]")
                global_predictions = self.global_model.predict_ensemble(future_data)
                if global_predictions:
                    global_predictions['used_fallback'] = True
                    global_predictions['prediction_source'] = 'global_fallback'
                    global_predictions['original_local'] = (self.region_id, self.business_category)
                    return global_predictions
            
            return {'error': 'all_fallbacks_failed'}
            
        except Exception as e:
            logger.error(f"Prediction failed for Local Model [{self.region_id}, {self.business_category}]: {e}")
            return {'error': str(e)}
    
    def _predict_local_ensemble(self, future_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate local ensemble predictions."""
        if not self.is_trained:
            return {}
        
        # Individual predictions
        prophet_pred = self._predict_local_prophet(future_data) if self.prophet_model else np.array([])
        lightgbm_pred = self._predict_local_lightgbm(future_data) if self.lightgbm_model else np.array([])
        
        # Ensemble prediction
        predictions_available = []
        if len(prophet_pred) > 0:
            predictions_available.append(('prophet', prophet_pred, self.ensemble_weights['prophet']))
        if len(lightgbm_pred) > 0:
            predictions_available.append(('lightgbm', lightgbm_pred, self.ensemble_weights['lightgbm']))
        
        if not predictions_available:
            return {}
        
        # Calculate ensemble
        weights_sum = sum(weight for _, _, weight in predictions_available)
        ensemble_pred = np.zeros(len(future_data))
        
        for model_name, pred, weight in predictions_available:
            normalized_weight = weight / weights_sum
            ensemble_pred += pred * normalized_weight
        
        # Create results
        predictions = pd.DataFrame({
            'ds': future_data['ds'],
            'prophet_pred': prophet_pred if len(prophet_pred) == len(future_data) else np.nan,
            'lightgbm_pred': lightgbm_pred if len(lightgbm_pred) == len(future_data) else np.nan,
            'ensemble_pred': ensemble_pred,
            'actual': future_data['y'] if 'y' in future_data.columns else np.nan
        })
        
        return {
            'predictions': predictions,
            'ensemble_weights': self.ensemble_weights,
            'models_used': [name for name, _, _ in predictions_available],
            'region_id': self.region_id,
            'business_category': self.business_category,
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def _predict_local_prophet(self, future_data: pd.DataFrame) -> np.ndarray:
        """Generate Prophet predictions."""
        if not self.prophet_model:
            return np.array([])
        
        try:
            future_df = future_data[['ds']].copy()
            
            # Add regressors if available
            if 'is_peak_quarter' in future_data.columns:
                future_df['is_peak_quarter'] = future_data['is_peak_quarter']
            else:
                future_df['is_peak_quarter'] = future_data['ds'].dt.quarter.isin([4, 1])
            
            forecast = self.prophet_model.predict(future_df)
            return forecast['yhat'].values
            
        except Exception as e:
            logger.warning(f"Local Prophet prediction failed [{self.region_id}, {self.business_category}]: {e}")
            return np.array([])
    
    def _predict_local_lightgbm(self, future_data: pd.DataFrame) -> np.ndarray:
        """Generate LightGBM predictions."""
        if not self.lightgbm_model or not self.feature_columns:
            return np.array([])
        
        try:
            X_future = pd.DataFrame()
            
            # Basic time features
            X_future['quarter'] = future_data['ds'].dt.quarter
            X_future['year'] = future_data['ds'].dt.year
            
            # Lag features (use last training values as approximation)
            if hasattr(self, 'training_data') and not self.training_data.empty:
                last_y = self.training_data['y'].iloc[-1]
                if 'y_lag_1' in self.feature_columns:
                    X_future['y_lag_1'] = last_y
                if 'y_rolling_mean_2' in self.feature_columns:
                    last_mean = self.training_data['y'].tail(2).mean()
                    X_future['y_rolling_mean_2'] = last_mean
            
            # Business features
            if 'revenue_per_transaction' in self.feature_columns:
                if 'revenue_per_transaction' in future_data.columns:
                    X_future['revenue_per_transaction'] = future_data['revenue_per_transaction']
                else:
                    # Use historical average
                    if hasattr(self, 'training_data') and not self.training_data.empty:
                        avg_rpt = self.training_data.get('revenue_per_transaction', pd.Series([1000])).mean()
                        X_future['revenue_per_transaction'] = avg_rpt
                    else:
                        X_future['revenue_per_transaction'] = 1000  # Default
            
            # Ensure all features present
            for feature in self.feature_columns:
                if feature not in X_future.columns:
                    X_future[feature] = 0
            
            predictions = self.lightgbm_model.predict(X_future[self.feature_columns])
            return predictions
            
        except Exception as e:
            logger.warning(f"Local LightGBM prediction failed [{self.region_id}, {self.business_category}]: {e}")
            return np.array([])
    
    def _evaluate_local_models(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate local model performance."""
        results = {'test_metrics': {}}
        
        if test_data.empty:
            return results
        
        # Generate predictions
        prophet_pred = self._predict_local_prophet(test_data)
        lightgbm_pred = self._predict_local_lightgbm(test_data)
        
        actual = test_data['y'].values
        
        # Evaluate available models
        models = {}
        if len(prophet_pred) > 0:
            models['prophet'] = prophet_pred
        if len(lightgbm_pred) > 0:
            models['lightgbm'] = lightgbm_pred
        
        # Simple ensemble
        if len(models) > 1:
            ensemble_pred = np.mean(list(models.values()), axis=0)
            models['ensemble'] = ensemble_pred
        
        # Calculate metrics
        for model_name, pred in models.items():
            if len(pred) > 0 and len(actual) > 0:
                try:
                    metrics = {
                        'mae': mean_absolute_error(actual, pred),
                        'rmse': np.sqrt(mean_squared_error(actual, pred)),
                        'mape': mean_absolute_percentage_error(actual, pred) * 100
                    }
                    results['test_metrics'][model_name] = metrics
                except:
                    continue
        
        return results


class SeoulLocalModelManager:
    """
    Manager for coordinating all 72 local models (6 regions × 12 business categories).
    Handles batch training, parallel processing, and model orchestration.
    """
    
    def __init__(self, regional_manager: Optional[SeoulRegionalModelManager] = None,
                 global_model: Optional[SeoulGlobalModel] = None, config_path: Optional[str] = None):
        
        self.config = load_config(config_path)
        self.global_model = global_model
        self.regional_manager = regional_manager
        
        # Model configuration
        self.n_regions = self.config['clustering']['regional'].get('n_clusters', 6)
        self.n_business_categories = self.config['clustering']['business'].get('n_clusters', 12)
        self.total_models = self.n_regions * self.n_business_categories
        
        # Local models storage: {(region_id, business_category): SeoulLocalModel}
        self.local_models = {}
        
        # Training configuration
        self.training_config = LocalModelConfig(0, 0)  # Default config
        self.max_parallel_workers = min(4, mp.cpu_count())  # Limit CPU usage
        self.batch_size = 12  # Models to train in parallel
        
        # Model status tracking
        self.training_status = {}
        self.training_summary = {
            'total_models': self.total_models,
            'successfully_trained': 0,
            'fallback_required': 0,
            'training_failures': 0,
            'training_start_time': None,
            'training_end_time': None
        }
        
        logger.info(f"Local Model Manager initialized for {self.total_models} models ({self.n_regions}×{self.n_business_categories})")
    
    def create_all_local_models(self) -> None:
        """Create all 72 local model instances."""
        logger.info("Creating all local models...")
        
        for region_id in range(self.n_regions):
            for business_cat in range(self.n_business_categories):
                # Get regional model for fallback
                regional_model = None
                if self.regional_manager and region_id in self.regional_manager.regional_models:
                    regional_model = self.regional_manager.regional_models[region_id]
                
                # Create local model
                local_model = SeoulLocalModel(
                    region_id=region_id,
                    business_category=business_cat,
                    regional_model=regional_model,
                    global_model=self.global_model,
                    config=LocalModelConfig(region_id, business_cat)
                )
                
                self.local_models[(region_id, business_cat)] = local_model
        
        logger.info(f"Created {len(self.local_models)} local models")
    
    def train_all_local_models(self, data: pd.DataFrame, 
                             enable_parallel: bool = True) -> Dict[str, Any]:
        """
        Train all local models with parallel processing.
        
        Args:
            data: Training dataset
            enable_parallel: Whether to use parallel processing
            
        Returns:
            Training summary and results
        """
        self.training_summary['training_start_time'] = datetime.now().isoformat()
        logger.info(f"Training {self.total_models} local models (parallel: {enable_parallel})...")
        
        if not self.local_models:
            self.create_all_local_models()
        
        # Prepare model list for training
        model_items = list(self.local_models.items())
        
        if enable_parallel and len(model_items) > 1:
            results = self._train_parallel_batch(data, model_items)
        else:
            results = self._train_sequential(data, model_items)
        
        # Update training summary
        self.training_summary['training_end_time'] = datetime.now().isoformat()
        self._update_training_summary(results)
        
        logger.info(f"Local model training completed: {self.training_summary['successfully_trained']}/{self.total_models} successful")
        
        return {
            'training_summary': self.training_summary,
            'individual_results': results,
            'model_status': self.training_status
        }
    
    def _train_parallel_batch(self, data: pd.DataFrame, 
                            model_items: List[Tuple]) -> Dict[Tuple[int, int], Dict]:
        """Train models in parallel batches."""
        results = {}
        
        # Split into batches
        batches = [model_items[i:i + self.batch_size] 
                  for i in range(0, len(model_items), self.batch_size)]
        
        logger.info(f"Training in {len(batches)} batches of up to {self.batch_size} models")
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Training batch {batch_idx + 1}/{len(batches)} ({len(batch)} models)")
            
            with ThreadPoolExecutor(max_workers=min(self.max_parallel_workers, len(batch))) as executor:
                # Submit training jobs
                future_to_key = {
                    executor.submit(self._train_single_model_safe, data, key, model): key
                    for key, model in batch
                }
                
                # Collect results
                for future in future_to_key:
                    key = future_to_key[future]
                    try:
                        result = future.result(timeout=self.training_config.max_training_time_minutes * 60)
                        results[key] = result
                        self.training_status[key] = 'completed'
                    except Exception as e:
                        logger.error(f"Training failed for {key}: {e}")
                        results[key] = {'error': str(e)}
                        self.training_status[key] = 'failed'
        
        return results
    
    def _train_sequential(self, data: pd.DataFrame, 
                         model_items: List[Tuple]) -> Dict[Tuple[int, int], Dict]:
        """Train models sequentially."""
        results = {}
        
        for idx, (key, model) in enumerate(model_items):
            logger.info(f"Training model {idx + 1}/{len(model_items)}: {key}")
            
            try:
                result = self._train_single_model_safe(data, key, model)
                results[key] = result
                self.training_status[key] = 'completed'
            except Exception as e:
                logger.error(f"Training failed for {key}: {e}")
                results[key] = {'error': str(e)}
                self.training_status[key] = 'failed'
        
        return results
    
    def _train_single_model_safe(self, data: pd.DataFrame, 
                               key: Tuple[int, int], model: SeoulLocalModel) -> Dict[str, Any]:
        """Train single model with error handling."""
        try:
            result = model.train_local_models(data)
            
            # Update status tracking
            if result.get('error'):
                if 'insufficient_data' in result.get('error', ''):
                    self.training_status[key] = 'insufficient_data'
                else:
                    self.training_status[key] = 'training_error'
            else:
                self.training_status[key] = 'success'
            
            return result
            
        except Exception as e:
            self.training_status[key] = 'exception'
            raise e
    
    def _update_training_summary(self, results: Dict[Tuple[int, int], Dict]) -> None:
        """Update training summary statistics."""
        for key, result in results.items():
            if result.get('error'):
                if 'insufficient_data' in result.get('error', ''):
                    self.training_summary['fallback_required'] += 1
                else:
                    self.training_summary['training_failures'] += 1
            else:
                self.training_summary['successfully_trained'] += 1
    
    def get_model(self, region_id: int, business_category: int) -> Optional[SeoulLocalModel]:
        """Get specific local model."""
        return self.local_models.get((region_id, business_category))
    
    def predict_by_local(self, future_data: pd.DataFrame, 
                        region_id: int, business_category: int) -> Dict[str, Any]:
        """Generate predictions for specific local model."""
        model = self.get_model(region_id, business_category)
        
        if not model:
            return {'error': f'model_not_found_{region_id}_{business_category}'}
        
        return model.predict_with_fallback(future_data)
    
    def predict_batch(self, future_data: pd.DataFrame, 
                     model_combinations: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Dict]:
        """Generate predictions for multiple local models."""
        results = {}
        
        for region_id, business_cat in model_combinations:
            result = self.predict_by_local(future_data, region_id, business_cat)
            results[(region_id, business_cat)] = result
        
        return results
    
    def get_training_report(self) -> Dict[str, Any]:
        """Get comprehensive training report."""
        # Status distribution
        status_counts = {}
        for status in self.training_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Model health summary
        healthy_models = sum(1 for model in self.local_models.values() if model.is_trained)
        fallback_models = self.total_models - healthy_models
        
        return {
            'summary': self.training_summary,
            'status_distribution': status_counts,
            'model_health': {
                'total_models': self.total_models,
                'trained_models': healthy_models,
                'fallback_required': fallback_models,
                'health_rate': healthy_models / self.total_models * 100 if self.total_models > 0 else 0
            },
            'detailed_status': self.training_status
        }
    
    def save_all_models(self) -> Dict[str, Any]:
        """Save all trained local models."""
        saved_count = 0
        save_errors = []
        
        base_path = Path(self.config['models'].get('local', {}).get('save_path', 'src/models/local/saved_models'))
        base_path.mkdir(parents=True, exist_ok=True)
        
        for (region_id, business_cat), model in self.local_models.items():
            if model.is_trained:
                try:
                    model_path = base_path / f'local_model_{region_id}_{business_cat}.pkl'
                    joblib.dump(model, model_path)
                    saved_count += 1
                except Exception as e:
                    save_errors.append(f"Failed to save model [{region_id}, {business_cat}]: {e}")
        
        return {
            'saved_models': saved_count,
            'total_models': len([m for m in self.local_models.values() if m.is_trained]),
            'save_errors': save_errors
        }


def main():
    """Main function for testing local models."""
    print("\n=== LOCAL MODEL SYSTEM TEST ===")
    
    # Mock data for testing
    print("Creating test Local Model Manager...")
    
    # Test single local model
    local_model = SeoulLocalModel(
        region_id=0,
        business_category=5,
        config=LocalModelConfig(region_id=0, business_category=5)
    )
    
    print(f"Created Local Model: Region {local_model.region_id}, Business {local_model.business_category}")
    print(f"Data sufficiency: {local_model.data_sufficiency}")
    print(f"Training status: {local_model.is_trained}")
    
    # Test local model manager
    manager = SeoulLocalModelManager()
    print(f"Manager initialized for {manager.total_models} total models")
    print(f"Parallel workers: {manager.max_parallel_workers}")
    print(f"Batch size: {manager.batch_size}")
    
    print("\n=== LOCAL MODEL SYSTEM READY ===")


if __name__ == "__main__":
    main()