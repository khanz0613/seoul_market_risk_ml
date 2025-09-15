"""
Regional Model for Seoul Market Risk ML System
Enhanced forecasting models for each regional cluster using Prophet + ARIMA + LightGBM ensemble.
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
warnings.filterwarnings('ignore')

# Time series forecasting libraries
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# ML libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

# Internal imports
from ..utils.config_loader import load_config, get_data_paths
from .global_model import SeoulGlobalModel

logger = logging.getLogger(__name__)


class SeoulRegionalModel:
    """
    Seoul Regional Model - Enhanced forecasting for regional clusters
    
    Combines Prophet, ARIMA, and LightGBM models for superior regional predictions:
    1. Prophet: Handles seasonality, holidays, and trends with regional adaptations
    2. ARIMA: Captures complex autocorrelations and regional patterns
    3. LightGBM: Learns non-linear patterns and regional characteristics
    4. Ensemble: Intelligent weighted combination optimized per region
    
    Falls back to Global Model when regional data is insufficient.
    """
    
    def __init__(self, region_id: int, region_characteristics: Dict[str, Any], 
                 global_model: Optional[SeoulGlobalModel] = None, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.model_config = self.config['models'].get('regional', {})
        self.data_paths = get_data_paths(self.config)
        
        # Region identification
        self.region_id = region_id
        self.region_characteristics = region_characteristics or {}
        
        # Model storage paths
        self.model_dir = Path(self.model_config.get('save_path', 'src/models/regional/saved_models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.prophet_model = None
        self.arima_model = None
        self.lightgbm_model = None
        
        # Ensemble configuration - more complex than global model
        self.ensemble_weights = {'prophet': 0.4, 'arima': 0.3, 'lightgbm': 0.3}
        
        # Global model for fallback
        self.global_model = global_model
        
        # Model metadata
        self.training_data = None
        self.model_performance = {}
        self.feature_columns = []
        self.target_column = 'monthly_revenue'
        self.min_data_threshold = self.model_config.get('min_data_threshold', 12)  # 12 quarters minimum
        
        # Regional characteristics for feature engineering
        self.regional_features = ['income_level', 'foot_traffic', 'business_diversity', 
                                'district_density', 'competitor_count']
        
        # External features (inherited from global model)
        self.external_features = ['temperature_avg', 'precipitation', 'holiday_count', 
                                'gdp_growth_rate', 'inflation_rate', 'consumer_confidence']
        
        logger.info(f"Regional Model initialized for Region {region_id} with enhanced ensemble")
    
    def prepare_regional_dataset(self, df: pd.DataFrame, 
                               include_external: bool = True) -> pd.DataFrame:
        """
        Prepare dataset for regional model training.
        Filters data for specific region and adds regional characteristics.
        
        Args:
            df: Input DataFrame with sales data
            include_external: Whether to include external features
            
        Returns:
            Prepared regional time series dataset
        """
        logger.info(f"Preparing regional dataset for Region {self.region_id}...")
        
        # Filter data for this region
        if 'region_cluster' in df.columns:
            regional_data = df[df['region_cluster'] == self.region_id].copy()
        else:
            # Fallback: use all data if clustering not available
            logger.warning(f"No region_cluster column found, using sample of data for Region {self.region_id}")
            # Use deterministic sampling based on region_id
            sample_fraction = 1.0 / 6  # Assume 6 regions
            start_idx = int(self.region_id * len(df) * sample_fraction)
            end_idx = int((self.region_id + 1) * len(df) * sample_fraction)
            regional_data = df.iloc[start_idx:end_idx].copy()
        
        if len(regional_data) < self.min_data_threshold:
            logger.warning(f"Insufficient data for Region {self.region_id}: {len(regional_data)} records. Minimum: {self.min_data_threshold}")
            return pd.DataFrame()
        
        # Aggregate to regional quarterly data
        regional_series = regional_data.groupby('quarter_code').agg({
            'monthly_revenue': ['sum', 'mean', 'count'],
            'monthly_transactions': ['sum', 'mean'],
            'district_code': 'nunique',
            'business_type_code': 'nunique'
        }).round(2)
        
        # Flatten column names
        regional_series.columns = [f'{col[0]}_{col[1]}' for col in regional_series.columns]
        regional_series = regional_series.reset_index()
        
        # Create proper date column
        regional_series['year'] = regional_series['quarter_code'].astype(str).str[:4].astype(int)
        regional_series['quarter'] = regional_series['quarter_code'].astype(str).str[4:].astype(int)
        regional_series['ds'] = pd.to_datetime(
            regional_series['year'].astype(str) + '-' + 
            ((regional_series['quarter'] - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
        )
        
        # Main target variable (regional revenue)
        regional_series['y'] = regional_series['monthly_revenue_sum']
        
        # Add derived features specific to regional modeling
        regional_series['revenue_per_business'] = (
            regional_series['monthly_revenue_sum'] / regional_series['monthly_revenue_count']
        )
        regional_series['transactions_per_revenue'] = (
            regional_series['monthly_transactions_sum'] / regional_series['monthly_revenue_sum']
        )
        regional_series['business_density'] = (
            regional_series['business_type_code_nunique'] / regional_series['district_code_nunique']
        )
        
        # Add regional characteristics as features
        for feature, value in self.region_characteristics.items():
            if isinstance(value, (int, float)):
                regional_series[f'regional_{feature}'] = value
        
        # Add external features if available
        if include_external:
            regional_series = self._add_external_features(regional_series)
        
        # Sort by date
        regional_series = regional_series.sort_values('ds').reset_index(drop=True)
        
        # Add regional-specific time features
        regional_series['year_num'] = regional_series['ds'].dt.year
        regional_series['quarter_num'] = regional_series['ds'].dt.quarter
        regional_series['is_holiday_quarter'] = regional_series['quarter_num'].isin([1, 4])
        
        # Regional seasonality adjustments based on characteristics
        if self.region_characteristics.get('business_type_dominant') == 'tourism':
            regional_series['is_peak_season'] = regional_series['quarter_num'].isin([2, 3])  # Spring/Summer
        elif self.region_characteristics.get('business_type_dominant') == 'retail':
            regional_series['is_peak_season'] = regional_series['quarter_num'].isin([4, 1])  # Winter/New Year
        else:
            regional_series['is_peak_season'] = False
        
        logger.info(f"Regional dataset prepared: {len(regional_series)} quarters of data for Region {self.region_id}")
        return regional_series
    
    def _add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add external features adapted for regional model."""
        try:
            # Use similar approach to global model but with regional adjustments
            external_data_path = self.data_paths['external']
            
            # Weather data with regional weights
            weather_file = external_data_path / 'weather_data.csv'
            if weather_file.exists():
                weather_df = pd.read_csv(weather_file, parse_dates=['date'])
                weather_df['quarter_date'] = pd.to_datetime(weather_df['date']).dt.to_period('Q').dt.start_time
                
                weather_agg = weather_df.groupby('quarter_date').agg({
                    'temperature_avg': 'mean',
                    'precipitation': 'sum',
                    'humidity': 'mean'
                }).reset_index()
                weather_agg.rename(columns={'quarter_date': 'ds'}, inplace=True)
                
                # Regional adjustment for weather impact
                weather_sensitivity = self.region_characteristics.get('weather_sensitivity', 1.0)
                weather_agg['temperature_avg'] *= weather_sensitivity
                weather_agg['precipitation'] *= weather_sensitivity
                
                df = df.merge(weather_agg, on='ds', how='left')
                logger.info(f"Added weather features to regional model (Region {self.region_id})")
            
            # Holiday data
            holidays_file = external_data_path / 'holidays_data.csv'
            if holidays_file.exists():
                holidays_df = pd.read_csv(holidays_file, parse_dates=['date'])
                holidays_df['quarter_date'] = pd.to_datetime(holidays_df['date']).dt.to_period('Q').dt.start_time
                
                holiday_count = holidays_df.groupby('quarter_date').size().reset_index(name='holiday_count')
                holiday_count.rename(columns={'quarter_date': 'ds'}, inplace=True)
                
                df = df.merge(holiday_count, on='ds', how='left')
                logger.info(f"Added holiday features to regional model (Region {self.region_id})")
            
            # Economic data
            economic_file = external_data_path / 'economic_data.csv'
            if economic_file.exists():
                economic_df = pd.read_csv(economic_file, parse_dates=['date'])
                economic_df.rename(columns={'date': 'ds'}, inplace=True)
                
                df = df.merge(economic_df[['ds', 'gdp_growth_rate', 'inflation_rate', 
                                          'consumer_confidence']], on='ds', how='left')
                logger.info(f"Added economic indicators to regional model (Region {self.region_id})")
            
        except Exception as e:
            logger.warning(f"Could not load external features for Region {self.region_id}: {e}")
        
        # Fill missing external features
        external_cols = ['temperature_avg', 'precipitation', 'humidity', 'holiday_count',
                        'gdp_growth_rate', 'inflation_rate', 'consumer_confidence']
        
        for col in external_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        
        return df
    
    def train_prophet_model(self, train_data: pd.DataFrame) -> Prophet:
        """
        Train Prophet model with regional adaptations.
        
        Args:
            train_data: Training dataset with 'ds' and 'y' columns
            
        Returns:
            Trained Prophet model
        """
        logger.info(f"Training Prophet model for Region {self.region_id}...")
        
        # Regional-specific Prophet configuration
        region_type = self.region_characteristics.get('region_type', 'mixed')
        
        if region_type == 'business':
            # Business districts have stronger yearly seasonality
            prophet_model = Prophet(
                yearly_seasonality=True,
                quarterly_seasonality=True,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.08,  # More flexible
                seasonality_prior_scale=12,    # Strong seasonality
                holidays_prior_scale=8
            )
        elif region_type == 'residential':
            # Residential areas have different patterns
            prophet_model = Prophet(
                yearly_seasonality=True,
                quarterly_seasonality=True,
                seasonality_mode='additive',
                changepoint_prior_scale=0.03,  # More conservative
                seasonality_prior_scale=8,
                holidays_prior_scale=12        # Holiday effects more important
            )
        else:
            # Mixed or default
            prophet_model = Prophet(
                yearly_seasonality=True,
                quarterly_seasonality=True,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10
            )
        
        # Add external regressors
        for feature in self.external_features:
            if feature in train_data.columns:
                prophet_model.add_regressor(feature)
                logger.info(f"Added regressor to regional Prophet: {feature}")
        
        # Add regional regressors
        for feature in self.regional_features:
            col_name = f'regional_{feature}'
            if col_name in train_data.columns:
                prophet_model.add_regressor(col_name)
                logger.info(f"Added regional regressor: {col_name}")
        
        # Korean holidays
        holidays_df = self._create_korean_holidays(train_data)
        if not holidays_df.empty:
            prophet_model.holidays = holidays_df
        
        # Prepare training data
        prophet_data = train_data[['ds', 'y']].copy()
        
        # Add all regressors
        all_features = self.external_features + [f'regional_{f}' for f in self.regional_features]
        for feature in all_features:
            if feature in train_data.columns:
                prophet_data[feature] = train_data[feature]
        
        # Fit model
        try:
            prophet_model.fit(prophet_data)
            logger.info(f"Prophet model training completed for Region {self.region_id}")
        except Exception as e:
            logger.error(f"Prophet model training failed for Region {self.region_id}: {e}")
            # Fallback to simpler model
            prophet_model = Prophet(
                yearly_seasonality=True,
                quarterly_seasonality=True,
                seasonality_mode='additive'
            )
            prophet_model.fit(prophet_data[['ds', 'y']])
            logger.info(f"Trained simplified Prophet model for Region {self.region_id}")
        
        return prophet_model
    
    def _create_korean_holidays(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create Korean holidays DataFrame for Prophet (same as global model)."""
        holidays = []
        
        start_year = data['ds'].dt.year.min()
        end_year = data['ds'].dt.year.max()
        
        for year in range(start_year, end_year + 1):
            holidays.extend([
                {'holiday': 'New_Year', 'ds': f'{year}-01-01', 'lower_window': -1, 'upper_window': 1},
                {'holiday': 'Independence_Movement', 'ds': f'{year}-03-01', 'lower_window': 0, 'upper_window': 0},
                {'holiday': 'Labor_Day', 'ds': f'{year}-05-01', 'lower_window': 0, 'upper_window': 0},
                {'holiday': 'Childrens_Day', 'ds': f'{year}-05-05', 'lower_window': 0, 'upper_window': 0},
                {'holiday': 'Liberation_Day', 'ds': f'{year}-08-15', 'lower_window': 0, 'upper_window': 0},
                {'holiday': 'National_Foundation', 'ds': f'{year}-10-03', 'lower_window': 0, 'upper_window': 0},
                {'holiday': 'Christmas', 'ds': f'{year}-12-25', 'lower_window': -1, 'upper_window': 1}
            ])
        
        if holidays:
            holidays_df = pd.DataFrame(holidays)
            holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
            return holidays_df
        else:
            return pd.DataFrame()
    
    def train_arima_model(self, train_data: pd.DataFrame) -> ARIMA:
        """
        Train ARIMA model with regional optimizations.
        
        Args:
            train_data: Training dataset
            
        Returns:
            Trained ARIMA model
        """
        logger.info(f"Training ARIMA model for Region {self.region_id}...")
        
        # Prepare time series
        ts = train_data.set_index('ds')['y']
        
        # Check for stationarity
        adf_result = adfuller(ts.dropna())
        is_stationary = adf_result[1] <= 0.05
        
        if not is_stationary:
            ts_diff = ts.diff().dropna()
            adf_result_diff = adfuller(ts_diff)
            d = 1 if adf_result_diff[1] <= 0.05 else 2
            logger.info(f"Region {self.region_id}: Series not stationary, using d={d}")
        else:
            d = 0
            logger.info(f"Region {self.region_id}: Series is stationary, using d=0")
        
        # Auto-select ARIMA parameters - more extensive search for regional models
        best_aic = float('inf')
        best_params = (1, d, 1)
        
        # Extended grid search for regional models
        for p in range(0, 5):  # More AR terms
            for q in range(0, 5):  # More MA terms
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        
                except Exception as e:
                    continue
        
        logger.info(f"Region {self.region_id}: Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
        
        # Train final model
        try:
            arima_model = ARIMA(ts, order=best_params)
            fitted_arima = arima_model.fit()
            logger.info(f"ARIMA model training completed for Region {self.region_id}")
            return fitted_arima
        except Exception as e:
            logger.error(f"ARIMA model training failed for Region {self.region_id}: {e}")
            # Fallback
            arima_model = ARIMA(ts, order=(1, 1, 1))
            fitted_arima = arima_model.fit()
            logger.info(f"Trained fallback ARIMA(1,1,1) for Region {self.region_id}")
            return fitted_arima
    
    def train_lightgbm_model(self, train_data: pd.DataFrame) -> lgb.LGBMRegressor:
        """
        Train LightGBM model for non-linear pattern recognition.
        
        Args:
            train_data: Training dataset
            
        Returns:
            Trained LightGBM model
        """
        logger.info(f"Training LightGBM model for Region {self.region_id}...")
        
        # Prepare features for LightGBM
        feature_columns = []
        X_train = pd.DataFrame()
        
        # Time-based features
        X_train['year'] = train_data['ds'].dt.year
        X_train['quarter'] = train_data['ds'].dt.quarter
        X_train['year_quarter'] = X_train['year'] * 10 + X_train['quarter']
        feature_columns.extend(['year', 'quarter', 'year_quarter'])
        
        # Lag features (important for time series)
        y_series = train_data['y']
        for lag in [1, 2, 4]:  # 1, 2, and 4 quarters ago
            if len(y_series) > lag:
                X_train[f'y_lag_{lag}'] = y_series.shift(lag)
                feature_columns.append(f'y_lag_{lag}')
        
        # Rolling statistics
        for window in [2, 4]:
            if len(y_series) > window:
                X_train[f'y_rolling_mean_{window}'] = y_series.rolling(window=window).mean()
                X_train[f'y_rolling_std_{window}'] = y_series.rolling(window=window).std()
                feature_columns.extend([f'y_rolling_mean_{window}', f'y_rolling_std_{window}'])
        
        # External features
        for feature in self.external_features:
            if feature in train_data.columns:
                X_train[feature] = train_data[feature]
                feature_columns.append(feature)
        
        # Regional characteristics
        for feature in self.regional_features:
            col_name = f'regional_{feature}'
            if col_name in train_data.columns:
                X_train[feature] = train_data[col_name]
                feature_columns.append(feature)
        
        # Business-derived features
        derived_features = ['revenue_per_business', 'transactions_per_revenue', 'business_density']
        for feature in derived_features:
            if feature in train_data.columns:
                X_train[feature] = train_data[feature]
                feature_columns.append(feature)
        
        # Remove rows with NaN (due to lag features)
        valid_indices = X_train.dropna().index
        X_train = X_train.loc[valid_indices]
        y_train = train_data.loc[valid_indices, 'y']
        
        self.feature_columns = feature_columns
        
        if len(X_train) < 6:  # Minimum data for LightGBM
            logger.warning(f"Insufficient data for LightGBM in Region {self.region_id}: {len(X_train)} samples")
            return None
        
        # LightGBM parameters optimized for regional modeling
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': min(31, 2 ** int(np.log2(len(X_train))) - 1),  # Adaptive based on data size
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_child_samples': max(5, len(X_train) // 10),  # Adaptive
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        try:
            # Train LightGBM
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(
                X_train[feature_columns], 
                y_train,
                eval_set=[(X_train[feature_columns], y_train)],
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
            )
            
            logger.info(f"LightGBM model training completed for Region {self.region_id}")
            return lgb_model
            
        except Exception as e:
            logger.error(f"LightGBM model training failed for Region {self.region_id}: {e}")
            return None
    
    def train_regional_ensemble(self, df: pd.DataFrame, 
                              train_test_split: float = 0.8) -> Dict[str, Any]:
        """
        Train complete regional model ensemble.
        
        Args:
            df: Input dataset
            train_test_split: Ratio for train/test split
            
        Returns:
            Training results and performance metrics
        """
        logger.info(f"Training Regional Model ensemble for Region {self.region_id}...")
        
        # Prepare regional dataset
        regional_data = self.prepare_regional_dataset(df)
        
        if len(regional_data) < self.min_data_threshold:
            logger.error(f"Insufficient data for Region {self.region_id}. Fallback to Global Model required.")
            return {'error': 'insufficient_data', 'fallback_required': True}
        
        self.training_data = regional_data.copy()
        
        # Split data
        split_idx = int(len(regional_data) * train_test_split)
        train_data = regional_data[:split_idx].copy()
        test_data = regional_data[split_idx:].copy()
        
        logger.info(f"Region {self.region_id}: Training on {len(train_data)} quarters, testing on {len(test_data)} quarters")
        
        # Train individual models
        self.prophet_model = self.train_prophet_model(train_data)
        self.arima_model = self.train_arima_model(train_data)
        self.lightgbm_model = self.train_lightgbm_model(train_data)
        
        # Evaluate models
        results = self.evaluate_models(train_data, test_data)
        
        # Optimize ensemble weights
        if len(test_data) > 0:
            self.ensemble_weights = self._optimize_ensemble_weights(test_data, results)
        
        # Save models
        self.save_models()
        
        logger.info(f"Regional Model training completed for Region {self.region_id}. Weights: {self.ensemble_weights}")
        return results
    
    def predict_with_fallback(self, future_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate predictions with fallback to Global Model if needed.
        
        Args:
            future_data: Future data for prediction
            
        Returns:
            Prediction results with fallback information
        """
        logger.info(f"Generating predictions for Region {self.region_id} with fallback capability")
        
        try:
            # Try regional prediction first
            regional_predictions = self.predict_ensemble(future_data)
            
            if regional_predictions and 'predictions' in regional_predictions:
                pred_df = regional_predictions['predictions']
                # Check prediction quality
                if not pred_df.empty and not pred_df['ensemble_pred'].isna().all():
                    regional_predictions['used_fallback'] = False
                    regional_predictions['prediction_source'] = f'regional_{self.region_id}'
                    return regional_predictions
            
            # If regional prediction failed or poor quality, use global fallback
            logger.warning(f"Regional prediction failed for Region {self.region_id}, using Global Model fallback")
            
            if self.global_model is None:
                logger.error("No Global Model available for fallback")
                return {'error': 'no_fallback_available'}
            
            global_predictions = self.global_model.predict_ensemble(future_data)
            if global_predictions:
                global_predictions['used_fallback'] = True
                global_predictions['prediction_source'] = 'global_fallback'
                global_predictions['original_region'] = self.region_id
            
            return global_predictions
            
        except Exception as e:
            logger.error(f"Prediction failed for Region {self.region_id}: {e}")
            
            # Final fallback to global model
            if self.global_model:
                try:
                    global_predictions = self.global_model.predict_ensemble(future_data)
                    global_predictions['used_fallback'] = True
                    global_predictions['prediction_source'] = 'global_emergency_fallback'
                    global_predictions['error'] = str(e)
                    return global_predictions
                except Exception as global_e:
                    logger.error(f"Global fallback also failed: {global_e}")
            
            return {'error': f'all_predictions_failed: {e}'}
    
    def predict_ensemble(self, future_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble predictions using all three models."""
        logger.info(f"Generating ensemble predictions for Region {self.region_id}: {len(future_data)} periods")
        
        # Individual predictions
        prophet_pred = self.predict_prophet(future_data) if self.prophet_model else np.array([])
        arima_pred = self.predict_arima(future_data) if self.arima_model else np.array([])
        lightgbm_pred = self.predict_lightgbm(future_data) if self.lightgbm_model else np.array([])
        
        # Ensemble prediction with adaptive weights
        predictions_available = []
        weights_sum = 0
        
        if len(prophet_pred) > 0:
            predictions_available.append(('prophet', prophet_pred, self.ensemble_weights['prophet']))
            weights_sum += self.ensemble_weights['prophet']
        
        if len(arima_pred) > 0:
            predictions_available.append(('arima', arima_pred, self.ensemble_weights['arima']))
            weights_sum += self.ensemble_weights['arima']
        
        if len(lightgbm_pred) > 0:
            predictions_available.append(('lightgbm', lightgbm_pred, self.ensemble_weights['lightgbm']))
            weights_sum += self.ensemble_weights['lightgbm']
        
        if not predictions_available:
            logger.error(f"No models available for prediction in Region {self.region_id}")
            return {}
        
        # Normalize weights
        ensemble_pred = np.zeros(len(future_data))
        for model_name, pred, weight in predictions_available:
            normalized_weight = weight / weights_sum
            ensemble_pred += pred * normalized_weight
        
        # Create results DataFrame
        predictions = pd.DataFrame({
            'ds': future_data['ds'],
            'prophet_pred': prophet_pred if len(prophet_pred) == len(future_data) else np.nan,
            'arima_pred': arima_pred if len(arima_pred) == len(future_data) else np.nan,
            'lightgbm_pred': lightgbm_pred if len(lightgbm_pred) == len(future_data) else np.nan,
            'ensemble_pred': ensemble_pred,
            'actual': future_data['y'] if 'y' in future_data.columns else np.nan
        })
        
        return {
            'predictions': predictions,
            'ensemble_weights': self.ensemble_weights,
            'models_used': [name for name, _, _ in predictions_available],
            'region_id': self.region_id,
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def predict_prophet(self, future_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using Prophet model."""
        if self.prophet_model is None:
            return np.array([])
        
        try:
            future_df = future_data[['ds']].copy()
            
            # Add all regressors
            all_features = self.external_features + [f'regional_{f}' for f in self.regional_features]
            for feature in all_features:
                if feature in future_data.columns:
                    future_df[feature] = future_data[feature]
                else:
                    future_df[feature] = 0
            
            forecast = self.prophet_model.predict(future_df)
            return forecast['yhat'].values
            
        except Exception as e:
            logger.error(f"Prophet prediction failed for Region {self.region_id}: {e}")
            return np.array([])
    
    def predict_arima(self, future_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using ARIMA model."""
        if self.arima_model is None:
            return np.array([])
        
        try:
            n_periods = len(future_data)
            forecast = self.arima_model.forecast(steps=n_periods)
            return forecast
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed for Region {self.region_id}: {e}")
            return np.array([])
    
    def predict_lightgbm(self, future_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using LightGBM model."""
        if self.lightgbm_model is None or not self.feature_columns:
            return np.array([])
        
        try:
            # Prepare features (similar to training)
            X_future = pd.DataFrame()
            
            # Time-based features
            X_future['year'] = future_data['ds'].dt.year
            X_future['quarter'] = future_data['ds'].dt.quarter
            X_future['year_quarter'] = X_future['year'] * 10 + X_future['quarter']
            
            # For lag features, use last known values (simplified approach)
            if hasattr(self, 'training_data') and not self.training_data.empty:
                last_y = self.training_data['y'].iloc[-1]
                for lag in [1, 2, 4]:
                    X_future[f'y_lag_{lag}'] = last_y  # Simplified - use last value
            
            # Rolling statistics (simplified)
            for window in [2, 4]:
                if hasattr(self, 'training_data') and not self.training_data.empty:
                    last_mean = self.training_data['y'].tail(window).mean()
                    last_std = self.training_data['y'].tail(window).std()
                    X_future[f'y_rolling_mean_{window}'] = last_mean
                    X_future[f'y_rolling_std_{window}'] = last_std
            
            # External and regional features
            for feature in self.external_features:
                if feature in future_data.columns:
                    X_future[feature] = future_data[feature]
                else:
                    X_future[feature] = 0
            
            for feature in self.regional_features:
                col_name = f'regional_{feature}'
                if col_name in future_data.columns:
                    X_future[feature] = future_data[col_name]
                elif feature in self.region_characteristics:
                    X_future[feature] = self.region_characteristics[feature]
                else:
                    X_future[feature] = 0
            
            # Business-derived features (use reasonable defaults)
            derived_defaults = {'revenue_per_business': 1000000, 'transactions_per_revenue': 0.1, 'business_density': 5.0}
            for feature in ['revenue_per_business', 'transactions_per_revenue', 'business_density']:
                if feature in future_data.columns:
                    X_future[feature] = future_data[feature]
                else:
                    X_future[feature] = derived_defaults.get(feature, 0)
            
            # Ensure all required features are present
            for feature in self.feature_columns:
                if feature not in X_future.columns:
                    X_future[feature] = 0
            
            # Make prediction
            predictions = self.lightgbm_model.predict(X_future[self.feature_columns])
            return predictions
            
        except Exception as e:
            logger.error(f"LightGBM prediction failed for Region {self.region_id}: {e}")
            return np.array([])
    
    def evaluate_models(self, train_data: pd.DataFrame, 
                       test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate individual models and ensemble performance."""
        results = {
            'train_metrics': {},
            'test_metrics': {},
            'predictions': {},
            'model_comparison': {}
        }
        
        if len(test_data) == 0:
            logger.warning(f"No test data available for evaluation in Region {self.region_id}")
            return results
        
        # Generate predictions
        prophet_pred = self.predict_prophet(test_data)
        arima_pred = self.predict_arima(test_data)
        lightgbm_pred = self.predict_lightgbm(test_data)
        
        # Calculate ensemble
        predictions_available = []
        if len(prophet_pred) > 0:
            predictions_available.append(('prophet', prophet_pred))
        if len(arima_pred) > 0:
            predictions_available.append(('arima', arima_pred))
        if len(lightgbm_pred) > 0:
            predictions_available.append(('lightgbm', lightgbm_pred))
        
        if not predictions_available:
            return results
        
        # Simple ensemble (equal weights for evaluation)
        ensemble_pred = np.mean([pred for _, pred in predictions_available], axis=0)
        
        actual = test_data['y'].values
        
        # Calculate metrics
        models = {
            'prophet': prophet_pred if len(prophet_pred) > 0 else None,
            'arima': arima_pred if len(arima_pred) > 0 else None,
            'lightgbm': lightgbm_pred if len(lightgbm_pred) > 0 else None,
            'ensemble': ensemble_pred
        }
        
        for model_name, pred in models.items():
            if pred is not None and len(pred) > 0 and len(actual) > 0:
                metrics = {
                    'mae': mean_absolute_error(actual, pred),
                    'rmse': np.sqrt(mean_squared_error(actual, pred)),
                    'mape': mean_absolute_percentage_error(actual, pred) * 100,
                    'r2': 1 - (np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
                }
                results['test_metrics'][model_name] = metrics
        
        return results
    
    def _optimize_ensemble_weights(self, test_data: pd.DataFrame, 
                                 results: Dict[str, Any]) -> Dict[str, float]:
        """Optimize ensemble weights based on test performance."""
        if not results.get('test_metrics'):
            return self.ensemble_weights
        
        # Get MAPE scores (lower is better)
        prophet_mape = results['test_metrics'].get('prophet', {}).get('mape', 100)
        arima_mape = results['test_metrics'].get('arima', {}).get('mape', 100)
        lightgbm_mape = results['test_metrics'].get('lightgbm', {}).get('mape', 100)
        
        # Weight inversely proportional to error
        prophet_weight = 1 / (prophet_mape + 1) if prophet_mape < 100 else 0
        arima_weight = 1 / (arima_mape + 1) if arima_mape < 100 else 0
        lightgbm_weight = 1 / (lightgbm_mape + 1) if lightgbm_mape < 100 else 0
        
        total_weight = prophet_weight + arima_weight + lightgbm_weight
        
        if total_weight > 0:
            optimized_weights = {
                'prophet': prophet_weight / total_weight,
                'arima': arima_weight / total_weight,
                'lightgbm': lightgbm_weight / total_weight
            }
        else:
            # Fallback to equal weights
            optimized_weights = {'prophet': 0.33, 'arima': 0.33, 'lightgbm': 0.34}
        
        logger.info(f"Region {self.region_id} optimized weights: {optimized_weights}")
        return optimized_weights
    
    def save_models(self) -> Dict[str, Path]:
        """Save trained models to disk."""
        saved_files = {}
        
        try:
            region_model_dir = self.model_dir / f'region_{self.region_id}'
            region_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save Prophet model
            if self.prophet_model:
                prophet_file = region_model_dir / f'prophet_region_{self.region_id}.json'
                with open(prophet_file, 'w') as f:
                    json.dump(self.prophet_model.to_json(), f, indent=2)
                saved_files['prophet'] = prophet_file
            
            # Save ARIMA model
            if self.arima_model:
                arima_file = region_model_dir / f'arima_region_{self.region_id}.pkl'
                joblib.dump(self.arima_model, arima_file)
                saved_files['arima'] = arima_file
            
            # Save LightGBM model
            if self.lightgbm_model:
                lgb_file = region_model_dir / f'lightgbm_region_{self.region_id}.pkl'
                joblib.dump(self.lightgbm_model, lgb_file)
                saved_files['lightgbm'] = lgb_file
            
            # Save metadata
            metadata = {
                'region_id': self.region_id,
                'region_characteristics': self.region_characteristics,
                'ensemble_weights': self.ensemble_weights,
                'feature_columns': self.feature_columns,
                'training_timestamp': datetime.now().isoformat(),
                'target_column': self.target_column,
                'external_features': self.external_features,
                'regional_features': self.regional_features
            }
            
            metadata_file = region_model_dir / f'regional_model_metadata_{self.region_id}.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            saved_files['metadata'] = metadata_file
            
            logger.info(f"Regional models saved for Region {self.region_id}: {len(saved_files)} files")
            
        except Exception as e:
            logger.error(f"Error saving models for Region {self.region_id}: {e}")
        
        return saved_files
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            region_model_dir = self.model_dir / f'region_{self.region_id}'
            
            # Load metadata
            metadata_file = region_model_dir / f'regional_model_metadata_{self.region_id}.json'
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.ensemble_weights = metadata.get('ensemble_weights', self.ensemble_weights)
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.region_characteristics = metadata.get('region_characteristics', {})
            
            # Load Prophet model
            prophet_file = region_model_dir / f'prophet_region_{self.region_id}.json'
            if prophet_file.exists():
                with open(prophet_file, 'r') as f:
                    self.prophet_model = Prophet().from_json(json.load(f))
                logger.info(f"Prophet model loaded for Region {self.region_id}")
            
            # Load ARIMA model
            arima_file = region_model_dir / f'arima_region_{self.region_id}.pkl'
            if arima_file.exists():
                self.arima_model = joblib.load(arima_file)
                logger.info(f"ARIMA model loaded for Region {self.region_id}")
            
            # Load LightGBM model
            lgb_file = region_model_dir / f'lightgbm_region_{self.region_id}.pkl'
            if lgb_file.exists():
                self.lightgbm_model = joblib.load(lgb_file)
                logger.info(f"LightGBM model loaded for Region {self.region_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models for Region {self.region_id}: {e}")
            return False


class SeoulRegionalModelManager:
    """
    Manager for coordinating multiple regional models.
    Handles training, prediction, and fallback logic across all regions.
    """
    
    def __init__(self, global_model: Optional[SeoulGlobalModel] = None, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.global_model = global_model
        self.regional_models = {}  # {region_id: SeoulRegionalModel}
        self.n_regions = self.config['clustering']['regional'].get('n_clusters', 6)
        
        logger.info(f"Regional Model Manager initialized for {self.n_regions} regions")
    
    def create_regional_models(self, region_characteristics: Dict[int, Dict[str, Any]]) -> None:
        """Create all regional models with their characteristics."""
        for region_id in range(self.n_regions):
            characteristics = region_characteristics.get(region_id, {})
            self.regional_models[region_id] = SeoulRegionalModel(
                region_id=region_id,
                region_characteristics=characteristics,
                global_model=self.global_model
            )
        
        logger.info(f"Created {len(self.regional_models)} regional models")
    
    def train_all_regional_models(self, data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Train all regional models in parallel."""
        results = {}
        
        for region_id, model in self.regional_models.items():
            logger.info(f"Training Regional Model {region_id}...")
            try:
                result = model.train_regional_ensemble(data)
                results[region_id] = result
            except Exception as e:
                logger.error(f"Failed to train Regional Model {region_id}: {e}")
                results[region_id] = {'error': str(e)}
        
        return results
    
    def predict_by_region(self, future_data: pd.DataFrame, region_id: int) -> Dict[str, Any]:
        """Generate predictions for specific region."""
        if region_id not in self.regional_models:
            logger.error(f"Region {region_id} not found")
            return {'error': f'region_{region_id}_not_found'}
        
        return self.regional_models[region_id].predict_with_fallback(future_data)


def main():
    """Main function for testing regional models."""
    from .global_model import SeoulGlobalModel
    
    # Initialize Global Model first
    global_model = SeoulGlobalModel()
    
    # Mock regional characteristics for testing
    region_characteristics = {
        0: {'region_type': 'business', 'income_level': 1.2, 'foot_traffic': 1.5, 'weather_sensitivity': 0.8},
        1: {'region_type': 'residential', 'income_level': 0.9, 'foot_traffic': 0.7, 'weather_sensitivity': 1.1},
        2: {'region_type': 'mixed', 'income_level': 1.0, 'foot_traffic': 1.0, 'weather_sensitivity': 1.0},
        3: {'region_type': 'business', 'income_level': 1.1, 'foot_traffic': 1.3, 'weather_sensitivity': 0.9},
        4: {'region_type': 'residential', 'income_level': 0.8, 'foot_traffic': 0.6, 'weather_sensitivity': 1.2},
        5: {'region_type': 'mixed', 'income_level': 0.95, 'foot_traffic': 0.9, 'weather_sensitivity': 1.05}
    }
    
    # Test single regional model
    regional_model = SeoulRegionalModel(
        region_id=0,
        region_characteristics=region_characteristics[0],
        global_model=global_model
    )
    
    print("\n=== REGIONAL MODEL TEST ===")
    print(f"Initialized Regional Model for Region 0")
    print(f"Region characteristics: {region_characteristics[0]}")
    print(f"Fallback available: {regional_model.global_model is not None}")


if __name__ == "__main__":
    main()