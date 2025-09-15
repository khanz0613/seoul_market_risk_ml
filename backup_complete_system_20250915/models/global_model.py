"""
Global Model for Seoul Market Risk ML System
Baseline forecasting model using Prophet + ARIMA for entire Seoul market patterns.
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ML libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression

from ..utils.config_loader import load_config, get_data_paths


logger = logging.getLogger(__name__)


class SeoulGlobalModel:
    """
    Seoul Global Model - Baseline forecasting for entire market
    
    Combines Prophet and ARIMA models to create robust baseline predictions:
    1. Prophet: Handles seasonality, holidays, and trends automatically
    2. ARIMA: Captures complex autocorrelations and provides stability
    3. Ensemble: Weighted combination for improved accuracy
    
    Serves as foundation and fallback for Regional and Local models.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.model_config = self.config['models']['global']
        self.data_paths = get_data_paths(self.config)
        
        # Model storage paths
        self.model_dir = Path(self.model_config['save_path'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.prophet_model = None
        self.arima_model = None
        self.ensemble_weights = {'prophet': 0.6, 'arima': 0.4}
        
        # Model metadata
        self.training_data = None
        self.model_performance = {}
        self.feature_columns = []
        self.target_column = 'monthly_revenue'
        
        # External features for Prophet
        self.external_features = ['temperature_avg', 'precipitation', 'holiday_count', 
                                'gdp_growth_rate', 'inflation_rate', 'consumer_confidence']
        
        logger.info("Seoul Global Model initialized with Prophet + ARIMA ensemble")
    
    def prepare_global_dataset(self, df: pd.DataFrame, 
                             include_external: bool = True) -> pd.DataFrame:
        """
        Prepare dataset for global model training.
        Aggregates all Seoul data into time series.
        
        Args:
            df: Input DataFrame with sales data
            include_external: Whether to include external features
            
        Returns:
            Prepared time series dataset
        """
        logger.info("Preparing global dataset for model training...")
        
        # Aggregate to Seoul-wide quarterly data
        global_series = df.groupby('quarter_code').agg({
            'monthly_revenue': ['sum', 'mean', 'count'],
            'monthly_transactions': ['sum', 'mean'],
            'district_code': 'nunique',
            'business_type_code': 'nunique'
        }).round(2)
        
        # Flatten column names
        global_series.columns = [f'{col[0]}_{col[1]}' for col in global_series.columns]
        global_series = global_series.reset_index()
        
        # Create proper date column
        global_series['year'] = global_series['quarter_code'].astype(str).str[:4].astype(int)
        global_series['quarter'] = global_series['quarter_code'].astype(str).str[4:].astype(int)
        global_series['ds'] = pd.to_datetime(
            global_series['year'].astype(str) + '-' + 
            ((global_series['quarter'] - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
        )
        
        # Main target variable (total Seoul revenue)
        global_series['y'] = global_series['monthly_revenue_sum']
        
        # Add derived features
        global_series['revenue_per_business'] = (
            global_series['monthly_revenue_sum'] / global_series['monthly_revenue_count']
        )
        global_series['transactions_per_revenue'] = (
            global_series['monthly_transactions_sum'] / global_series['monthly_revenue_sum']
        )
        
        # Add external features if available
        if include_external:
            global_series = self._add_external_features(global_series)
        
        # Sort by date
        global_series = global_series.sort_values('ds').reset_index(drop=True)
        
        # Add time-based features for modeling
        global_series['year_num'] = global_series['ds'].dt.year
        global_series['quarter_num'] = global_series['ds'].dt.quarter
        global_series['is_holiday_quarter'] = global_series['quarter_num'].isin([1, 4])  # New Year and Christmas
        
        logger.info(f"Global dataset prepared: {len(global_series)} quarters of data")
        return global_series
    
    def _add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add external features from weather, holidays, economic data."""
        try:
            # Try to load external data
            external_data_path = self.data_paths['external']
            
            # Weather data
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
                
                df = df.merge(weather_agg, on='ds', how='left')
                logger.info("Added weather features to global model")
            
            # Holiday data
            holidays_file = external_data_path / 'holidays_data.csv'
            if holidays_file.exists():
                holidays_df = pd.read_csv(holidays_file, parse_dates=['date'])
                holidays_df['quarter_date'] = pd.to_datetime(holidays_df['date']).dt.to_period('Q').dt.start_time
                
                holiday_count = holidays_df.groupby('quarter_date').size().reset_index(name='holiday_count')
                holiday_count.rename(columns={'quarter_date': 'ds'}, inplace=True)
                
                df = df.merge(holiday_count, on='ds', how='left')
                logger.info("Added holiday features to global model")
            
            # Economic data
            economic_file = external_data_path / 'economic_data.csv'
            if economic_file.exists():
                economic_df = pd.read_csv(economic_file, parse_dates=['date'])
                economic_df.rename(columns={'date': 'ds'}, inplace=True)
                
                df = df.merge(economic_df[['ds', 'gdp_growth_rate', 'inflation_rate', 
                                          'consumer_confidence']], on='ds', how='left')
                logger.info("Added economic indicators to global model")
            
        except Exception as e:
            logger.warning(f"Could not load external features: {e}")
        
        # Fill missing external features with reasonable defaults
        external_cols = ['temperature_avg', 'precipitation', 'humidity', 'holiday_count',
                        'gdp_growth_rate', 'inflation_rate', 'consumer_confidence']
        
        for col in external_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        
        return df
    
    def train_prophet_model(self, train_data: pd.DataFrame) -> Prophet:
        """
        Train Prophet model with seasonality and external regressors.
        
        Args:
            train_data: Training dataset with 'ds' and 'y' columns
            
        Returns:
            Trained Prophet model
        """
        logger.info("Training Prophet model...")
        
        # Configure Prophet model
        prophet_model = Prophet(
            yearly_seasonality=True,
            quarterly_seasonality=True,
            weekly_seasonality=False,  # Not relevant for quarterly data
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,  # Conservative changepoint detection
            seasonality_prior_scale=10,    # Allow strong seasonality
            holidays_prior_scale=10,       # Allow holiday effects
            mcmc_samples=0,                # Use MAP instead of MCMC for speed
            uncertainty_samples=1000
        )
        
        # Add external regressors if available
        for feature in self.external_features:
            if feature in train_data.columns:
                prophet_model.add_regressor(feature)
                logger.info(f"Added regressor: {feature}")
        
        # Add custom holidays/events
        holidays_df = self._create_korean_holidays(train_data)
        if not holidays_df.empty:
            prophet_model.holidays = holidays_df
            logger.info(f"Added {len(holidays_df)} Korean holidays")
        
        # Prepare training data
        prophet_data = train_data[['ds', 'y']].copy()
        
        # Add external features
        for feature in self.external_features:
            if feature in train_data.columns:
                prophet_data[feature] = train_data[feature]
        
        # Fit model
        try:
            prophet_model.fit(prophet_data)
            logger.info("Prophet model training completed successfully")
        except Exception as e:
            logger.error(f"Prophet model training failed: {e}")
            # Fallback to simpler model without external regressors
            prophet_model = Prophet(
                yearly_seasonality=True,
                quarterly_seasonality=True,
                seasonality_mode='additive'
            )
            prophet_model.fit(prophet_data[['ds', 'y']])
            logger.info("Trained simplified Prophet model as fallback")
        
        return prophet_model
    
    def _create_korean_holidays(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create Korean holidays DataFrame for Prophet."""
        holidays = []
        
        # Get date range from data
        start_year = data['ds'].dt.year.min()
        end_year = data['ds'].dt.year.max()
        
        for year in range(start_year, end_year + 1):
            # Major Korean holidays (simplified - in production use proper lunar calendar)
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
        Train ARIMA model with automatic parameter selection.
        
        Args:
            train_data: Training dataset
            
        Returns:
            Trained ARIMA model
        """
        logger.info("Training ARIMA model...")
        
        # Prepare time series
        ts = train_data.set_index('ds')['y']
        
        # Check for stationarity and difference if needed
        adf_result = adfuller(ts.dropna())
        is_stationary = adf_result[1] <= 0.05
        
        if not is_stationary:
            # First difference
            ts_diff = ts.diff().dropna()
            adf_result_diff = adfuller(ts_diff)
            d = 1 if adf_result_diff[1] <= 0.05 else 2
            logger.info(f"Series not stationary, using d={d}")
        else:
            d = 0
            logger.info("Series is stationary, using d=0")
        
        # Auto-select ARIMA parameters using AIC
        best_aic = float('inf')
        best_params = (1, d, 1)
        
        # Grid search for p and q (limited range for quarterly data)
        for p in range(0, 4):  # AR terms
            for q in range(0, 4):  # MA terms
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        
                except Exception as e:
                    continue
        
        logger.info(f"Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
        
        # Train final model
        try:
            arima_model = ARIMA(ts, order=best_params)
            fitted_arima = arima_model.fit()
            logger.info("ARIMA model training completed successfully")
            return fitted_arima
        except Exception as e:
            logger.error(f"ARIMA model training failed: {e}")
            # Fallback to simple ARIMA(1,1,1)
            arima_model = ARIMA(ts, order=(1, 1, 1))
            fitted_arima = arima_model.fit()
            logger.info("Trained fallback ARIMA(1,1,1) model")
            return fitted_arima
    
    def train_global_models(self, df: pd.DataFrame, 
                          train_test_split: float = 0.8) -> Dict[str, Any]:
        """
        Train complete global model ensemble.
        
        Args:
            df: Input dataset
            train_test_split: Ratio for train/test split
            
        Returns:
            Training results and performance metrics
        """
        logger.info("Training Global Model ensemble...")
        
        # Prepare dataset
        global_data = self.prepare_global_dataset(df)
        self.training_data = global_data.copy()
        
        # Split data
        split_idx = int(len(global_data) * train_test_split)
        train_data = global_data[:split_idx].copy()
        test_data = global_data[split_idx:].copy()
        
        logger.info(f"Training on {len(train_data)} quarters, testing on {len(test_data)} quarters")
        
        # Train individual models
        self.prophet_model = self.train_prophet_model(train_data)
        self.arima_model = self.train_arima_model(train_data)
        
        # Evaluate models
        results = self.evaluate_models(train_data, test_data)
        
        # Optimize ensemble weights
        if len(test_data) > 0:
            self.ensemble_weights = self._optimize_ensemble_weights(test_data, results)
        
        # Save models
        self.save_models()
        
        logger.info(f"Global Model training completed. Ensemble weights: {self.ensemble_weights}")
        return results
    
    def evaluate_models(self, train_data: pd.DataFrame, 
                       test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate individual models and ensemble performance."""
        logger.info("Evaluating model performance...")
        
        results = {
            'train_metrics': {},
            'test_metrics': {},
            'predictions': {},
            'model_comparison': {}
        }
        
        if len(test_data) == 0:
            logger.warning("No test data available for evaluation")
            return results
        
        # Generate predictions
        prophet_pred = self.predict_prophet(test_data)
        arima_pred = self.predict_arima(test_data)
        
        # Ensemble prediction
        ensemble_pred = (
            prophet_pred * self.ensemble_weights['prophet'] + 
            arima_pred * self.ensemble_weights['arima']
        )
        
        actual = test_data['y'].values
        
        # Calculate metrics for each model
        models = {
            'prophet': prophet_pred,
            'arima': arima_pred,
            'ensemble': ensemble_pred
        }
        
        for model_name, pred in models.items():
            if len(pred) > 0 and len(actual) > 0:
                metrics = {
                    'mae': mean_absolute_error(actual, pred),
                    'rmse': np.sqrt(mean_squared_error(actual, pred)),
                    'mape': mean_absolute_percentage_error(actual, pred) * 100,
                    'r2': 1 - (np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
                }
                results['test_metrics'][model_name] = metrics
        
        # Store predictions for analysis
        results['predictions'] = {
            'actual': actual.tolist(),
            'prophet': prophet_pred.tolist() if len(prophet_pred) > 0 else [],
            'arima': arima_pred.tolist() if len(arima_pred) > 0 else [],
            'ensemble': ensemble_pred.tolist() if len(ensemble_pred) > 0 else []
        }
        
        return results
    
    def predict_prophet(self, future_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using Prophet model."""
        if self.prophet_model is None:
            logger.warning("Prophet model not trained")
            return np.array([])
        
        try:
            # Prepare future dataframe
            future_df = future_data[['ds']].copy()
            
            # Add external regressors
            for feature in self.external_features:
                if feature in future_data.columns:
                    future_df[feature] = future_data[feature]
                else:
                    # Use last known value or zero
                    future_df[feature] = 0
            
            forecast = self.prophet_model.predict(future_df)
            return forecast['yhat'].values
            
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            return np.array([])
    
    def predict_arima(self, future_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using ARIMA model."""
        if self.arima_model is None:
            logger.warning("ARIMA model not trained")
            return np.array([])
        
        try:
            # Forecast
            n_periods = len(future_data)
            forecast = self.arima_model.forecast(steps=n_periods)
            return forecast
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            return np.array([])
    
    def predict_ensemble(self, future_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble predictions with confidence intervals."""
        logger.info(f"Generating ensemble predictions for {len(future_data)} periods")
        
        # Individual predictions
        prophet_pred = self.predict_prophet(future_data)
        arima_pred = self.predict_arima(future_data)
        
        # Ensemble prediction
        if len(prophet_pred) > 0 and len(arima_pred) > 0:
            ensemble_pred = (
                prophet_pred * self.ensemble_weights['prophet'] + 
                arima_pred * self.ensemble_weights['arima']
            )
        elif len(prophet_pred) > 0:
            ensemble_pred = prophet_pred
            logger.warning("Using Prophet only for predictions")
        elif len(arima_pred) > 0:
            ensemble_pred = arima_pred
            logger.warning("Using ARIMA only for predictions")
        else:
            logger.error("No models available for prediction")
            return {}
        
        # Create results
        predictions = pd.DataFrame({
            'ds': future_data['ds'],
            'prophet_pred': prophet_pred if len(prophet_pred) == len(future_data) else np.nan,
            'arima_pred': arima_pred if len(arima_pred) == len(future_data) else np.nan,
            'ensemble_pred': ensemble_pred,
            'actual': future_data['y'] if 'y' in future_data.columns else np.nan
        })
        
        return {
            'predictions': predictions,
            'ensemble_weights': self.ensemble_weights,
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def _optimize_ensemble_weights(self, test_data: pd.DataFrame, 
                                 results: Dict[str, Any]) -> Dict[str, float]:
        """Optimize ensemble weights based on test performance."""
        if not results.get('test_metrics'):
            return self.ensemble_weights
        
        prophet_mape = results['test_metrics'].get('prophet', {}).get('mape', 100)
        arima_mape = results['test_metrics'].get('arima', {}).get('mape', 100)
        
        # Weight inversely proportional to error
        prophet_weight = 1 / (prophet_mape + 1)
        arima_weight = 1 / (arima_mape + 1)
        
        total_weight = prophet_weight + arima_weight
        
        optimized_weights = {
            'prophet': prophet_weight / total_weight,
            'arima': arima_weight / total_weight
        }
        
        logger.info(f"Optimized ensemble weights: {optimized_weights}")
        return optimized_weights
    
    def save_models(self) -> Dict[str, Path]:
        """Save trained models to disk."""
        saved_files = {}
        
        try:
            # Save Prophet model
            if self.prophet_model:
                prophet_file = self.model_dir / 'prophet_global.json'
                with open(prophet_file, 'w') as f:
                    json.dump(self.prophet_model.to_json(), f, indent=2)
                saved_files['prophet'] = prophet_file
            
            # Save ARIMA model
            if self.arima_model:
                arima_file = self.model_dir / 'arima_global.pkl'
                joblib.dump(self.arima_model, arima_file)
                saved_files['arima'] = arima_file
            
            # Save ensemble metadata
            metadata = {
                'ensemble_weights': self.ensemble_weights,
                'training_timestamp': datetime.now().isoformat(),
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'external_features': self.external_features
            }
            
            metadata_file = self.model_dir / 'global_model_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            saved_files['metadata'] = metadata_file
            
            logger.info(f"Global models saved to {len(saved_files)} files")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
        
        return saved_files
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            # Load metadata
            metadata_file = self.model_dir / 'global_model_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.ensemble_weights = metadata.get('ensemble_weights', self.ensemble_weights)
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.external_features = metadata.get('external_features', self.external_features)
            
            # Load Prophet model
            prophet_file = self.model_dir / 'prophet_global.json'
            if prophet_file.exists():
                with open(prophet_file, 'r') as f:
                    self.prophet_model = Prophet().from_json(json.load(f))
                logger.info("Prophet model loaded successfully")
            
            # Load ARIMA model
            arima_file = self.model_dir / 'arima_global.pkl'
            if arima_file.exists():
                self.arima_model = joblib.load(arima_file)
                logger.info("ARIMA model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


def main():
    """Main function for testing global model."""
    global_model = SeoulGlobalModel()
    
    try:
        # Load processed data
        processed_data_path = Path("data/processed/seoul_sales_combined.csv")
        if not processed_data_path.exists():
            logger.error("Processed sales data not found. Run preprocessing first.")
            return
        
        df = pd.read_csv(processed_data_path)
        logger.info(f"Loaded {len(df):,} records for global model training")
        
        # Train models
        results = global_model.train_global_models(df)
        
        print("\n=== GLOBAL MODEL TRAINING RESULTS ===")
        print(f"Training data shape: {len(global_model.training_data)}")
        print(f"Ensemble weights: {global_model.ensemble_weights}")
        
        if 'test_metrics' in results:
            print("\nTest Performance:")
            for model_name, metrics in results['test_metrics'].items():
                print(f"  {model_name.upper()}:")
                print(f"    MAE: {metrics['mae']:,.0f}")
                print(f"    RMSE: {metrics['rmse']:,.0f}")
                print(f"    MAPE: {metrics['mape']:.1f}%")
                print(f"    R²: {metrics['r2']:.3f}")
        
        print(f"\nModels saved to: {global_model.model_dir}")
        
    except Exception as e:
        logger.error(f"Global model training failed: {e}")
        raise


if __name__ == "__main__":
    main()