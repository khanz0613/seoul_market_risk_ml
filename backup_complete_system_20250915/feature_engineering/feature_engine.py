"""
Feature Engineering Engine for Seoul Market Risk ML System
Implements the 5-component Risk Score calculation system based on Altman Z-Score methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML and statistical libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from ..utils.config_loader import load_config, get_data_paths


logger = logging.getLogger(__name__)


class SeoulFeatureEngine:
    """
    Core Feature Engineering Engine for Risk Score Calculation
    
    Implements 5-component scoring system:
    - Revenue Change Rate: 30% weight (most critical)
    - Volatility Score: 20% weight 
    - Trend Analysis: 20% weight
    - Seasonal Deviation: 15% weight
    - Industry Comparison: 15% weight
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.risk_config = self.config['risk_scoring']
        self.weights = self.risk_config['weights']
        self.data_paths = get_data_paths(self.config)
        
        # Initialize scalers for normalization
        self.scalers = {
            'revenue_change': MinMaxScaler(feature_range=(0, 100)),
            'volatility': MinMaxScaler(feature_range=(0, 100)), 
            'trend': MinMaxScaler(feature_range=(0, 100)),
            'seasonal': MinMaxScaler(feature_range=(0, 100)),
            'industry': MinMaxScaler(feature_range=(0, 100))
        }
        
        logger.info("Seoul Feature Engine initialized with 5-component Risk Score system")
    
    def engineer_features(self, df: pd.DataFrame, save_intermediate: bool = True) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            df: Input DataFrame with sales data
            save_intermediate: Whether to save intermediate results
            
        Returns:
            DataFrame with engineered features and Risk Scores
        """
        logger.info(f"Starting feature engineering for {len(df):,} records")
        
        # Ensure proper data structure
        df_processed = self._prepare_data(df.copy())
        
        # Calculate each Risk Score component
        logger.info("Calculating Risk Score components...")
        
        # 1. Revenue Change Rate (30% weight)
        df_processed = self._calculate_revenue_change_score(df_processed)
        
        # 2. Volatility Score (20% weight)
        df_processed = self._calculate_volatility_score(df_processed)
        
        # 3. Trend Analysis Score (20% weight)
        df_processed = self._calculate_trend_score(df_processed)
        
        # 4. Seasonal Deviation Score (15% weight)
        df_processed = self._calculate_seasonal_deviation_score(df_processed)
        
        # 5. Industry Comparison Score (15% weight)
        df_processed = self._calculate_industry_comparison_score(df_processed)
        
        # Calculate final Risk Score
        df_processed = self._calculate_final_risk_score(df_processed)
        
        # Add feature metadata
        df_processed['feature_engineering_timestamp'] = datetime.now()
        df_processed['feature_version'] = '1.0'
        
        if save_intermediate:
            self._save_engineered_features(df_processed)
        
        logger.info(f"Feature engineering completed: {len(df_processed.columns)} total features")
        return df_processed
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for feature engineering."""
        logger.info("Preparing data for feature engineering...")
        
        # Ensure required columns exist
        required_columns = ['district_code', 'business_type_code', 'quarter_code', 'monthly_revenue']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort data for time series analysis
        df_sorted = df.sort_values(['district_code', 'business_type_code', 'quarter_code']).copy()
        
        # Create time series index
        df_sorted['year'] = df_sorted['quarter_code'].astype(str).str[:4].astype(int)
        df_sorted['quarter'] = df_sorted['quarter_code'].astype(str).str[4:].astype(int)
        
        # Create date for easier time series operations
        df_sorted['date'] = pd.to_datetime(
            df_sorted['year'].astype(str) + '-' + 
            ((df_sorted['quarter'] - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
        )
        
        # Handle missing or zero revenues
        df_sorted['monthly_revenue'] = df_sorted['monthly_revenue'].fillna(0)
        df_sorted['monthly_revenue'] = np.maximum(df_sorted['monthly_revenue'], 0.01)  # Avoid division by zero
        
        logger.info(f"Data prepared: {len(df_sorted)} records across {df_sorted['year'].nunique()} years")
        return df_sorted
    
    def _calculate_revenue_change_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Revenue Change Rate Score (30% weight).
        Measures actual vs predicted revenue performance.
        """
        logger.info("Calculating Revenue Change Rate scores...")
        
        results = []
        
        # Group by business and location for individual time series
        for (district, business_type), group in df.groupby(['district_code', 'business_type_code']):
            if len(group) < 3:  # Need minimum data points
                group = group.copy()
                group['revenue_change_score'] = 50.0  # Neutral score for insufficient data
                results.append(group)
                continue
            
            group = group.sort_values('date').copy()
            
            # Calculate quarter-over-quarter change
            group['revenue_pct_change'] = group['monthly_revenue'].pct_change() * 100
            
            # Calculate rolling averages for stability
            group['revenue_ma_4q'] = group['monthly_revenue'].rolling(window=4, min_periods=2).mean()
            group['revenue_ma_pct_change'] = group['revenue_ma_4q'].pct_change() * 100
            
            # Calculate recent vs historical performance
            if len(group) >= 8:
                recent_avg = group['monthly_revenue'].tail(4).mean()
                historical_avg = group['monthly_revenue'].head(-4).mean()
                recent_vs_historical = ((recent_avg - historical_avg) / historical_avg) * 100
            else:
                recent_vs_historical = group['revenue_pct_change'].mean()
            
            # Scoring logic: negative changes increase risk score
            revenue_change_raw = []
            for idx, row in group.iterrows():
                quarterly_change = row['revenue_pct_change'] if pd.notna(row['revenue_pct_change']) else 0
                ma_change = row['revenue_ma_pct_change'] if pd.notna(row['revenue_ma_pct_change']) else 0
                
                # Weighted combination of metrics
                combined_change = (quarterly_change * 0.6 + ma_change * 0.4)
                
                # Convert to risk score (negative change = higher risk)
                if combined_change <= -25:  # Severe decline
                    score = 90
                elif combined_change <= -15:  # Major decline
                    score = 75
                elif combined_change <= -5:   # Moderate decline
                    score = 60
                elif combined_change <= 5:    # Stable
                    score = 30
                elif combined_change <= 15:   # Good growth
                    score = 15
                else:  # Excellent growth
                    score = 5
                
                revenue_change_raw.append(score)
            
            group['revenue_change_score'] = revenue_change_raw
            group['recent_vs_historical_pct'] = recent_vs_historical
            results.append(group)
        
        # Combine all results
        df_with_revenue_scores = pd.concat(results, ignore_index=True)
        
        # Normalize scores to 0-100 range
        scores = df_with_revenue_scores['revenue_change_score'].values.reshape(-1, 1)
        self.scalers['revenue_change'].fit(scores)
        df_with_revenue_scores['revenue_change_score'] = self.scalers['revenue_change'].transform(scores).flatten()
        
        logger.info("Revenue Change Rate scores calculated")
        return df_with_revenue_scores
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volatility Score (20% weight).
        Measures revenue stability and predictability.
        """
        logger.info("Calculating Volatility scores...")
        
        results = []
        
        for (district, business_type), group in df.groupby(['district_code', 'business_type_code']):
            if len(group) < 4:
                group = group.copy()
                group['volatility_score'] = 50.0
                group['revenue_volatility'] = 0.0
                group['volatility_trend'] = 0.0
                results.append(group)
                continue
            
            group = group.sort_values('date').copy()
            
            # Calculate rolling standard deviation (volatility measure)
            group['revenue_std_4q'] = group['monthly_revenue'].rolling(window=4, min_periods=2).std()
            group['revenue_mean_4q'] = group['monthly_revenue'].rolling(window=4, min_periods=2).mean()
            
            # Coefficient of variation (normalized volatility)
            group['revenue_cv'] = (group['revenue_std_4q'] / group['revenue_mean_4q']) * 100
            
            # Calculate volatility trend (is volatility increasing?)
            group['volatility_change'] = group['revenue_cv'].pct_change() * 100
            
            # Overall revenue volatility for the series
            overall_cv = (group['monthly_revenue'].std() / group['monthly_revenue'].mean()) * 100
            
            # Score volatility (higher volatility = higher risk)
            volatility_scores = []
            for idx, row in group.iterrows():
                cv = row['revenue_cv'] if pd.notna(row['revenue_cv']) else overall_cv
                
                if cv >= 50:      # Very high volatility
                    score = 85
                elif cv >= 30:    # High volatility  
                    score = 70
                elif cv >= 20:    # Moderate volatility
                    score = 50
                elif cv >= 10:    # Low volatility
                    score = 25
                else:             # Very stable
                    score = 10
                
                volatility_scores.append(score)
            
            group['volatility_score'] = volatility_scores
            group['revenue_volatility'] = overall_cv
            group['volatility_trend'] = group['volatility_change'].mean()
            results.append(group)
        
        df_with_volatility = pd.concat(results, ignore_index=True)
        
        # Normalize volatility scores
        scores = df_with_volatility['volatility_score'].values.reshape(-1, 1)
        self.scalers['volatility'].fit(scores)
        df_with_volatility['volatility_score'] = self.scalers['volatility'].transform(scores).flatten()
        
        logger.info("Volatility scores calculated")
        return df_with_volatility
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Trend Analysis Score (20% weight).
        Measures long-term revenue trajectory.
        """
        logger.info("Calculating Trend Analysis scores...")
        
        results = []
        
        for (district, business_type), group in df.groupby(['district_code', 'business_type_code']):
            if len(group) < 4:
                group = group.copy()
                group['trend_score'] = 50.0
                group['trend_slope'] = 0.0
                group['trend_r_squared'] = 0.0
                results.append(group)
                continue
            
            group = group.sort_values('date').copy()
            
            # Linear trend analysis
            x = np.arange(len(group))
            y = group['monthly_revenue'].values
            
            # Calculate trend line
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                r_squared = r_value ** 2
            except:
                slope, r_squared = 0, 0
            
            # Trend strength and direction
            trend_strength = abs(slope) * len(group)  # Impact over time period
            trend_direction = 1 if slope > 0 else -1 if slope < 0 else 0
            
            # Calculate trend scores
            trend_scores = []
            for idx, row in group.iterrows():
                # Strong negative trend = high risk
                if trend_direction == -1 and r_squared > 0.5:
                    if trend_strength > group['monthly_revenue'].mean() * 0.1:
                        score = 80  # Strong declining trend
                    else:
                        score = 60  # Weak declining trend
                elif trend_direction == 1 and r_squared > 0.5:
                    if trend_strength > group['monthly_revenue'].mean() * 0.1:
                        score = 15  # Strong positive trend
                    else:
                        score = 30  # Weak positive trend
                else:
                    score = 45  # No clear trend
                
                trend_scores.append(score)
            
            group['trend_score'] = trend_scores
            group['trend_slope'] = slope
            group['trend_r_squared'] = r_squared
            results.append(group)
        
        df_with_trends = pd.concat(results, ignore_index=True)
        
        # Normalize trend scores
        scores = df_with_trends['trend_score'].values.reshape(-1, 1)
        self.scalers['trend'].fit(scores)
        df_with_trends['trend_score'] = self.scalers['trend'].transform(scores).flatten()
        
        logger.info("Trend Analysis scores calculated")
        return df_with_trends
    
    def _calculate_seasonal_deviation_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Seasonal Deviation Score (15% weight).
        Measures deviation from expected seasonal patterns.
        """
        logger.info("Calculating Seasonal Deviation scores...")
        
        results = []
        
        for (district, business_type), group in df.groupby(['district_code', 'business_type_code']):
            if len(group) < 8:  # Need at least 2 years for seasonal analysis
                group = group.copy()
                group['seasonal_deviation_score'] = 50.0
                group['seasonal_component'] = 0.0
                group['seasonal_deviation'] = 0.0
                results.append(group)
                continue
            
            group = group.sort_values('date').copy()
            
            try:
                # Seasonal decomposition
                ts = group.set_index('date')['monthly_revenue']
                decomposition = seasonal_decompose(ts, model='additive', period=4)  # Quarterly seasonality
                
                seasonal_component = decomposition.seasonal
                residual_component = decomposition.resid
                
                # Calculate expected vs actual for current periods
                group['seasonal_component'] = seasonal_component.values
                group['seasonal_residual'] = residual_component.values
                
                # Seasonal deviation score
                seasonal_std = seasonal_component.std()
                residual_std = residual_component.std()
                
                seasonal_scores = []
                for idx, row in group.iterrows():
                    residual = abs(row['seasonal_residual']) if pd.notna(row['seasonal_residual']) else 0
                    
                    if residual_std > 0:
                        deviation_magnitude = residual / residual_std
                        
                        if deviation_magnitude > 2.5:      # Very high deviation
                            score = 80
                        elif deviation_magnitude > 1.5:    # High deviation
                            score = 60
                        elif deviation_magnitude > 1.0:    # Moderate deviation
                            score = 40
                        else:                              # Normal seasonal behavior
                            score = 20
                    else:
                        score = 25  # No clear seasonal pattern
                    
                    seasonal_scores.append(score)
                
                group['seasonal_deviation_score'] = seasonal_scores
                group['seasonal_deviation'] = group['seasonal_residual'].abs().mean()
                
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed for {district}-{business_type}: {e}")
                group['seasonal_deviation_score'] = 50.0
                group['seasonal_component'] = 0.0
                group['seasonal_deviation'] = 0.0
            
            results.append(group)
        
        df_with_seasonal = pd.concat(results, ignore_index=True)
        
        # Normalize seasonal scores
        scores = df_with_seasonal['seasonal_deviation_score'].values.reshape(-1, 1)
        self.scalers['seasonal'].fit(scores)
        df_with_seasonal['seasonal_deviation_score'] = self.scalers['seasonal'].transform(scores).flatten()
        
        logger.info("Seasonal Deviation scores calculated")
        return df_with_seasonal
    
    def _calculate_industry_comparison_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Industry Comparison Score (15% weight).
        Measures performance relative to same business type.
        """
        logger.info("Calculating Industry Comparison scores...")
        
        # Calculate industry benchmarks by business type and quarter
        industry_benchmarks = df.groupby(['business_type_code', 'quarter_code']).agg({
            'monthly_revenue': ['mean', 'median', 'std']
        }).reset_index()
        
        # Flatten column names
        industry_benchmarks.columns = ['business_type_code', 'quarter_code', 
                                     'industry_mean', 'industry_median', 'industry_std']
        
        # Merge benchmarks back to main dataset
        df_with_benchmarks = df.merge(industry_benchmarks, on=['business_type_code', 'quarter_code'], how='left')
        
        # Calculate industry comparison scores
        industry_scores = []
        industry_percentiles = []
        
        for idx, row in df_with_benchmarks.iterrows():
            revenue = row['monthly_revenue']
            industry_mean = row['industry_mean']
            industry_std = row['industry_std']
            
            if pd.notna(industry_mean) and pd.notna(industry_std) and industry_std > 0:
                # Calculate z-score relative to industry
                z_score = (revenue - industry_mean) / industry_std
                
                # Convert z-score to percentile
                percentile = stats.norm.cdf(z_score) * 100
                
                # Risk score based on industry position (lower percentile = higher risk)
                if percentile <= 10:        # Bottom 10%
                    score = 85
                elif percentile <= 25:      # Bottom quartile
                    score = 65
                elif percentile <= 50:      # Below median
                    score = 45
                elif percentile <= 75:      # Above median
                    score = 25
                else:                       # Top quartile
                    score = 15
            else:
                score = 50  # No comparison data available
                percentile = 50
            
            industry_scores.append(score)
            industry_percentiles.append(percentile)
        
        df_with_benchmarks['industry_comparison_score'] = industry_scores
        df_with_benchmarks['industry_percentile'] = industry_percentiles
        
        # Calculate relative performance metrics
        df_with_benchmarks['performance_vs_industry'] = (
            (df_with_benchmarks['monthly_revenue'] - df_with_benchmarks['industry_mean']) / 
            df_with_benchmarks['industry_mean'] * 100
        ).fillna(0)
        
        # Normalize industry comparison scores
        scores = np.array(industry_scores).reshape(-1, 1)
        self.scalers['industry'].fit(scores)
        df_with_benchmarks['industry_comparison_score'] = self.scalers['industry'].transform(scores).flatten()
        
        logger.info("Industry Comparison scores calculated")
        return df_with_benchmarks
    
    def _calculate_final_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final Risk Score using weighted combination of all components.
        """
        logger.info("Calculating final Risk Scores...")
        
        # Apply weights from configuration
        df['final_risk_score'] = (
            df['revenue_change_score'] * self.weights['revenue_change'] +
            df['volatility_score'] * self.weights['volatility'] +
            df['trend_score'] * self.weights['trend'] + 
            df['seasonal_deviation_score'] * self.weights['seasonal_deviation'] +
            df['industry_comparison_score'] * self.weights['industry_comparison']
        )
        
        # Ensure scores are within 0-100 range
        df['final_risk_score'] = np.clip(df['final_risk_score'], 0, 100)
        
        # Assign risk levels based on score
        df['risk_level'] = df['final_risk_score'].apply(self._get_risk_level)
        df['risk_label'] = df['final_risk_score'].apply(self._get_risk_label)
        
        # Add component contributions for explainability
        df['revenue_change_contribution'] = df['revenue_change_score'] * self.weights['revenue_change']
        df['volatility_contribution'] = df['volatility_score'] * self.weights['volatility']
        df['trend_contribution'] = df['trend_score'] * self.weights['trend']
        df['seasonal_contribution'] = df['seasonal_deviation_score'] * self.weights['seasonal_deviation']
        df['industry_contribution'] = df['industry_comparison_score'] * self.weights['industry_comparison']
        
        # Summary statistics
        risk_distribution = df['risk_level'].value_counts().sort_index()
        logger.info(f"Final Risk Score distribution: {dict(risk_distribution)}")
        
        return df
    
    def _get_risk_level(self, score: float) -> int:
        """Convert risk score to level (1-5)."""
        if score <= 20:
            return 1  # 안전
        elif score <= 40:
            return 2  # 주의
        elif score <= 60:
            return 3  # 경계
        elif score <= 80:
            return 4  # 위험
        else:
            return 5  # 매우위험
    
    def _get_risk_label(self, score: float) -> str:
        """Convert risk score to Korean label."""
        level = self._get_risk_level(score)
        labels = {1: "안전", 2: "주의", 3: "경계", 4: "위험", 5: "매우위험"}
        return labels[level]
    
    def _save_engineered_features(self, df: pd.DataFrame) -> None:
        """Save engineered features to file."""
        output_path = self.data_paths['processed'] / 'features_engineered.csv'
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Engineered features saved to {output_path}")
        
        # Save feature summary
        feature_summary = {
            'total_records': len(df),
            'risk_score_distribution': df['risk_level'].value_counts().to_dict(),
            'average_risk_score': float(df['final_risk_score'].mean()),
            'feature_weights': self.weights,
            'generated_timestamp': datetime.now().isoformat()
        }
        
        import json
        summary_path = self.data_paths['processed'] / 'feature_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(feature_summary, f, indent=2, ensure_ascii=False)


def main():
    """Main function for testing feature engineering."""
    engine = SeoulFeatureEngine()
    
    try:
        # Load processed data
        processed_data_path = Path("data/processed/seoul_sales_combined.csv")
        if not processed_data_path.exists():
            logger.error("Processed data not found. Run preprocessing first.")
            return
        
        df = pd.read_csv(processed_data_path)
        logger.info(f"Loaded {len(df):,} records for feature engineering")
        
        # Run feature engineering
        df_featured = engine.engineer_features(df)
        
        print("\n=== FEATURE ENGINEERING RESULTS ===")
        print(f"Total records: {len(df_featured):,}")
        print(f"Total features: {len(df_featured.columns)}")
        print(f"Average Risk Score: {df_featured['final_risk_score'].mean():.1f}")
        print("\nRisk Level Distribution:")
        for level, count in df_featured['risk_level'].value_counts().sort_index().items():
            label = df_featured[df_featured['risk_level']==level]['risk_label'].iloc[0]
            print(f"  Level {level} ({label}): {count:,} records ({count/len(df_featured)*100:.1f}%)")
        
        print("\nComponent Score Averages:")
        print(f"  Revenue Change: {df_featured['revenue_change_score'].mean():.1f}")
        print(f"  Volatility: {df_featured['volatility_score'].mean():.1f}")
        print(f"  Trend: {df_featured['trend_score'].mean():.1f}")
        print(f"  Seasonal Deviation: {df_featured['seasonal_deviation_score'].mean():.1f}")
        print(f"  Industry Comparison: {df_featured['industry_comparison_score'].mean():.1f}")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()