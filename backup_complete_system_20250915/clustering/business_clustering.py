"""
Business Category Clustering Module for Seoul Market Risk ML System
Groups business types into 12-15 categories based on revenue pattern similarity using DTW.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Time series clustering libraries
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

# ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from ..utils.config_loader import load_config, get_data_paths


logger = logging.getLogger(__name__)


class SeoulBusinessClusterer:
    """
    Seoul Business Category Clustering System
    
    Uses Dynamic Time Warping (DTW) to cluster business types based on:
    1. Revenue pattern similarity over time
    2. Seasonal behavior patterns
    3. Weekly/daily transaction patterns
    4. Business lifecycle characteristics
    
    Groups similar businesses into 12-15 major categories (e.g., 한식/일식/양식 → 외식업)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.clustering_config = self.config['clustering']['business']
        self.data_paths = get_data_paths(self.config)
        
        # Clustering parameters
        self.n_categories = self.clustering_config.get('n_categories', 12)
        self.distance_metric = self.clustering_config.get('distance_metric', 'dtw')
        self.algorithm = self.clustering_config.get('algorithm', 'dtw')
        
        # Initialize components
        self.ts_scaler = TimeSeriesScalerMeanVariance()
        self.cluster_model = None
        self.business_profiles = None
        self.category_mapping = None
        
        logger.info(f"Seoul Business Clusterer initialized for {self.n_categories} categories using {self.distance_metric}")
    
    def create_business_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive time series profiles for each business type.
        
        Args:
            df: Sales data with business type information
            
        Returns:
            DataFrame with business profiles and time series features
        """
        logger.info("Creating business type profiles for DTW clustering...")
        
        # Get business type statistics
        business_stats = df.groupby('business_type_code').agg({
            'monthly_revenue': ['mean', 'median', 'std', 'count', 'sum'],
            'monthly_transactions': ['mean', 'median', 'sum'],
            'district_code': 'nunique',  # Number of districts with this business
            'quarter_code': ['min', 'max', 'nunique']
        }).round(2)
        
        # Flatten column names
        business_stats.columns = [f'{col[0]}_{col[1]}' for col in business_stats.columns]
        business_stats = business_stats.reset_index()
        
        # Get business type names
        business_names = df.groupby('business_type_code')['business_type_name'].first().reset_index()
        business_stats = business_stats.merge(business_names, on='business_type_code', how='left')
        
        # Create time series features for each business type
        logger.info("Extracting time series patterns...")
        
        time_series_data = []
        business_ts_profiles = []
        
        for business_code in business_stats['business_type_code'].unique():
            business_data = df[df['business_type_code'] == business_code].copy()
            
            if len(business_data) < 4:  # Need minimum data points
                continue
            
            # Sort by time
            business_data = business_data.sort_values(['district_code', 'quarter_code'])
            
            # Create quarterly time series aggregated across all districts
            quarterly_series = business_data.groupby('quarter_code').agg({
                'monthly_revenue': 'mean',
                'monthly_transactions': 'mean'
            }).reset_index()
            
            if len(quarterly_series) < 4:  # Need at least 1 year of data
                continue
            
            # Extract time series features
            revenue_series = quarterly_series['monthly_revenue'].values
            transaction_series = quarterly_series['monthly_transactions'].values
            
            # Calculate time series characteristics
            ts_profile = self._extract_time_series_features(
                revenue_series, transaction_series, business_code
            )
            
            business_ts_profiles.append(ts_profile)
            time_series_data.append(revenue_series)
        
        # Convert to DataFrame
        ts_profiles_df = pd.DataFrame(business_ts_profiles)
        
        # Merge with business stats
        business_profiles = business_stats.merge(
            ts_profiles_df, on='business_type_code', how='inner'
        )
        
        # Add derived clustering features
        business_profiles = self._add_clustering_features(business_profiles)
        
        # Store time series data for DTW clustering
        self.time_series_data = np.array([
            self._normalize_time_series(ts) for ts in time_series_data
        ])
        
        self.business_profiles = business_profiles
        logger.info(f"Created profiles for {len(business_profiles)} business types")
        
        return business_profiles
    
    def _extract_time_series_features(self, revenue_series: np.ndarray, 
                                    transaction_series: np.ndarray, 
                                    business_code: str) -> Dict[str, Any]:
        """Extract comprehensive time series features from revenue patterns."""
        
        try:
            features = {
                'business_type_code': business_code,
                'ts_length': len(revenue_series),
                'revenue_mean': float(np.mean(revenue_series)),
                'revenue_std': float(np.std(revenue_series)),
                'revenue_cv': float(np.std(revenue_series) / (np.mean(revenue_series) + 0.01)),
                'transaction_mean': float(np.mean(transaction_series)),
                'transaction_std': float(np.std(transaction_series))
            }
            
            # Trend analysis
            if len(revenue_series) > 1:
                x = np.arange(len(revenue_series))
                trend_coef = np.polyfit(x, revenue_series, 1)[0]
                features['trend_slope'] = float(trend_coef)
                features['trend_strength'] = float(abs(trend_coef) * len(revenue_series) / np.mean(revenue_series))
            else:
                features['trend_slope'] = 0.0
                features['trend_strength'] = 0.0
            
            # Seasonality detection (quarterly patterns)
            if len(revenue_series) >= 8:  # At least 2 years
                quarterly_means = self._calculate_seasonal_pattern(revenue_series)
                features['seasonal_strength'] = float(np.std(quarterly_means) / np.mean(quarterly_means))
                features['peak_quarter'] = int(np.argmax(quarterly_means) + 1)
                features['trough_quarter'] = int(np.argmin(quarterly_means) + 1)
            else:
                features['seasonal_strength'] = 0.0
                features['peak_quarter'] = 1
                features['trough_quarter'] = 1
            
            # Volatility characteristics
            if len(revenue_series) > 2:
                pct_changes = np.diff(revenue_series) / (revenue_series[:-1] + 0.01)
                features['volatility_mean'] = float(np.mean(np.abs(pct_changes)))
                features['volatility_max'] = float(np.max(np.abs(pct_changes)))
                features['negative_growth_ratio'] = float(np.sum(pct_changes < 0) / len(pct_changes))
            else:
                features['volatility_mean'] = 0.0
                features['volatility_max'] = 0.0
                features['negative_growth_ratio'] = 0.0
            
            # Business lifecycle stage (growth pattern)
            if len(revenue_series) >= 4:
                early_mean = np.mean(revenue_series[:len(revenue_series)//2])
                late_mean = np.mean(revenue_series[len(revenue_series)//2:])
                features['lifecycle_growth'] = float((late_mean - early_mean) / early_mean)
                
                # Stability measure (consistency over time)
                rolling_std = []
                window = min(4, len(revenue_series)//2)
                for i in range(window, len(revenue_series)):
                    rolling_std.append(np.std(revenue_series[i-window:i]))
                features['stability_score'] = float(1.0 / (np.mean(rolling_std) + 0.01))
            else:
                features['lifecycle_growth'] = 0.0
                features['stability_score'] = 1.0
                
        except Exception as e:
            logger.warning(f"Error extracting features for business {business_code}: {e}")
            # Return default values
            features = {
                'business_type_code': business_code,
                'ts_length': len(revenue_series),
                'revenue_mean': float(np.mean(revenue_series)) if len(revenue_series) > 0 else 0.0,
                'revenue_std': 0.0, 'revenue_cv': 0.0, 'transaction_mean': 0.0, 'transaction_std': 0.0,
                'trend_slope': 0.0, 'trend_strength': 0.0, 'seasonal_strength': 0.0,
                'peak_quarter': 1, 'trough_quarter': 1, 'volatility_mean': 0.0,
                'volatility_max': 0.0, 'negative_growth_ratio': 0.0,
                'lifecycle_growth': 0.0, 'stability_score': 1.0
            }
        
        return features
    
    def _calculate_seasonal_pattern(self, series: np.ndarray) -> np.ndarray:
        """Calculate quarterly seasonal pattern."""
        # Group by quarters (assuming quarterly data)
        n_years = len(series) // 4
        if n_years < 2:
            return np.ones(4) * np.mean(series)
        
        # Reshape to years x quarters
        reshaped = series[:n_years*4].reshape(n_years, 4)
        quarterly_means = np.mean(reshaped, axis=0)
        
        return quarterly_means
    
    def _normalize_time_series(self, series: np.ndarray, target_length: int = 20) -> np.ndarray:
        """Normalize time series to fixed length for DTW clustering."""
        if len(series) == 0:
            return np.zeros(target_length)
        
        # Interpolate to target length
        from scipy.interpolate import interp1d
        
        try:
            if len(series) == 1:
                return np.full(target_length, series[0])
            
            x_old = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, target_length)
            
            interpolator = interp1d(x_old, series, kind='linear')
            normalized = interpolator(x_new)
            
            # Standardize to zero mean and unit variance
            if np.std(normalized) > 0:
                normalized = (normalized - np.mean(normalized)) / np.std(normalized)
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Error normalizing time series: {e}")
            return np.zeros(target_length)
    
    def _add_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional features for business clustering."""
        logger.info("Adding derived clustering features...")
        
        # Market penetration (how widespread is this business type)
        total_districts = df['district_code_nunique'].max()
        df['market_penetration'] = df['district_code_nunique'] / total_districts
        
        # Business scale categories
        df['business_scale'] = pd.cut(
            df['monthly_revenue_mean'],
            bins=[0, df['monthly_revenue_mean'].quantile(0.33), 
                  df['monthly_revenue_mean'].quantile(0.67), float('inf')],
            labels=['소규모', '중규모', '대규모']
        )
        
        # Revenue efficiency (revenue per transaction)
        df['revenue_efficiency'] = (
            df['monthly_revenue_sum'] / (df['monthly_transactions_sum'] + 0.01)
        )
        
        # Market concentration (inverse of district spread)
        df['market_concentration'] = 1.0 / (df['district_code_nunique'] + 1)
        
        # Data coverage quality
        df['data_quality'] = (
            df['quarter_code_nunique'] / 
            (df['quarter_code_max'].astype(str).str[:4].astype(int) - 
             df['quarter_code_min'].astype(str).str[:4].astype(int) + 1) / 4
        ).fillna(1.0)
        
        return df
    
    def perform_dtw_clustering(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform DTW-based time series clustering on business types.
        
        Args:
            n_clusters: Number of clusters (if None, auto-determined)
            
        Returns:
            Dictionary with clustering results and metadata
        """
        logger.info("Performing DTW-based business clustering...")
        
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters()
        
        if self.time_series_data is None or len(self.time_series_data) == 0:
            raise ValueError("No time series data available. Run create_business_profiles first.")
        
        # Prepare time series data for tslearn
        ts_dataset = to_time_series_dataset(self.time_series_data)
        
        # Perform DTW-based clustering
        self.cluster_model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            max_iter=50,
            random_state=42,
            n_init=5
        )
        
        try:
            cluster_labels = self.cluster_model.fit_predict(ts_dataset)
        except Exception as e:
            logger.warning(f"DTW clustering failed, falling back to euclidean: {e}")
            # Fallback to euclidean if DTW fails
            self.cluster_model = TimeSeriesKMeans(
                n_clusters=n_clusters,
                metric="euclidean",
                max_iter=50,
                random_state=42,
                n_init=5
            )
            cluster_labels = self.cluster_model.fit_predict(ts_dataset)
        
        # Add cluster assignments to business profiles
        df_clustered = self.business_profiles.copy()
        df_clustered['category_id'] = cluster_labels
        df_clustered['category_name'] = df_clustered['category_id'].map(
            self._get_category_names(n_clusters)
        )
        
        # Calculate clustering quality metrics
        try:
            # Convert time series to 2D for silhouette score
            ts_flat = ts_dataset.reshape(len(ts_dataset), -1)
            silhouette_avg = silhouette_score(ts_flat, cluster_labels)
        except:
            silhouette_avg = 0.0
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_business_clusters(df_clustered)
        
        # Create category mapping
        self.category_mapping = self._create_category_mapping(df_clustered)
        
        results = {
            'clustered_businesses': df_clustered,
            'cluster_analysis': cluster_analysis,
            'category_mapping': self.category_mapping,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_centers': self.cluster_model.cluster_centers_ if hasattr(self.cluster_model, 'cluster_centers_') else None,
            'model_params': self.cluster_model.get_params()
        }
        
        logger.info(f"Business clustering completed: {n_clusters} categories, silhouette score: {silhouette_avg:.3f}")
        return results
    
    def determine_optimal_clusters(self, max_clusters: int = 20) -> int:
        """Determine optimal number of clusters for business categorization."""
        logger.info("Determining optimal number of business categories...")
        
        if self.time_series_data is None or len(self.time_series_data) < 4:
            logger.warning("Insufficient data for optimal cluster determination, using default")
            return min(self.n_categories, len(self.time_series_data) if self.time_series_data is not None else 12)
        
        ts_dataset = to_time_series_dataset(self.time_series_data)
        
        # Test different cluster numbers
        max_test_clusters = min(max_clusters, len(self.time_series_data) // 2)
        cluster_range = range(2, max_test_clusters + 1)
        
        inertias = []
        
        for n_clusters in cluster_range:
            try:
                kmeans = TimeSeriesKMeans(
                    n_clusters=n_clusters, 
                    metric="dtw", 
                    max_iter=20,
                    random_state=42, 
                    n_init=3
                )
                kmeans.fit(ts_dataset)
                inertias.append(kmeans.inertia_)
            except:
                # Fallback to euclidean if DTW fails
                kmeans = TimeSeriesKMeans(
                    n_clusters=n_clusters, 
                    metric="euclidean", 
                    max_iter=20,
                    random_state=42, 
                    n_init=3
                )
                kmeans.fit(ts_dataset)
                inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) > 2:
            optimal_clusters = self._find_elbow_point(list(cluster_range), inertias)
        else:
            optimal_clusters = cluster_range[0]
        
        # Ensure within target range (12-15 as per template)
        optimal_clusters = max(12, min(15, optimal_clusters))
        
        logger.info(f"Selected {optimal_clusters} clusters for business categorization")
        return optimal_clusters
    
    def _find_elbow_point(self, x_values: List[int], y_values: List[float]) -> int:
        """Find elbow point using rate of change analysis."""
        if len(x_values) < 3:
            return x_values[-1]
        
        # Calculate second derivative to find elbow
        differences = np.diff(y_values)
        differences2 = np.diff(differences)
        
        if len(differences2) > 0:
            # Find maximum curvature point
            elbow_idx = np.argmax(differences2) + 2
            if elbow_idx < len(x_values):
                return x_values[elbow_idx]
        
        # Fallback: find where improvement drops below threshold
        for i in range(1, len(differences)):
            improvement_ratio = abs(differences[i]) / abs(differences[i-1])
            if improvement_ratio < 0.2:  # Less than 20% improvement
                return x_values[i + 1]
        
        return x_values[len(x_values) // 2]
    
    def _get_category_names(self, n_clusters: int) -> Dict[int, str]:
        """Generate descriptive category names for business clusters."""
        # Predefined business category names based on common Korean business types
        category_names = {
            0: "외식업_한식", 1: "외식업_양식", 2: "외식업_일식_중식", 3: "카페_디저트",
            4: "편의점_마트", 5: "패션_의류", 6: "뷰티_미용", 7: "의료_약국",
            8: "교육_학원", 9: "부동산_금융", 10: "운동_헬스", 11: "문화_오락",
            12: "전자_통신", 13: "생활_서비스", 14: "기타_전문업"
        }
        
        return {i: category_names.get(i, f"비즈니스_카테고리_{i+1}") for i in range(n_clusters)}
    
    def _analyze_business_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze business cluster characteristics."""
        logger.info("Analyzing business cluster characteristics...")
        
        cluster_summary = df.groupby('category_id').agg({
            'business_type_code': 'count',
            'revenue_mean': ['mean', 'std'],
            'market_penetration': ['mean', 'std'],
            'seasonal_strength': 'mean',
            'volatility_mean': 'mean',
            'trend_strength': 'mean',
            'stability_score': 'mean'
        }).round(3)
        
        # Flatten column names
        cluster_summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in cluster_summary.columns]
        cluster_summary = cluster_summary.reset_index()
        
        # Add category names
        category_names = self._get_category_names(df['category_id'].nunique())
        cluster_summary['category_name'] = cluster_summary['category_id'].map(category_names)
        
        # Detailed characteristics for each cluster
        cluster_characteristics = {}
        for category_id in df['category_id'].unique():
            category_data = df[df['category_id'] == category_id]
            
            characteristics = {
                'size': len(category_data),
                'business_types': category_data['business_type_name'].tolist(),
                'avg_revenue': float(category_data['revenue_mean'].mean()),
                'avg_market_penetration': float(category_data['market_penetration'].mean()),
                'seasonality_strength': float(category_data['seasonal_strength'].mean()),
                'volatility_level': float(category_data['volatility_mean'].mean()),
                'growth_trend': float(category_data['trend_strength'].mean()),
                'stability': float(category_data['stability_score'].mean()),
                'profile': self._describe_business_category_profile(category_data)
            }
            
            cluster_characteristics[int(category_id)] = characteristics
        
        return {
            'summary_table': cluster_summary,
            'detailed_characteristics': cluster_characteristics,
            'total_business_types': len(df),
            'clustering_features_importance': self._calculate_business_feature_importance(df)
        }
    
    def _describe_business_category_profile(self, category_data: pd.DataFrame) -> str:
        """Generate descriptive profile for a business category."""
        
        # Analyze key characteristics
        avg_revenue = category_data['revenue_mean'].mean()
        avg_seasonality = category_data['seasonal_strength'].mean()
        avg_volatility = category_data['volatility_mean'].mean()
        avg_stability = category_data['stability_score'].mean()
        
        profile_parts = []
        
        # Revenue level
        if avg_revenue > category_data['revenue_mean'].quantile(0.75):
            profile_parts.append("고매출업종")
        elif avg_revenue > category_data['revenue_mean'].quantile(0.25):
            profile_parts.append("중간매출업종")
        else:
            profile_parts.append("저매출업종")
        
        # Seasonality
        if avg_seasonality > 0.3:
            profile_parts.append("계절성강함")
        elif avg_seasonality > 0.1:
            profile_parts.append("계절성보통")
        else:
            profile_parts.append("계절성약함")
        
        # Volatility
        if avg_volatility > 0.2:
            profile_parts.append("변동성높음")
        elif avg_volatility > 0.1:
            profile_parts.append("변동성보통")
        else:
            profile_parts.append("안정적")
        
        # Stability
        if avg_stability > category_data['stability_score'].median():
            profile_parts.append("예측가능")
        else:
            profile_parts.append("불확실성있음")
        
        return ", ".join(profile_parts)
    
    def _calculate_business_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance for business clustering."""
        features = ['revenue_mean', 'seasonal_strength', 'volatility_mean', 'trend_strength', 'stability_score']
        
        feature_importance = {}
        for feature in features:
            if feature in df.columns:
                cluster_means = df.groupby('category_id')[feature].mean()
                variance = cluster_means.var()
                feature_importance[feature] = float(variance)
        
        # Normalize to percentages
        total_variance = sum(feature_importance.values())
        if total_variance > 0:
            feature_importance = {k: v/total_variance*100 for k, v in feature_importance.items()}
        
        return feature_importance
    
    def _create_category_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create mapping from business type codes to category names."""
        mapping = {}
        for _, row in df.iterrows():
            mapping[row['business_type_code']] = row['category_name']
        
        return mapping
    
    def save_business_clustering_results(self, results: Dict[str, Any], 
                                       output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save business clustering results to files."""
        if output_dir is None:
            output_dir = self.data_paths['processed']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save clustered businesses
        businesses_file = output_dir / 'business_categories.csv'
        results['clustered_businesses'].to_csv(businesses_file, index=False, encoding='utf-8')
        saved_files['businesses'] = businesses_file
        
        # Save category mapping
        mapping_file = output_dir / 'business_category_mapping.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(results['category_mapping'], f, indent=2, ensure_ascii=False)
        saved_files['mapping'] = mapping_file
        
        # Save analysis results
        analysis_file = output_dir / 'business_cluster_analysis.json'
        analysis_data = {
            'metadata': {
                'n_clusters': results['n_clusters'],
                'silhouette_score': results['silhouette_score'],
                'clustering_timestamp': datetime.now().isoformat(),
                'algorithm': 'dtw_kmeans'
            },
            'cluster_analysis': {
                'summary_table': results['cluster_analysis']['summary_table'].to_dict('records'),
                'detailed_characteristics': results['cluster_analysis']['detailed_characteristics'],
                'feature_importance': results['cluster_analysis']['clustering_features_importance']
            }
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        saved_files['analysis'] = analysis_file
        
        logger.info(f"Business clustering results saved to {len(saved_files)} files")
        return saved_files


def main():
    """Main function for testing business clustering."""
    clusterer = SeoulBusinessClusterer()
    
    try:
        # Load processed sales data
        processed_data_path = Path("data/processed/seoul_sales_combined.csv")
        if not processed_data_path.exists():
            logger.error("Processed sales data not found. Run preprocessing first.")
            return
        
        df = pd.read_csv(processed_data_path)
        logger.info(f"Loaded {len(df):,} sales records for business clustering")
        
        # Create business profiles
        business_profiles = clusterer.create_business_profiles(df)
        
        # Perform DTW clustering
        clustering_results = clusterer.perform_dtw_clustering()
        
        # Save results
        saved_files = clusterer.save_business_clustering_results(clustering_results)
        
        print("\n=== BUSINESS CLUSTERING RESULTS ===")
        print(f"Total business types: {len(business_profiles)}")
        print(f"Number of categories: {clustering_results['n_clusters']}")
        print(f"Silhouette score: {clustering_results['silhouette_score']:.3f}")
        
        print("\nBusiness Category Distribution:")
        category_dist = clustering_results['clustered_businesses']['category_id'].value_counts().sort_index()
        for category_id, count in category_dist.items():
            category_name = clustering_results['clustered_businesses'][
                clustering_results['clustered_businesses']['category_id']==category_id
            ]['category_name'].iloc[0]
            print(f"  Category {category_id} ({category_name}): {count} business types")
        
        print(f"\nFiles saved: {list(saved_files.values())}")
        
    except Exception as e:
        logger.error(f"Business clustering failed: {e}")
        raise


if __name__ == "__main__":
    main()