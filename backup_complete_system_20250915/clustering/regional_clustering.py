"""
Regional Clustering Module for Seoul Market Risk ML System
Groups Seoul administrative districts into 6-8 clusters based on socio-economic characteristics.
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

# ML libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.config_loader import load_config, get_data_paths


logger = logging.getLogger(__name__)


class SeoulRegionalClusterer:
    """
    Seoul Regional Clustering System
    
    Groups administrative districts (행정동) into homogeneous clusters based on:
    1. Income Level (소득수준) - Average revenue per business
    2. Foot Traffic (유동인구) - Business density and transaction volumes
    3. Business Diversity (업종다양성) - Number and variety of business types
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.clustering_config = self.config['clustering']['regional']
        self.data_paths = get_data_paths(self.config)
        
        # Clustering parameters
        self.n_clusters = self.clustering_config.get('n_clusters', 6)
        self.algorithm = self.clustering_config.get('algorithm', 'kmeans')
        self.features = self.clustering_config.get('features', ['income_level', 'foot_traffic', 'business_diversity'])
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.district_profiles = None
        self.cluster_centers = None
        
        logger.info(f"Seoul Regional Clusterer initialized for {self.n_clusters} clusters")
    
    def create_district_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive profiles for each administrative district.
        
        Args:
            df: Sales data with district information
            
        Returns:
            DataFrame with district profiles and clustering features
        """
        logger.info("Creating district profiles for clustering...")
        
        # Group by district to create profiles
        district_stats = df.groupby('district_code').agg({
            'monthly_revenue': ['mean', 'median', 'std', 'count', 'sum'],
            'monthly_transactions': ['mean', 'median', 'sum'],
            'business_type_code': ['nunique'],
            'quarter_code': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        district_stats.columns = [f'{col[0]}_{col[1]}' for col in district_stats.columns]
        district_stats = district_stats.reset_index()
        
        # Get district names
        district_names = df.groupby('district_code')['district_name'].first().reset_index()
        district_stats = district_stats.merge(district_names, on='district_code', how='left')
        
        # Calculate clustering features
        logger.info("Calculating clustering features...")
        
        # 1. Income Level (소득수준) - Revenue per business indicator
        district_stats['income_level'] = district_stats['monthly_revenue_mean']
        
        # 2. Foot Traffic (유동인구) - Business activity intensity
        district_stats['foot_traffic'] = (
            district_stats['monthly_transactions_sum'] / 
            district_stats['monthly_revenue_count']  # transactions per business
        ).fillna(0)
        
        # 3. Business Diversity (업종다양성) - Variety of business types
        district_stats['business_diversity'] = district_stats['business_type_code_nunique']
        
        # Additional derived features for better clustering
        district_stats['revenue_stability'] = (
            district_stats['monthly_revenue_mean'] / 
            (district_stats['monthly_revenue_std'] + 0.01)  # Avoid division by zero
        )
        
        district_stats['market_size'] = district_stats['monthly_revenue_sum']
        district_stats['business_density'] = district_stats['monthly_revenue_count']
        
        # Revenue per transaction (economic efficiency)
        district_stats['revenue_per_transaction'] = (
            district_stats['monthly_revenue_sum'] / 
            (district_stats['monthly_transactions_sum'] + 0.01)
        )
        
        # Time series coverage
        district_stats['data_years'] = (
            district_stats['quarter_code_max'].astype(str).str[:4].astype(int) -
            district_stats['quarter_code_min'].astype(str).str[:4].astype(int) + 1
        )
        
        # Handle missing values and outliers
        district_stats = self._clean_district_profiles(district_stats)
        
        self.district_profiles = district_stats
        logger.info(f"Created profiles for {len(district_stats)} districts")
        
        return district_stats
    
    def _clean_district_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate district profiles."""
        logger.info("Cleaning district profiles...")
        
        # Fill missing values with appropriate defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Remove extreme outliers using IQR method for clustering features
        clustering_features = ['income_level', 'foot_traffic', 'business_diversity']
        
        for feature in clustering_features:
            if feature in df.columns:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                # Cap extreme outliers (keep within 3*IQR)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"Capping {outliers} outliers in {feature}")
                    df[feature] = df[feature].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        return df
    
    def determine_optimal_clusters(self, df: pd.DataFrame, max_clusters: int = 10) -> int:
        """
        Determine optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            df: District profiles DataFrame
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        logger.info("Determining optimal number of clusters...")
        
        # Prepare features for clustering
        feature_columns = ['income_level', 'foot_traffic', 'business_diversity']
        X = df[feature_columns].values
        
        # Standardize features
        X_scaled = StandardScaler().fit_transform(X)
        
        # Test different cluster numbers
        cluster_range = range(2, min(max_clusters + 1, len(df) // 2))
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                calinski_scores.append(calinski_score)
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
        
        # Find optimal clusters using elbow method
        optimal_clusters = self._find_elbow_point(list(cluster_range), inertias)
        
        # Validate with silhouette score
        if silhouette_scores:
            best_silhouette_idx = np.argmax(silhouette_scores)
            best_silhouette_clusters = list(cluster_range)[best_silhouette_idx]
            
            logger.info(f"Elbow method suggests: {optimal_clusters} clusters")
            logger.info(f"Best silhouette score at: {best_silhouette_clusters} clusters ({silhouette_scores[best_silhouette_idx]:.3f})")
            
            # Use silhouette result if significantly better and within reasonable range
            if (abs(optimal_clusters - best_silhouette_clusters) <= 2 and 
                silhouette_scores[best_silhouette_idx] > 0.3):
                optimal_clusters = best_silhouette_clusters
        
        # Ensure within configured range (6-8 clusters as per template)
        optimal_clusters = max(6, min(8, optimal_clusters))
        
        logger.info(f"Selected {optimal_clusters} clusters for regional grouping")
        return optimal_clusters
    
    def _find_elbow_point(self, x_values: List[int], y_values: List[float]) -> int:
        """Find elbow point in elbow method curve."""
        if len(x_values) < 3:
            return x_values[0]
        
        # Calculate the rate of change
        differences = np.diff(y_values)
        differences2 = np.diff(differences)
        
        # Find the point where the rate of change decreases most
        if len(differences2) > 0:
            elbow_idx = np.argmax(differences2) + 2  # Adjust for diff operations
            if elbow_idx < len(x_values):
                return x_values[elbow_idx]
        
        # Fallback: point where improvement is less than 10% of previous improvement
        for i in range(1, len(differences)):
            if abs(differences[i]) < 0.1 * abs(differences[i-1]):
                return x_values[i + 1]
        
        return x_values[len(x_values) // 2]  # Default to middle
    
    def perform_clustering(self, df: pd.DataFrame, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform K-means clustering on district profiles.
        
        Args:
            df: District profiles DataFrame
            n_clusters: Number of clusters (if None, auto-determined)
            
        Returns:
            Dictionary with clustering results and metadata
        """
        logger.info("Performing regional clustering...")
        
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters(df)
        
        # Prepare clustering features
        feature_columns = ['income_level', 'foot_traffic', 'business_diversity']
        X = df[feature_columns].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-means clustering
        self.cluster_model = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = self.cluster_model.fit_predict(X_scaled)
        
        # Add cluster assignments to profiles
        df_clustered = df.copy()
        df_clustered['cluster_id'] = cluster_labels
        df_clustered['cluster_name'] = df_clustered['cluster_id'].map(self._get_cluster_names(n_clusters))
        
        # Calculate cluster centers in original scale
        self.cluster_centers = pd.DataFrame(
            self.scaler.inverse_transform(self.cluster_model.cluster_centers_),
            columns=feature_columns
        )
        self.cluster_centers['cluster_id'] = range(n_clusters)
        
        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_clusters(df_clustered, feature_columns)
        
        results = {
            'clustered_districts': df_clustered,
            'cluster_centers': self.cluster_centers,
            'cluster_analysis': cluster_analysis,
            'n_clusters': n_clusters,
            'feature_columns': feature_columns,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_score,
            'model_params': self.cluster_model.get_params()
        }
        
        logger.info(f"Clustering completed: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
        return results
    
    def _get_cluster_names(self, n_clusters: int) -> Dict[int, str]:
        """Generate descriptive cluster names."""
        cluster_names = {
            0: "강남권_고소득", 1: "강북권_중산층", 2: "서초권_고급상업", 3: "마포권_문화상업",
            4: "송파권_신도시", 5: "성북권_전통상업", 6: "영등포권_업무상업", 7: "기타지역"
        }
        
        # Return only the required number
        return {i: cluster_names.get(i, f"클러스터_{i+1}") for i in range(n_clusters)}
    
    def _analyze_clusters(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze cluster characteristics and profiles."""
        logger.info("Analyzing cluster characteristics...")
        
        cluster_summary = df.groupby('cluster_id').agg({
            'district_code': 'count',
            'income_level': ['mean', 'std'],
            'foot_traffic': ['mean', 'std'], 
            'business_diversity': ['mean', 'std'],
            'market_size': ['mean', 'sum'],
            'business_density': ['mean', 'sum'],
            'revenue_per_transaction': 'mean'
        }).round(2)
        
        # Flatten column names
        cluster_summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in cluster_summary.columns]
        cluster_summary = cluster_summary.reset_index()
        
        # Add cluster names
        cluster_names = self._get_cluster_names(df['cluster_id'].nunique())
        cluster_summary['cluster_name'] = cluster_summary['cluster_id'].map(cluster_names)
        
        # Calculate cluster characteristics
        cluster_characteristics = {}
        for cluster_id in df['cluster_id'].unique():
            cluster_data = df[df['cluster_id'] == cluster_id]
            
            characteristics = {
                'size': len(cluster_data),
                'districts': cluster_data['district_name'].tolist(),
                'avg_income_level': float(cluster_data['income_level'].mean()),
                'avg_foot_traffic': float(cluster_data['foot_traffic'].mean()),
                'avg_business_diversity': float(cluster_data['business_diversity'].mean()),
                'total_market_size': float(cluster_data['market_size'].sum()),
                'profile': self._describe_cluster_profile(cluster_data, df)
            }
            
            cluster_characteristics[int(cluster_id)] = characteristics
        
        return {
            'summary_table': cluster_summary,
            'detailed_characteristics': cluster_characteristics,
            'total_districts': len(df),
            'feature_importance': self._calculate_feature_importance(df, feature_columns)
        }
    
    def _describe_cluster_profile(self, cluster_data: pd.DataFrame, all_data: pd.DataFrame) -> str:
        """Generate descriptive profile for a cluster."""
        cluster_size = len(cluster_data)
        total_size = len(all_data)
        
        # Compare cluster averages to overall averages
        income_vs_avg = cluster_data['income_level'].mean() / all_data['income_level'].mean()
        traffic_vs_avg = cluster_data['foot_traffic'].mean() / all_data['foot_traffic'].mean()
        diversity_vs_avg = cluster_data['business_diversity'].mean() / all_data['business_diversity'].mean()
        
        profile_parts = []
        
        # Income level description
        if income_vs_avg > 1.3:
            profile_parts.append("고소득지역")
        elif income_vs_avg > 1.1:
            profile_parts.append("중상위소득지역")
        elif income_vs_avg > 0.9:
            profile_parts.append("중간소득지역")
        else:
            profile_parts.append("서민지역")
        
        # Traffic level description
        if traffic_vs_avg > 1.3:
            profile_parts.append("유동인구매우높음")
        elif traffic_vs_avg > 1.1:
            profile_parts.append("유동인구높음")
        elif traffic_vs_avg > 0.9:
            profile_parts.append("유동인구보통")
        else:
            profile_parts.append("유동인구낮음")
        
        # Diversity description
        if diversity_vs_avg > 1.2:
            profile_parts.append("업종다양성높음")
        elif diversity_vs_avg > 0.9:
            profile_parts.append("업종다양성보통")
        else:
            profile_parts.append("업종집중지역")
        
        return ", ".join(profile_parts) + f" ({cluster_size}개 행정동, {cluster_size/total_size*100:.1f}%)"
    
    def _calculate_feature_importance(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, float]:
        """Calculate feature importance for clustering."""
        # Use cluster centers variance as proxy for feature importance
        feature_importance = {}
        
        for feature in feature_columns:
            cluster_means = df.groupby('cluster_id')[feature].mean()
            variance = cluster_means.var()
            feature_importance[feature] = float(variance)
        
        # Normalize to percentages
        total_variance = sum(feature_importance.values())
        if total_variance > 0:
            feature_importance = {k: v/total_variance*100 for k, v in feature_importance.items()}
        
        return feature_importance
    
    def save_clustering_results(self, results: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save clustering results to files."""
        if output_dir is None:
            output_dir = self.data_paths['processed']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save clustered districts
        districts_file = output_dir / 'regional_clusters.csv'
        results['clustered_districts'].to_csv(districts_file, index=False, encoding='utf-8')
        saved_files['districts'] = districts_file
        
        # Save cluster centers
        centers_file = output_dir / 'regional_cluster_centers.csv' 
        results['cluster_centers'].to_csv(centers_file, index=False, encoding='utf-8')
        saved_files['centers'] = centers_file
        
        # Save cluster analysis
        analysis_file = output_dir / 'regional_cluster_analysis.json'
        analysis_data = {
            'metadata': {
                'n_clusters': results['n_clusters'],
                'silhouette_score': results['silhouette_score'],
                'calinski_score': results['calinski_score'],
                'feature_columns': results['feature_columns'],
                'clustering_timestamp': datetime.now().isoformat()
            },
            'cluster_analysis': {
                'summary_table': results['cluster_analysis']['summary_table'].to_dict('records'),
                'detailed_characteristics': results['cluster_analysis']['detailed_characteristics'],
                'feature_importance': results['cluster_analysis']['feature_importance']
            }
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        saved_files['analysis'] = analysis_file
        
        logger.info(f"Clustering results saved to {len(saved_files)} files")
        return saved_files
    
    def load_clustering_results(self, results_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load previously saved clustering results."""
        if results_dir is None:
            results_dir = self.data_paths['processed']
        
        results_dir = Path(results_dir)
        
        try:
            # Load clustered districts
            districts_file = results_dir / 'regional_clusters.csv'
            if not districts_file.exists():
                logger.warning("Regional clustering results not found")
                return None
            
            districts = pd.read_csv(districts_file)
            
            # Load cluster centers
            centers_file = results_dir / 'regional_cluster_centers.csv'
            centers = pd.read_csv(centers_file) if centers_file.exists() else None
            
            # Load analysis
            analysis_file = results_dir / 'regional_cluster_analysis.json'
            analysis = None
            if analysis_file.exists():
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
            
            logger.info(f"Loaded regional clustering results: {len(districts)} districts, {districts['cluster_id'].nunique()} clusters")
            
            return {
                'clustered_districts': districts,
                'cluster_centers': centers,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error loading clustering results: {e}")
            return None


def main():
    """Main function for testing regional clustering."""
    clusterer = SeoulRegionalClusterer()
    
    try:
        # Load processed sales data
        processed_data_path = Path("data/processed/seoul_sales_combined.csv")
        if not processed_data_path.exists():
            logger.error("Processed sales data not found. Run preprocessing first.")
            return
        
        df = pd.read_csv(processed_data_path)
        logger.info(f"Loaded {len(df):,} sales records for clustering")
        
        # Create district profiles
        district_profiles = clusterer.create_district_profiles(df)
        
        # Perform clustering
        clustering_results = clusterer.perform_clustering(district_profiles)
        
        # Save results
        saved_files = clusterer.save_clustering_results(clustering_results)
        
        print("\n=== REGIONAL CLUSTERING RESULTS ===")
        print(f"Total districts: {len(district_profiles)}")
        print(f"Number of clusters: {clustering_results['n_clusters']}")
        print(f"Silhouette score: {clustering_results['silhouette_score']:.3f}")
        print(f"Calinski-Harabasz score: {clustering_results['calinski_score']:.1f}")
        
        print("\nCluster Distribution:")
        cluster_dist = clustering_results['clustered_districts']['cluster_id'].value_counts().sort_index()
        for cluster_id, count in cluster_dist.items():
            cluster_name = clustering_results['clustered_districts'][
                clustering_results['clustered_districts']['cluster_id']==cluster_id
            ]['cluster_name'].iloc[0]
            print(f"  Cluster {cluster_id} ({cluster_name}): {count} districts")
        
        print(f"\nFiles saved: {list(saved_files.values())}")
        
    except Exception as e:
        logger.error(f"Regional clustering failed: {e}")
        raise


if __name__ == "__main__":
    main()