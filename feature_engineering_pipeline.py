#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Seoul Market Risk ML
====================================================

Transforms simple 5 inputs into 50+ sophisticated ML features using
408,221 Seoul commercial district benchmark records.

Input (Simple):
- total_available_assets: 30,000,000
- monthly_revenue: 8,000,000
- monthly_expenses: {labor_cost, food_materials, rent, others}
- business_type: "한식음식점"
- location: "강남구"

Output: 50+ engineered features for ML training

Author: Seoul Market Risk ML System
Date: 2025-09-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineeringPipeline:
    """Advanced feature engineering pipeline for cashflow risk prediction"""

    def __init__(self, benchmark_data_path: str = "ml_analysis_results/seoul_commercial_labeled_dataset.csv"):
        self.benchmark_data_path = benchmark_data_path
        self.benchmark_data = None
        self.business_benchmarks = {}
        self.regional_benchmarks = {}
        self.industry_stats = {}
        self.scalers = {}
        self.encoders = {}

        # Load benchmark data for feature engineering
        self._load_benchmark_data()
        self._calculate_benchmarks()

    def _load_benchmark_data(self) -> None:
        """Load Seoul commercial district benchmark data"""
        print("🔧 Loading benchmark data for feature engineering...")

        try:
            self.benchmark_data = pd.read_csv(self.benchmark_data_path, encoding='utf-8')
            print(f"✅ Loaded {len(self.benchmark_data):,} benchmark records")
        except FileNotFoundError:
            print("❌ Benchmark data not found. Run data_analysis_and_labeling.py first!")
            raise
        except Exception as e:
            print(f"❌ Error loading benchmark data: {e}")
            raise

    def _calculate_benchmarks(self) -> None:
        """Calculate industry and regional benchmarks from historical data"""
        print("📊 Calculating industry and regional benchmarks...")

        if self.benchmark_data is None:
            return

        # Business type benchmarks
        business_stats = self.benchmark_data.groupby('서비스_업종_코드').agg({
            '당월_매출_금액': ['mean', 'median', 'std', 'count'],
            '당월_매출_건수': ['mean', 'median', 'std'],
            'risk_label': ['mean', 'std']
        }).round(2)

        self.business_benchmarks = {}
        for business_code in business_stats.index:
            self.business_benchmarks[business_code] = {
                'revenue_mean': business_stats.loc[business_code, ('당월_매출_금액', 'mean')],
                'revenue_median': business_stats.loc[business_code, ('당월_매출_금액', 'median')],
                'revenue_std': business_stats.loc[business_code, ('당월_매출_금액', 'std')],
                'transaction_mean': business_stats.loc[business_code, ('당월_매출_건수', 'mean')],
                'risk_mean': business_stats.loc[business_code, ('risk_label', 'mean')],
                'sample_count': business_stats.loc[business_code, ('당월_매출_금액', 'count')]
            }

        # Regional benchmarks
        regional_stats = self.benchmark_data.groupby('행정동_코드_명').agg({
            '당월_매출_금액': ['mean', 'median', 'std', 'count'],
            '당월_매출_건수': ['mean', 'median', 'std'],
            'risk_label': ['mean', 'std']
        }).round(2)

        self.regional_benchmarks = {}
        for region_name in regional_stats.index:
            self.regional_benchmarks[region_name] = {
                'revenue_mean': regional_stats.loc[region_name, ('당월_매출_금액', 'mean')],
                'revenue_median': regional_stats.loc[region_name, ('당월_매출_금액', 'median')],
                'revenue_std': regional_stats.loc[region_name, ('당월_매출_금액', 'std')],
                'transaction_mean': regional_stats.loc[region_name, ('당월_매출_건수', 'mean')],
                'risk_mean': regional_stats.loc[region_name, ('risk_label', 'mean')],
                'sample_count': regional_stats.loc[region_name, ('당월_매출_금액', 'count')]
            }

        print(f"✅ Calculated benchmarks for {len(self.business_benchmarks)} business types")
        print(f"✅ Calculated benchmarks for {len(self.regional_benchmarks)} regions")

    def transform_simple_input(self, input_data: Dict) -> Dict[str, float]:
        """
        Transform simple 5 inputs into 50+ ML features

        Args:
            input_data: {
                "total_available_assets": 30000000,
                "monthly_revenue": 8000000,
                "monthly_expenses": {
                    "labor_cost": 4000000,
                    "food_materials": 3000000,
                    "rent": 2000000,
                    "others": 1000000
                },
                "business_type": "한식음식점",
                "location": "관악구"
            }

        Returns:
            Dictionary with 50+ engineered features
        """
        print(f"⚙️ Engineering features from simple input...")

        features = {}

        # Extract basic values
        assets = input_data['total_available_assets']
        revenue = input_data['monthly_revenue']
        expenses = input_data['monthly_expenses']
        business_type = input_data['business_type']
        location = input_data['location']

        total_expenses = sum(expenses.values())

        # =================================================
        # 1. BASIC FINANCIAL RATIOS (10 features)
        # =================================================
        features['monthly_profit'] = revenue - total_expenses
        features['profit_margin'] = features['monthly_profit'] / revenue if revenue > 0 else 0
        features['expense_ratio'] = total_expenses / revenue if revenue > 0 else 1
        features['asset_utilization'] = revenue / assets if assets > 0 else 0
        features['cash_conversion'] = assets / (total_expenses * 12) if total_expenses > 0 else 12
        features['revenue_to_expense'] = revenue / total_expenses if total_expenses > 0 else 0
        features['break_even_coverage'] = features['monthly_profit'] / total_expenses if total_expenses > 0 else 0
        features['operating_leverage'] = total_expenses / revenue if revenue > 0 else 0
        features['liquidity_months'] = assets / total_expenses if total_expenses > 0 else 12
        features['roi_monthly'] = features['monthly_profit'] / assets if assets > 0 else 0

        # =================================================
        # 2. EXPENSE STRUCTURE ANALYSIS (12 features)
        # =================================================
        features['labor_cost_ratio'] = expenses['labor_cost'] / total_expenses if total_expenses > 0 else 0
        features['materials_cost_ratio'] = expenses['food_materials'] / total_expenses if total_expenses > 0 else 0
        features['rent_cost_ratio'] = expenses['rent'] / total_expenses if total_expenses > 0 else 0
        features['others_cost_ratio'] = expenses['others'] / total_expenses if total_expenses > 0 else 0

        features['labor_to_revenue'] = expenses['labor_cost'] / revenue if revenue > 0 else 0
        features['materials_to_revenue'] = expenses['food_materials'] / revenue if revenue > 0 else 0
        features['rent_to_revenue'] = expenses['rent'] / revenue if revenue > 0 else 0
        features['others_to_revenue'] = expenses['others'] / revenue if revenue > 0 else 0

        # Expense diversity (lower = more concentrated risk)
        expense_values = list(expenses.values())
        features['expense_diversity_index'] = self._calculate_diversity_index(expense_values)
        features['expense_concentration_risk'] = max(expense_values) / sum(expense_values) if sum(expense_values) > 0 else 0
        features['fixed_vs_variable'] = (expenses['rent']) / (expenses['labor_cost'] + expenses['food_materials']) if (expenses['labor_cost'] + expenses['food_materials']) > 0 else 0
        features['cost_structure_risk'] = (expenses['labor_cost'] + expenses['rent']) / total_expenses if total_expenses > 0 else 0

        # =================================================
        # 3. INDUSTRY BENCHMARKING (12 features)
        # =================================================
        industry_benchmark = self._get_industry_benchmark(business_type)

        features['industry_revenue_ratio'] = revenue / industry_benchmark['revenue_mean'] if industry_benchmark['revenue_mean'] > 0 else 1
        features['industry_revenue_zscore'] = (revenue - industry_benchmark['revenue_mean']) / industry_benchmark['revenue_std'] if industry_benchmark['revenue_std'] > 0 else 0
        features['industry_percentile'] = self._calculate_percentile(revenue, business_type, 'revenue')

        features['industry_expense_ratio'] = total_expenses / (industry_benchmark['revenue_mean'] * 0.7) if industry_benchmark['revenue_mean'] > 0 else 1
        features['industry_profit_gap'] = features['monthly_profit'] - (industry_benchmark['revenue_mean'] * 0.3)
        features['industry_risk_deviation'] = industry_benchmark['risk_mean'] - 3.0  # 3.0 is middle risk level

        features['above_industry_median'] = 1 if revenue > industry_benchmark['revenue_median'] else 0
        features['industry_top_quartile'] = 1 if features['industry_percentile'] > 0.75 else 0
        features['industry_bottom_quartile'] = 1 if features['industry_percentile'] < 0.25 else 0

        features['industry_stability_score'] = min(1.0, industry_benchmark['sample_count'] / 1000)
        features['industry_volatility'] = industry_benchmark['revenue_std'] / industry_benchmark['revenue_mean'] if industry_benchmark['revenue_mean'] > 0 else 1
        features['industry_competitive_position'] = self._calculate_competitive_position(revenue, business_type)

        # =================================================
        # 4. REGIONAL BENCHMARKING (10 features)
        # =================================================
        regional_benchmark = self._get_regional_benchmark(location)

        features['regional_revenue_ratio'] = revenue / regional_benchmark['revenue_mean'] if regional_benchmark['revenue_mean'] > 0 else 1
        features['regional_revenue_zscore'] = (revenue - regional_benchmark['revenue_mean']) / regional_benchmark['revenue_std'] if regional_benchmark['revenue_std'] > 0 else 0
        features['regional_percentile'] = self._calculate_percentile(revenue, location, 'region')

        features['regional_expense_ratio'] = total_expenses / (regional_benchmark['revenue_mean'] * 0.7) if regional_benchmark['revenue_mean'] > 0 else 1
        features['regional_risk_factor'] = regional_benchmark['risk_mean']

        features['regional_market_strength'] = min(1.0, regional_benchmark['sample_count'] / 2000)
        features['regional_above_median'] = 1 if revenue > regional_benchmark['revenue_median'] else 0
        features['regional_top_performer'] = 1 if features['regional_percentile'] > 0.8 else 0
        features['regional_volatility'] = regional_benchmark['revenue_std'] / regional_benchmark['revenue_mean'] if regional_benchmark['revenue_mean'] > 0 else 1
        features['location_advantage'] = self._calculate_location_advantage(location)

        # =================================================
        # 5. RISK INDICATORS (8 features)
        # =================================================
        features['cash_runway_months'] = assets / total_expenses if total_expenses > 0 else 12
        features['debt_service_capacity'] = features['monthly_profit'] / (total_expenses * 0.1) if total_expenses > 0 else 0  # Assuming 10% debt service
        features['business_sustainability'] = (revenue - expenses['labor_cost'] - expenses['rent']) / revenue if revenue > 0 else 0

        features['operational_risk'] = expenses['labor_cost'] / revenue + features['materials_cost_ratio']  # Higher = more operational dependency
        features['market_risk'] = 1 - features['industry_stability_score']
        features['financial_stress'] = 1 - features['profit_margin'] if features['profit_margin'] < 1 else 0

        features['early_warning_score'] = (
            features['cash_runway_months'] * 0.3 +
            features['profit_margin'] * 0.4 +
            features['industry_revenue_ratio'] * 0.3
        ) / 3

        features['composite_risk_score'] = 1 - features['early_warning_score']

        # =================================================
        # 6. CATEGORICAL ENCODINGS (Business + Region)
        # =================================================
        # Business type encoding
        business_encoded = self._encode_business_type(business_type)
        for key, value in business_encoded.items():
            features[f'business_{key}'] = value

        # Regional encoding
        regional_encoded = self._encode_location(location)
        for key, value in regional_encoded.items():
            features[f'region_{key}'] = value

        print(f"✅ Generated {len(features)} engineered features")
        return features

    def _calculate_diversity_index(self, values: List[float]) -> float:
        """Calculate diversity index (higher = more diverse)"""
        if not values or sum(values) == 0:
            return 0

        proportions = [v / sum(values) for v in values]
        # Shannon entropy as diversity measure
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in proportions)
        # Normalize to 0-1 scale
        max_entropy = np.log(len(values))
        return entropy / max_entropy if max_entropy > 0 else 0

    def _get_industry_benchmark(self, business_type: str) -> Dict[str, float]:
        """Get industry benchmark statistics"""
        # Map Korean business names to codes (simplified)
        business_mapping = {
            "한식음식점": "CS200001",
            "카페": "CS200007",
            "편의점": "CS100001",
            "미용실": "CS300002",
            "치킨": "CS200028",
            "피자": "CS200006",
            "중국음식점": "CS200005"
        }

        business_code = business_mapping.get(business_type)

        if business_code and business_code in self.business_benchmarks:
            return self.business_benchmarks[business_code]
        else:
            # Return average benchmarks
            return {
                'revenue_mean': 1390033072,  # Average from analysis
                'revenue_median': 253656152,
                'revenue_std': 1000000000,
                'transaction_mean': 51639,
                'risk_mean': 1.5,
                'sample_count': 1000
            }

    def _get_regional_benchmark(self, location: str) -> Dict[str, float]:
        """Get regional benchmark statistics"""
        # Map location names (simplified)
        location_mapping = {
            "강남구": "역삼1동",
            "서초구": "서초1동",
            "마포구": "서교동",
            "종로구": "종로1가동",
            "관악구": "신림동"
        }

        region_name = location_mapping.get(location)

        if region_name and region_name in self.regional_benchmarks:
            return self.regional_benchmarks[region_name]
        else:
            # Return average benchmarks
            return {
                'revenue_mean': 1390033072,
                'revenue_median': 253656152,
                'revenue_std': 1000000000,
                'transaction_mean': 51639,
                'risk_mean': 1.5,
                'sample_count': 1000
            }

    def _calculate_percentile(self, value: float, category: str, category_type: str) -> float:
        """Calculate percentile within industry or region"""
        if category_type == 'revenue':
            # Industry percentile
            business_code = None
            # Get business code mapping...
            return 0.5  # Simplified for now
        else:
            # Regional percentile
            return 0.5  # Simplified for now

    def _calculate_competitive_position(self, revenue: float, business_type: str) -> float:
        """Calculate competitive position score (0-1)"""
        benchmark = self._get_industry_benchmark(business_type)
        ratio = revenue / benchmark['revenue_mean'] if benchmark['revenue_mean'] > 0 else 1

        # Convert ratio to 0-1 competitive position score
        if ratio >= 2.0:
            return 1.0  # Market leader
        elif ratio >= 1.5:
            return 0.8  # Strong position
        elif ratio >= 1.0:
            return 0.6  # Above average
        elif ratio >= 0.5:
            return 0.4  # Below average
        else:
            return 0.2  # Struggling

    def _calculate_location_advantage(self, location: str) -> float:
        """Calculate location advantage score (0-1)"""
        # Simplified location scoring
        premium_locations = ["강남구", "서초구", "종로구"]
        good_locations = ["마포구", "송파구", "영등포구"]

        if location in premium_locations:
            return 1.0
        elif location in good_locations:
            return 0.7
        else:
            return 0.5

    def _encode_business_type(self, business_type: str) -> Dict[str, float]:
        """Encode business type into multiple features"""
        # Business category features
        food_service = ["한식음식점", "중국음식점", "일식음식점", "치킨", "피자", "카페"]
        retail = ["편의점", "의류", "화장품"]
        service = ["미용실", "세탁소", "PC방"]

        return {
            'is_food_service': 1.0 if business_type in food_service else 0.0,
            'is_retail': 1.0 if business_type in retail else 0.0,
            'is_service': 1.0 if business_type in service else 0.0,
            'is_restaurant': 1.0 if "음식점" in business_type else 0.0,
            'covid_impact_risk': 1.0 if business_type in ["카페", "한식음식점", "미용실"] else 0.3
        }

    def _encode_location(self, location: str) -> Dict[str, float]:
        """Encode location into multiple features"""
        # Location category features
        affluent_areas = ["강남구", "서초구", "송파구"]
        business_districts = ["종로구", "중구", "영등포구"]
        residential = ["관악구", "동작구", "서대문구"]

        return {
            'is_affluent_area': 1.0 if location in affluent_areas else 0.0,
            'is_business_district': 1.0 if location in business_districts else 0.0,
            'is_residential': 1.0 if location in residential else 0.0,
            'foot_traffic_score': self._get_foot_traffic_score(location),
            'rent_burden_score': self._get_rent_burden_score(location)
        }

    def _get_foot_traffic_score(self, location: str) -> float:
        """Get estimated foot traffic score (0-1)"""
        high_traffic = ["강남구", "종로구", "마포구"]
        medium_traffic = ["서초구", "영등포구", "송파구"]

        if location in high_traffic:
            return 1.0
        elif location in medium_traffic:
            return 0.7
        else:
            return 0.5

    def _get_rent_burden_score(self, location: str) -> float:
        """Get rent burden score (higher = more expensive)"""
        expensive = ["강남구", "서초구", "종로구"]
        moderate = ["마포구", "송파구", "영등포구"]

        if location in expensive:
            return 1.0
        elif location in moderate:
            return 0.6
        else:
            return 0.3

    def create_training_features(self, sample_data: List[Dict]) -> pd.DataFrame:
        """Convert list of simple inputs to ML training dataframe"""
        print(f"🔄 Converting {len(sample_data)} samples to ML features...")

        all_features = []
        for i, data in enumerate(sample_data):
            if i % 1000 == 0 and i > 0:
                print(f"  Processed: {i:,}/{len(sample_data):,}")

            features = self.transform_simple_input(data)
            all_features.append(features)

        df = pd.DataFrame(all_features)
        print(f"✅ Created training dataset: {df.shape}")
        return df

def demo_feature_engineering():
    """Demonstrate feature engineering with sample input"""
    print("🧪 Feature Engineering Pipeline Demo")
    print("=" * 40)

    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline()

    # Sample input (as per requirements)
    sample_input = {
        "total_available_assets": 30000000,    # 30M won
        "monthly_revenue": 8000000,            # 8M won
        "monthly_expenses": {
            "labor_cost": 4000000,             # 4M won
            "food_materials": 3000000,         # 3M won
            "rent": 2000000,                   # 2M won
            "others": 1000000                  # 1M won
        },
        "business_type": "한식음식점",
        "location": "관악구"
    }

    print("📥 Sample Input:")
    print(f"  Assets: {sample_input['total_available_assets']:,} won")
    print(f"  Revenue: {sample_input['monthly_revenue']:,} won")
    print(f"  Expenses: {sum(sample_input['monthly_expenses'].values()):,} won")
    print(f"  Business: {sample_input['business_type']}")
    print(f"  Location: {sample_input['location']}")

    # Generate features
    features = pipeline.transform_simple_input(sample_input)

    print(f"\n📤 Generated Features ({len(features)} total):")
    print("-" * 50)

    # Group and display features
    feature_groups = {
        'Basic Financial': [k for k in features.keys() if any(x in k for x in ['profit', 'margin', 'ratio', 'utilization'])],
        'Expense Structure': [k for k in features.keys() if 'cost' in k or 'expense' in k],
        'Industry Benchmark': [k for k in features.keys() if 'industry' in k],
        'Regional Benchmark': [k for k in features.keys() if 'regional' in k],
        'Risk Indicators': [k for k in features.keys() if 'risk' in k or 'runway' in k or 'warning' in k],
        'Business Encoding': [k for k in features.keys() if 'business_' in k],
        'Location Encoding': [k for k in features.keys() if 'region_' in k]
    }

    for group_name, feature_list in feature_groups.items():
        if feature_list:
            print(f"\n🎯 {group_name} ({len(feature_list)} features):")
            for feature in feature_list[:5]:  # Show top 5 in each group
                value = features[feature]
                print(f"   {feature}: {value:.3f}")
            if len(feature_list) > 5:
                print(f"   ... and {len(feature_list) - 5} more")

    print(f"\n✅ Feature engineering pipeline ready for ML training!")
    return features

if __name__ == "__main__":
    demo_feature_engineering()