#!/usr/bin/env python3
"""
Efficient Data Preprocessing for Seoul Market Risk ML
====================================================

Optimized version for handling 408K records efficiently.
Uses sampling and vectorized operations for speed.

Author: Seoul Market Risk ML System
Date: 2025-09-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class EfficientDataPreprocessor:
    """Optimized data preprocessor for large datasets"""

    def __init__(self, labeled_data_path: str = "ml_analysis_results/seoul_commercial_labeled_dataset.csv"):
        self.labeled_data_path = labeled_data_path
        self.class_weights = {}

    def load_and_sample_data(self, sample_size: int = 50000) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and sample data for efficient processing"""
        print(f"📂 Loading and sampling data (target size: {sample_size:,})...")

        # Load data
        data = pd.read_csv(self.labeled_data_path, encoding='utf-8')
        print(f"✅ Loaded {len(data):,} total records")

        # Stratified sampling to maintain class distribution
        if len(data) > sample_size:
            sampler = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
            sample_idx, _ = next(sampler.split(data, data['risk_label']))
            data_sampled = data.iloc[sample_idx].reset_index(drop=True)
            print(f"🎯 Sampled {len(data_sampled):,} records (maintaining class distribution)")
        else:
            data_sampled = data

        # Show class distribution
        risk_dist = data_sampled['risk_label'].value_counts().sort_index()
        print("📊 Sample Risk Distribution:")
        for level, count in risk_dist.items():
            pct = (count / len(data_sampled)) * 100
            print(f"   Level {level}: {count:,} ({pct:.1f}%)")

        return data_sampled, data_sampled['risk_label']

    def create_efficient_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features efficiently using vectorized operations"""
        print("⚙️ Creating features efficiently...")

        # Select key financial columns
        key_cols = [
            '당월_매출_금액', '당월_매출_건수', '주중_매출_금액', '주말_매출_금액',
            '시간대_11~14_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액',
            '남성_매출_금액', '여성_매출_금액',
            '연령대_20_매출_금액', '연령대_30_매출_금액', '연령대_40_매출_금액'
        ]

        # Filter available columns
        available_cols = [col for col in key_cols if col in data.columns] + ['데이터연도']
        X = data[available_cols].fillna(0).copy()

        # Vectorized feature engineering
        if '당월_매출_금액' in X.columns and '당월_매출_건수' in X.columns:
            # Basic financial metrics
            X['avg_transaction'] = X['당월_매출_금액'] / (X['당월_매출_건수'] + 1)
            X['revenue_log'] = np.log1p(X['당월_매출_금액'])  # Log transform for revenue

        if '주중_매출_금액' in X.columns and '주말_매출_금액' in X.columns:
            # Weekend performance ratio
            total_wk = X['주중_매출_금액'] + X['주말_매출_금액'] + 1
            X['weekend_ratio'] = X['주말_매출_금액'] / total_wk
            X['weekday_strength'] = X['주중_매출_금액'] / total_wk

        # Peak hour performance
        peak_cols = ['시간대_11~14_매출_금액', '시간대_17~21_매출_금액']
        available_peak = [col for col in peak_cols if col in X.columns]
        if len(available_peak) >= 2:
            X['peak_performance'] = X[available_peak].sum(axis=1)
            X['peak_ratio'] = X['peak_performance'] / (X['당월_매출_금액'] + 1)

        # Gender balance
        if '남성_매출_금액' in X.columns and '여성_매출_금액' in X.columns:
            total_gender = X['남성_매출_금액'] + X['여성_매출_금액'] + 1
            X['female_ratio'] = X['여성_매출_금액'] / total_gender

        # Age group concentration (middle-aged customers)
        age_cols = ['연령대_20_매출_금액', '연령대_30_매출_금액', '연령대_40_매출_금액']
        available_age = [col for col in age_cols if col in X.columns]
        if len(available_age) >= 2:
            X['middle_age_focus'] = X[available_age].sum(axis=1)

        # Time-based features
        if '데이터연도' in X.columns:
            X['covid_period'] = ((X['데이터연도'] == 2020) | (X['데이터연도'] == 2021)).astype(int)
            X['post_covid'] = (X['데이터연도'] >= 2022).astype(int)

        # Add top business types (simplified encoding)
        if '서비스_업종_코드' in data.columns:
            top_businesses = data['서비스_업종_코드'].value_counts().head(10).index
            for i, business in enumerate(top_businesses):
                X[f'biz_type_{i}'] = (data['서비스_업종_코드'] == business).astype(int)

        # Add top regions (simplified encoding)
        if '행정동_코드_명' in data.columns:
            top_regions = data['행정동_코드_명'].value_counts().head(15).index
            for i, region in enumerate(top_regions):
                X[f'region_{i}'] = (data['행정동_코드_명'] == region).astype(int)

        print(f"✅ Created {X.shape[1]} features efficiently")
        return X

    def balance_classes_fast(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Fast class balancing using SMOTE with reduced complexity"""
        print("⚖️ Balancing classes efficiently...")

        original_dist = y.value_counts().sort_index()
        print("Original Distribution:")
        for level, count in original_dist.items():
            pct = (count / len(y)) * 100
            print(f"   Level {level}: {count:,} ({pct:.1f}%)")

        try:
            # Use SMOTE with fewer neighbors for speed
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X, y)

            new_dist = pd.Series(y_balanced).value_counts().sort_index()
            print("Balanced Distribution:")
            for level, count in new_dist.items():
                pct = (count / len(y_balanced)) * 100
                print(f"   Level {level}: {count:,} ({pct:.1f}%)")

            return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)

        except Exception as e:
            print(f"⚠️ SMOTE failed: {e}")
            print("Using original data with class weights...")
            return X, y

    def compute_class_weights_fast(self, y: pd.Series) -> Dict[int, float]:
        """Compute class weights quickly"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, weights))

        print("📊 Class Weights:")
        for class_label, weight in class_weight_dict.items():
            print(f"   Level {class_label}: {weight:.3f}")

        self.class_weights = class_weight_dict
        return class_weight_dict

    def split_and_scale(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Efficient train/val/test split and scaling"""
        print("📊 Splitting and scaling data...")

        # Simple stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
        )

        # Robust scaling
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )

        print(f"✅ Data splits created:")
        print(f"   Training: {len(X_train_scaled):,}")
        print(f"   Validation: {len(X_val_scaled):,}")
        print(f"   Test: {len(X_test_scaled):,}")

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

    def save_preprocessed_data_fast(self, splits: tuple, scaler, output_dir: str = "ml_preprocessed_data"):
        """Save preprocessed data efficiently"""
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"💾 Saving to {output_dir}/...")

        # Save datasets
        train_data = X_train.copy()
        train_data['risk_label'] = y_train.values
        train_data.to_csv(output_path / "train_data.csv", index=False)

        val_data = X_val.copy()
        val_data['risk_label'] = y_val.values
        val_data.to_csv(output_path / "validation_data.csv", index=False)

        test_data = X_test.copy()
        test_data['risk_label'] = y_test.values
        test_data.to_csv(output_path / "test_data.csv", index=False)

        # Save artifacts
        import joblib
        joblib.dump(scaler, output_path / "scaler.joblib")
        joblib.dump(self.class_weights, output_path / "class_weights.joblib")

        # Save feature info
        feature_info = {
            'n_features': len(X_train.columns),
            'feature_names': list(X_train.columns),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }

        import json
        with open(output_path / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)

        print(f"✅ Saved {len(X_train.columns)} features")
        print(f"🎯 Ready for ML training!")

def main():
    """Main efficient preprocessing pipeline"""
    print("🚀 Efficient Data Preprocessing Pipeline")
    print("=" * 45)

    preprocessor = EfficientDataPreprocessor()

    try:
        # Step 1: Load and sample data
        data, labels = preprocessor.load_and_sample_data(sample_size=50000)

        # Step 2: Create features efficiently
        X = preprocessor.create_efficient_features(data)

        # Step 3: Balance classes
        X_balanced, y_balanced = preprocessor.balance_classes_fast(X, labels)

        # Step 4: Compute class weights
        class_weights = preprocessor.compute_class_weights_fast(labels)

        # Step 5: Split and scale
        splits = preprocessor.split_and_scale(X_balanced, y_balanced)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = splits

        # Step 6: Save results
        preprocessor.save_preprocessed_data_fast(
            (X_train, X_val, X_test, y_train, y_val, y_test), scaler
        )

        print(f"\n✅ Efficient preprocessing complete!")
        print(f"📊 Training set: {X_train.shape}")
        print(f"🎯 Ready for ensemble ML training!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()