#!/usr/bin/env python3
"""
Data Preprocessing and Train/Test Splitting for Seoul Market Risk ML
===================================================================

Handles the critical class imbalance issue and creates proper train/test splits
considering temporal, regional, and industry factors.

Class Distribution Issue:
- Level 1 (매우여유): 87.5% ← SEVERE IMBALANCE
- Level 2-5: Only 12.5% combined

Solutions Applied:
- SMOTE oversampling
- Stratified sampling
- Class weight balancing
- Time-aware splitting

Author: Seoul Market Risk ML System
Date: 2025-09-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Advanced data preprocessing with class imbalance handling"""

    def __init__(self, labeled_data_path: str = "ml_analysis_results/seoul_commercial_labeled_dataset.csv"):
        self.labeled_data_path = labeled_data_path
        self.raw_data = None
        self.feature_columns = []
        self.scalers = {}
        self.class_weights = {}
        self.preprocessing_stats = {}

    def load_labeled_data(self) -> pd.DataFrame:
        """Load the labeled Seoul commercial dataset"""
        print("📂 Loading labeled dataset for preprocessing...")

        try:
            self.raw_data = pd.read_csv(self.labeled_data_path, encoding='utf-8')
            print(f"✅ Loaded {len(self.raw_data):,} labeled records")

            # Display class distribution
            risk_dist = self.raw_data['risk_label'].value_counts().sort_index()
            print(f"\n📊 Original Risk Distribution:")
            risk_names = {1: "매우여유", 2: "여유", 3: "보통", 4: "위험", 5: "매우위험"}

            for level, count in risk_dist.items():
                pct = (count / len(self.raw_data)) * 100
                print(f"   Level {level} ({risk_names[level]}): {count:,} ({pct:.1f}%)")

            return self.raw_data

        except FileNotFoundError:
            print("❌ Labeled data not found. Run data_analysis_and_labeling.py first!")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def prepare_ml_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract relevant features for ML training from raw data"""
        print("\n🔧 Preparing ML features from raw data...")

        if self.raw_data is None:
            raise ValueError("Data not loaded")

        # Select relevant financial columns for ML features
        financial_cols = [
            '당월_매출_금액', '당월_매출_건수',
            '주중_매출_금액', '주말_매출_금액',
            '월요일_매출_금액', '화요일_매출_금액', '수요일_매출_금액', '목요일_매출_금액',
            '금요일_매출_금액', '토요일_매출_금액', '일요일_매출_금액',
            '시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액',
            '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액',
            '남성_매출_금액', '여성_매출_금액',
            '연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액',
            '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액'
        ]

        # Add transaction count features
        transaction_cols = [col for col in self.raw_data.columns if '매출_건수' in col]

        # Combine all feature columns (remove duplicates)
        all_feature_cols = financial_cols + transaction_cols + ['데이터연도']
        all_feature_cols = list(set(all_feature_cols))  # Remove duplicates

        # Filter existing columns
        available_cols = [col for col in all_feature_cols if col in self.raw_data.columns]
        self.feature_columns = available_cols

        print(f"✅ Selected {len(available_cols)} feature columns from raw data")

        # Create feature matrix
        X = self.raw_data[available_cols].copy()
        y = self.raw_data['risk_label'].copy()

        # Reset index to avoid alignment issues
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Add categorical features
        if '서비스_업종_코드' in self.raw_data.columns:
            # Business type one-hot encoding (top 20 most common)
            top_business_types = self.raw_data['서비스_업종_코드'].value_counts().head(20).index
            for business_type in top_business_types:
                X[f'business_{business_type}'] = (self.raw_data['서비스_업종_코드'].reset_index(drop=True) == business_type).astype(int)

        if '행정동_코드_명' in self.raw_data.columns:
            # Region one-hot encoding (top 30 most common)
            top_regions = self.raw_data['행정동_코드_명'].value_counts().head(30).index
            for region in top_regions:
                X[f'region_{region}'] = (self.raw_data['행정동_코드_명'].reset_index(drop=True) == region).astype(int)

        # Handle missing values
        X = X.fillna(0)

        # Ensure no duplicate columns
        X = X.loc[:, ~X.columns.duplicated()]

        print(f"📊 Final Feature Matrix: {X.shape}")
        print(f"🎯 Target Distribution: {y.value_counts().to_dict()}")

        return X, y

    def create_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features from raw financial data"""
        print("⚙️ Creating engineered features...")

        X_eng = X.copy()

        # Financial ratios and indicators
        if '당월_매출_금액' in X_eng.columns and '당월_매출_건수' in X_eng.columns:
            # Average transaction value
            X_eng['avg_transaction_value'] = X_eng['당월_매출_금액'] / (X_eng['당월_매출_건수'] + 1)

            # Weekend vs weekday performance
            if '주중_매출_금액' in X_eng.columns and '주말_매출_금액' in X_eng.columns:
                total_revenue = X_eng['주중_매출_금액'] + X_eng['주말_매출_금액']
                X_eng['weekend_ratio'] = X_eng['주말_매출_금액'] / (total_revenue + 1)
                X_eng['weekday_dominance'] = X_eng['주중_매출_금액'] / (X_eng['주말_매출_금액'] + 1)

            # Time-based performance patterns
            time_cols = [col for col in X_eng.columns if '시간대_' in col and '매출_금액' in col]
            if len(time_cols) >= 6:
                # Peak vs off-peak performance
                peak_hours = ['시간대_11~14_매출_금액', '시간대_17~21_매출_금액']
                off_peak = ['시간대_00~06_매출_금액', '시간대_21~24_매출_금액']

                peak_revenue = sum(X_eng[col] for col in peak_hours if col in X_eng.columns)
                off_peak_revenue = sum(X_eng[col] for col in off_peak if col in X_eng.columns)

                X_eng['peak_performance_ratio'] = peak_revenue / (off_peak_revenue + 1)
                X_eng['revenue_consistency'] = 1 - (X_eng[time_cols].std(axis=1) / (X_eng[time_cols].mean(axis=1) + 1))

            # Customer demographics
            demo_cols = [col for col in X_eng.columns if '연령대_' in col and '매출_금액' in col]
            if len(demo_cols) >= 5:
                total_demo_revenue = X_eng[demo_cols].sum(axis=1)
                X_eng['young_customer_ratio'] = (X_eng['연령대_10_매출_금액'] + X_eng['연령대_20_매출_금액']) / (total_demo_revenue + 1)
                X_eng['senior_customer_ratio'] = X_eng['연령대_60_이상_매출_금액'] / (total_demo_revenue + 1)

                # Customer diversity (entropy-based)
                demo_proportions = X_eng[demo_cols].div(total_demo_revenue + 1, axis=0)
                X_eng['customer_diversity'] = -demo_proportions.multiply(np.log(demo_proportions + 1e-10)).sum(axis=1)

            # Gender balance
            if '남성_매출_금액' in X_eng.columns and '여성_매출_금액' in X_eng.columns:
                total_gender_revenue = X_eng['남성_매출_금액'] + X_eng['여성_매출_금액']
                X_eng['gender_balance'] = 1 - abs(X_eng['남성_매출_금액'] - X_eng['여성_매출_금액']) / (total_gender_revenue + 1)

        # Year-based features (economic cycle indicators)
        if '데이터연도' in X_eng.columns:
            X_eng['covid_impact_year'] = ((X_eng['데이터연도'] == 2020) | (X_eng['데이터연도'] == 2021)).astype(int)
            X_eng['recovery_year'] = ((X_eng['데이터연도'] == 2022) | (X_eng['데이터연도'] == 2023)).astype(int)
            X_eng['economic_cycle'] = X_eng['데이터연도'] - 2019  # Years since baseline

        print(f"✅ Enhanced feature matrix: {X_eng.shape}")
        return X_eng

    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series,
                             method: str = 'smote_tomek') -> Tuple[pd.DataFrame, pd.Series]:
        """Handle severe class imbalance using various techniques"""
        print(f"\n⚖️ Handling Class Imbalance (Method: {method})")

        original_dist = y.value_counts().sort_index()
        print("Original Distribution:")
        for level, count in original_dist.items():
            pct = (count / len(y)) * 100
            print(f"   Level {level}: {count:,} ({pct:.1f}%)")

        if method == 'smote':
            # Standard SMOTE
            sampler = SMOTE(random_state=42, k_neighbors=5)
        elif method == 'borderline_smote':
            # Borderline SMOTE (focuses on difficult cases)
            sampler = BorderlineSMOTE(random_state=42, k_neighbors=5)
        elif method == 'adasyn':
            # ADASYN (adaptive synthetic sampling)
            sampler = ADASYN(random_state=42, n_neighbors=5)
        elif method == 'smote_tomek':
            # SMOTE + Tomek links cleaning
            sampler = SMOTETomek(random_state=42)
        elif method == 'smote_enn':
            # SMOTE + Edited Nearest Neighbours
            sampler = SMOTEENN(random_state=42)
        else:
            # No sampling
            return X, y

        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)

            new_dist = pd.Series(y_resampled).value_counts().sort_index()
            print(f"\nResampled Distribution ({method}):")
            for level, count in new_dist.items():
                pct = (count / len(y_resampled)) * 100
                print(f"   Level {level}: {count:,} ({pct:.1f}%)")

            print(f"📊 Dataset size: {len(X):,} → {len(X_resampled):,}")

            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

        except Exception as e:
            print(f"❌ Resampling failed: {e}")
            print("Using original data...")
            return X, y

    def compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Compute class weights for imbalanced learning"""
        print("\n⚖️ Computing class weights...")

        # Compute balanced class weights
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)

        class_weight_dict = dict(zip(classes, weights))

        print("Class Weights:")
        for class_label, weight in class_weight_dict.items():
            print(f"   Level {class_label}: {weight:.3f}")

        self.class_weights = class_weight_dict
        return class_weight_dict

    def split_data_time_aware(self, X: pd.DataFrame, y: pd.Series,
                             test_size: float = 0.2, validation_size: float = 0.15) -> Tuple:
        """Create time-aware train/validation/test splits"""
        print(f"\n📅 Creating time-aware data splits...")
        print(f"   Test Size: {test_size:.1%}")
        print(f"   Validation Size: {validation_size:.1%}")

        # Check if we have year information
        if '데이터연도' in X.columns:
            print("Using time-aware splitting based on year...")

            # Sort by year for temporal splitting
            time_idx = X['데이터연도'].argsort()
            X_sorted = X.iloc[time_idx]
            y_sorted = y.iloc[time_idx]

            # Calculate split points
            n_total = len(X_sorted)
            n_test = int(n_total * test_size)
            n_val = int(n_total * validation_size)
            n_train = n_total - n_test - n_val

            # Time-based splits (newer data for test)
            X_train = X_sorted.iloc[:n_train]
            y_train = y_sorted.iloc[:n_train]

            X_val = X_sorted.iloc[n_train:n_train + n_val]
            y_val = y_sorted.iloc[n_train:n_train + n_val]

            X_test = X_sorted.iloc[n_train + n_val:]
            y_test = y_sorted.iloc[n_train + n_val:]

        else:
            print("Using stratified random splitting...")

            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )

            # Second split: train vs val
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
            )

        # Print split statistics
        print(f"\n📊 Data Split Summary:")
        print(f"   Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

        # Check class distribution in each split
        for split_name, split_y in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
            dist = split_y.value_counts().sort_index()
            print(f"\n   {split_name} Distribution:")
            for level, count in dist.items():
                pct = (count / len(split_y)) * 100
                print(f"      Level {level}: {count:,} ({pct:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                      method: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale features using robust scaling (handles outliers better)"""
        print(f"\n📐 Scaling features (Method: {method})")

        if method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        # Fit on training data only
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        # Transform validation and test
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        # Store scaler for future use
        self.scalers['feature_scaler'] = scaler

        print(f"✅ Features scaled using {method} scaler")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def save_preprocessed_data(self, X_train, X_val, X_test, y_train, y_val, y_test,
                              output_dir: str = "ml_preprocessed_data") -> None:
        """Save all preprocessed data for ML training"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n💾 Saving preprocessed data to {output_dir}/")

        # Save training data
        train_data = X_train.copy()
        train_data['risk_label'] = y_train
        train_data.to_csv(output_path / "train_data.csv", index=False)

        # Save validation data
        val_data = X_val.copy()
        val_data['risk_label'] = y_val
        val_data.to_csv(output_path / "validation_data.csv", index=False)

        # Save test data
        test_data = X_test.copy()
        test_data['risk_label'] = y_test
        test_data.to_csv(output_path / "test_data.csv", index=False)

        # Save preprocessing artifacts
        import joblib
        joblib.dump(self.scalers, output_path / "scalers.joblib")
        joblib.dump(self.class_weights, output_path / "class_weights.joblib")

        # Save feature names
        feature_info = {
            'feature_columns': list(X_train.columns),
            'n_features': len(X_train.columns),
            'preprocessing_stats': self.preprocessing_stats
        }

        import json
        with open(output_path / "feature_info.json", 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved training data: {len(train_data):,} samples")
        print(f"✅ Saved validation data: {len(val_data):,} samples")
        print(f"✅ Saved test data: {len(test_data):,} samples")
        print(f"✅ Saved {len(X_train.columns)} features")
        print(f"🎯 Ready for ensemble ML model training!")

def main():
    """Main preprocessing pipeline execution"""
    print("🚀 Data Preprocessing & Splitting Pipeline")
    print("=" * 50)

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    try:
        # Step 1: Load labeled data
        raw_data = preprocessor.load_labeled_data()

        # Step 2: Prepare ML features
        X, y = preprocessor.prepare_ml_features()

        # Step 3: Create engineered features
        X_enhanced = preprocessor.create_engineered_features(X)

        # Step 4: Handle class imbalance
        X_balanced, y_balanced = preprocessor.handle_class_imbalance(X_enhanced, y, method='smote_tomek')

        # Step 5: Compute class weights (for algorithms that support it)
        class_weights = preprocessor.compute_class_weights(y)

        # Step 6: Create train/val/test splits
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data_time_aware(
            X_balanced, y_balanced, test_size=0.2, validation_size=0.15
        )

        # Step 7: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(
            X_train, X_val, X_test, method='robust'
        )

        # Step 8: Save preprocessed data
        preprocessor.save_preprocessed_data(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test
        )

        print(f"\n✅ Preprocessing Complete!")
        print(f"📊 Final Training Set: {X_train_scaled.shape}")
        print(f"🎯 Ready for Ensemble ML Model Training")

    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()