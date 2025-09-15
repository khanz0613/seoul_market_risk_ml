"""
Synthetic Data Generator
소상공인 비용 예측을 위한 합성 학습 데이터 생성기

data/raw의 실제 매출 데이터 + 업종별 비용 구조를 활용하여
ML 학습용 합성 데이터 생성
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import glob

# 프로젝트 경로 추가
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_processing.industry_mapper import IndustryMapper

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """합성 학습 데이터 생성기"""

    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize the synthetic data generator

        Args:
            data_path: Path to raw CSV data files
        """
        self.data_path = Path(data_path)
        self.industry_mapper = IndustryMapper()
        self.raw_data = None
        self.synthetic_data = None

        logger.info(f"SyntheticDataGenerator 초기화: {data_path}")

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load all CSV files from data/raw directory

        Returns:
            Combined DataFrame with all years of data
        """
        logger.info("원시 데이터 로딩 시작...")

        # Find all CSV files in the directory
        csv_files = list(self.data_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")

        logger.info(f"발견된 CSV 파일 수: {len(csv_files)}")

        dataframes = []
        total_rows = 0

        for csv_file in csv_files:
            logger.info(f"로딩 중: {csv_file.name}")

            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                dataframes.append(df)
                total_rows += len(df)
                logger.info(f"  로딩 완료: {len(df):,} rows")

            except Exception as e:
                logger.error(f"파일 로딩 실패 {csv_file.name}: {e}")
                continue

        if not dataframes:
            raise ValueError("유효한 CSV 파일을 찾을 수 없습니다")

        # Combine all dataframes
        self.raw_data = pd.concat(dataframes, ignore_index=True)

        logger.info(f"전체 데이터 로딩 완료: {total_rows:,} rows, {len(self.raw_data.columns)} columns")
        logger.info(f"컬럼: {list(self.raw_data.columns)}")

        return self.raw_data

    def clean_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and filter the raw data for ML training

        Args:
            df: Raw data DataFrame

        Returns:
            Cleaned DataFrame ready for synthetic data generation
        """
        logger.info("데이터 정제 및 필터링 시작...")

        initial_rows = len(df)

        # Remove rows with missing critical data
        required_columns = ['서비스_업종_코드', '당월_매출_금액', '행정동_코드']
        df = df.dropna(subset=required_columns)

        # Filter out zero or negative sales
        df = df[df['당월_매출_금액'] > 0]

        # Filter out extremely high sales (outliers) - top 0.1%
        sales_99_9_percentile = df['당월_매출_금액'].quantile(0.999)
        df = df[df['당월_매출_금액'] <= sales_99_9_percentile]

        # Filter out industries not in our mapping
        valid_industry_codes = set(self.industry_mapper.INDUSTRY_MAPPING.keys())
        df = df[df['서비스_업종_코드'].isin(valid_industry_codes)]

        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        removal_rate = (removed_rows / initial_rows) * 100

        logger.info(f"데이터 정제 완료:")
        logger.info(f"  초기 행 수: {initial_rows:,}")
        logger.info(f"  최종 행 수: {final_rows:,}")
        logger.info(f"  제거된 행: {removed_rows:,} ({removal_rate:.1f}%)")

        return df

    def add_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features for better ML performance

        Args:
            df: Cleaned DataFrame

        Returns:
            DataFrame with additional features
        """
        logger.info("피처 엔지니어링 시작...")

        # Map industry codes to categories
        df['통합업종카테고리'] = df['서비스_업종_코드'].map(
            self.industry_mapper.INDUSTRY_MAPPING
        )

        # Revenue scale features
        df['매출규모_로그'] = np.log1p(df['당월_매출_금액'])

        # Create revenue scale categories
        revenue_bins = [0, 1_000_000, 5_000_000, 20_000_000, float('inf')]
        revenue_labels = ['소규모', '중소규모', '중규모', '대규모']
        df['매출규모_카테고리'] = pd.cut(df['당월_매출_금액'],
                                    bins=revenue_bins,
                                    labels=revenue_labels)

        # Time-based features from 기준_년분기_코드
        df['년도'] = df['기준_년분기_코드'].astype(str).str[:4].astype(int)
        df['분기'] = df['기준_년분기_코드'].astype(str).str[4:].astype(int)

        # Regional features (first 5 digits of 행정동_코드)
        df['시군구코드'] = df['행정동_코드'].astype(str).str[:5]

        logger.info(f"피처 엔지니어링 완료. 새 컬럼 수: {len(df.columns)}")

        return df

    def generate_synthetic_expenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic expense data using industry cost structures

        Args:
            df: DataFrame with sales and features

        Returns:
            DataFrame with synthetic expense breakdowns
        """
        logger.info("합성 비용 데이터 생성 시작...")

        # Create a copy for synthetic data
        synthetic_df = df.copy()

        # Initialize expense columns
        expense_categories = ['재료비', '인건비', '임대료', '기타']
        for category in expense_categories:
            synthetic_df[f'예측_{category}'] = 0.0

        # Generate synthetic expenses for each industry category
        for industry_category in self.industry_mapper.CATEGORY_MAPPING.keys():

            # Filter data for this industry category
            category_mask = synthetic_df['통합업종카테고리'] == industry_category
            category_data = synthetic_df[category_mask]

            if len(category_data) == 0:
                continue

            logger.info(f"  {industry_category}: {len(category_data):,} records")

            # Get cost structure for this category
            cost_structure = self.industry_mapper.CATEGORY_MAPPING[industry_category]['cost_structure']

            # Generate synthetic expenses with realistic variations
            for expense_category in expense_categories:
                if expense_category in cost_structure:
                    base_ratio = cost_structure[expense_category]

                    # Add realistic noise based on business characteristics
                    noise_std = self._get_noise_std(expense_category, industry_category)

                    # Generate noise (normal distribution)
                    noise = np.random.normal(0, noise_std, len(category_data))

                    # Apply noise but keep ratios positive and reasonable
                    actual_ratios = np.maximum(
                        base_ratio * (1 + noise),
                        base_ratio * 0.3  # Minimum 30% of base ratio
                    )
                    actual_ratios = np.minimum(
                        actual_ratios,
                        base_ratio * 2.0  # Maximum 200% of base ratio
                    )

                    # Calculate synthetic expenses
                    synthetic_expenses = category_data['당월_매출_금액'] * actual_ratios

                    # Update the DataFrame
                    synthetic_df.loc[category_mask, f'예측_{expense_category}'] = synthetic_expenses

        # Calculate total predicted expenses
        synthetic_df['예측_총비용'] = (
            synthetic_df['예측_재료비'] +
            synthetic_df['예측_인건비'] +
            synthetic_df['예측_임대료'] +
            synthetic_df['예측_기타']
        )

        # Calculate expense ratios for validation
        for category in expense_categories:
            ratio_col = f'{category}_비율'
            synthetic_df[ratio_col] = synthetic_df[f'예측_{category}'] / synthetic_df['당월_매출_금액']

        synthetic_df['총비용_비율'] = synthetic_df['예측_총비용'] / synthetic_df['당월_매출_금액']

        logger.info("합성 비용 데이터 생성 완료")

        return synthetic_df

    def _get_noise_std(self, expense_category: str, industry_category: str) -> float:
        """
        Get appropriate noise standard deviation for realistic variations

        Args:
            expense_category: Type of expense (재료비, 인건비, etc.)
            industry_category: Industry category

        Returns:
            Standard deviation for noise generation
        """
        # Base noise levels by expense type
        base_noise = {
            '재료비': 0.15,   # Materials can vary significantly
            '인건비': 0.10,   # Labor costs more stable
            '임대료': 0.05,   # Rent most stable
            '기타': 0.20      # Other expenses most variable
        }

        # Industry-specific multipliers
        industry_multipliers = {
            '숙박음식점업': 1.2,  # F&B has higher variation
            '도매소매업': 0.8,    # Retail more predictable
            '예술스포츠업': 1.5,  # Entertainment highly variable
            '개인서비스업': 1.1   # Personal services moderate variation
        }

        base = base_noise.get(expense_category, 0.15)
        multiplier = industry_multipliers.get(industry_category, 1.0)

        return base * multiplier

    def create_ml_dataset(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create ML training dataset from synthetic data

        Args:
            test_size: Proportion of data for testing

        Returns:
            Tuple of (training_data, test_data)
        """
        if self.synthetic_data is None:
            raise ValueError("먼저 합성 데이터를 생성하세요 (generate_training_data 호출)")

        logger.info("ML 학습 데이터셋 생성 시작...")

        # Select features for ML training
        feature_columns = [
            '당월_매출_금액',
            '통합업종카테고리',
            '행정동_코드',
            '매출규모_로그',
            '매출규모_카테고리',
            '년도',
            '분기',
            '시군구코드'
        ]

        target_columns = [
            '예측_재료비',
            '예측_인건비',
            '예측_임대료',
            '예측_기타'
        ]

        # Create feature and target DataFrames
        X = self.synthetic_data[feature_columns].copy()
        y = self.synthetic_data[target_columns].copy()

        # Encode categorical variables
        categorical_columns = ['통합업종카테고리', '매출규모_카테고리', '시군구코드']
        for col in categorical_columns:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes

        # Split into train and test
        split_index = int(len(X) * (1 - test_size))

        # Shuffle the data before splitting
        shuffled_indices = np.random.permutation(len(X))
        train_indices = shuffled_indices[:split_index]
        test_indices = shuffled_indices[split_index:]

        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_test = y.iloc[test_indices].reset_index(drop=True)

        # Combine features and targets
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        logger.info(f"데이터셋 분할 완료:")
        logger.info(f"  훈련 데이터: {len(train_data):,} rows")
        logger.info(f"  테스트 데이터: {len(test_data):,} rows")
        logger.info(f"  피처 수: {len(feature_columns)}")
        logger.info(f"  타겟 수: {len(target_columns)}")

        return train_data, test_data

    def generate_training_data(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Complete pipeline to generate synthetic training data

        Args:
            save_path: Optional path to save the generated data

        Returns:
            Generated synthetic training data
        """
        logger.info("=== 합성 학습 데이터 생성 파이프라인 시작 ===")

        # Step 1: Load raw data
        raw_data = self.load_raw_data()

        # Step 2: Clean and filter
        cleaned_data = self.clean_and_filter_data(raw_data)

        # Step 3: Feature engineering
        featured_data = self.add_feature_engineering(cleaned_data)

        # Step 4: Generate synthetic expenses
        self.synthetic_data = self.generate_synthetic_expenses(featured_data)

        # Step 5: Validation summary
        self._print_validation_summary()

        # Step 6: Save if requested
        if save_path:
            self.synthetic_data.to_csv(save_path, index=False, encoding='utf-8')
            logger.info(f"합성 데이터 저장 완료: {save_path}")

        logger.info("=== 합성 학습 데이터 생성 완료 ===")

        return self.synthetic_data

    def _print_validation_summary(self):
        """Print validation summary of synthetic data quality"""
        if self.synthetic_data is None:
            return

        logger.info("\n=== 합성 데이터 품질 검증 ===")

        # Overall statistics
        total_records = len(self.synthetic_data)
        logger.info(f"총 레코드 수: {total_records:,}")

        # Industry distribution
        industry_dist = self.synthetic_data['통합업종카테고리'].value_counts()
        logger.info(f"\n업종별 분포:")
        for industry, count in industry_dist.items():
            pct = (count / total_records) * 100
            logger.info(f"  {industry}: {count:,} ({pct:.1f}%)")

        # Average cost ratios by industry
        logger.info(f"\n업종별 평균 비용 비율:")
        expense_categories = ['재료비', '인건비', '임대료', '기타']

        for industry in industry_dist.index:
            industry_data = self.synthetic_data[
                self.synthetic_data['통합업종카테고리'] == industry
            ]
            logger.info(f"\n  {industry}:")
            for category in expense_categories:
                ratio_col = f'{category}_비율'
                if ratio_col in industry_data.columns:
                    avg_ratio = industry_data[ratio_col].mean()
                    logger.info(f"    {category}: {avg_ratio:.3f} ({avg_ratio*100:.1f}%)")

        # Overall expense ratio validation
        total_expense_ratio = self.synthetic_data['총비용_비율'].mean()
        logger.info(f"\n전체 평균 비용 비율: {total_expense_ratio:.3f} ({total_expense_ratio*100:.1f}%)")

        logger.info("=== 검증 완료 ===\n")


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator()

    # Generate synthetic training data
    synthetic_data = generator.generate_training_data(
        save_path="synthetic_training_data.csv"
    )

    # Create ML dataset
    train_data, test_data = generator.create_ml_dataset(test_size=0.2)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")