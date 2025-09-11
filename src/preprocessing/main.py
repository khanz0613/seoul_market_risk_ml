"""
Main Data Preprocessing Pipeline for Seoul Market Risk ML System
Handles encoding conversion, data validation, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json

from .data_loader import SeoulDataLoader
from ..utils.config_loader import load_config, setup_logging, get_data_paths


logger = logging.getLogger(__name__)


class SeoulDataPreprocessor:
    """Main data preprocessing pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.data_paths = get_data_paths(self.config)
        self.loader = SeoulDataLoader(config_path)
        
        # Setup logging
        setup_logging(self.config)
        
    def run_preprocessing_pipeline(self) -> Dict[str, any]:
        """Run the complete preprocessing pipeline."""
        logger.info("Starting Seoul Market Risk ML preprocessing pipeline...")
        
        pipeline_results = {
            'start_time': datetime.now(),
            'steps_completed': [],
            'data_summary': {},
            'quality_report': {},
            'errors': []
        }
        
        try:
            # Step 1: Load raw data with encoding handling
            logger.info("Step 1: Loading raw data...")
            data_by_year = self.loader.load_all_csv_files()
            pipeline_results['steps_completed'].append('data_loading')
            pipeline_results['data_summary']['years_loaded'] = list(data_by_year.keys())
            pipeline_results['data_summary']['total_records'] = sum(len(df) for df in data_by_year.values())
            
            # Step 2: Schema analysis
            logger.info("Step 2: Analyzing data schema...")
            schema_info = self.loader.get_schema_info(data_by_year)
            pipeline_results['steps_completed'].append('schema_analysis')
            pipeline_results['data_summary']['schema'] = schema_info
            
            # Step 3: Data quality validation
            logger.info("Step 3: Validating data quality...")
            quality_report = self.loader.validate_data_quality(data_by_year)
            pipeline_results['steps_completed'].append('quality_validation')
            pipeline_results['quality_report'] = quality_report
            
            # Step 4: Data cleaning and standardization
            logger.info("Step 4: Cleaning and standardizing data...")
            cleaned_data = self._clean_and_standardize_data(data_by_year)
            pipeline_results['steps_completed'].append('data_cleaning')
            
            # Step 5: Feature engineering preparation
            logger.info("Step 5: Preparing feature engineering...")
            engineered_data = self._prepare_features(cleaned_data)
            pipeline_results['steps_completed'].append('feature_preparation')
            
            # Step 6: Save processed data
            logger.info("Step 6: Saving processed data...")
            saved_files = self._save_processed_data(engineered_data)
            pipeline_results['steps_completed'].append('data_saving')
            pipeline_results['data_summary']['saved_files'] = saved_files
            
            # Step 7: Generate preprocessing report
            logger.info("Step 7: Generating preprocessing report...")
            report_path = self._generate_preprocessing_report(pipeline_results)
            pipeline_results['steps_completed'].append('report_generation')
            pipeline_results['report_path'] = str(report_path)
            
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration'] = (pipeline_results['end_time'] - pipeline_results['start_time']).total_seconds()
            
            logger.info(f"Preprocessing pipeline completed successfully in {pipeline_results['duration']:.1f} seconds")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            pipeline_results['errors'].append(str(e))
            pipeline_results['end_time'] = datetime.now()
            raise
    
    def _clean_and_standardize_data(self, data_by_year: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Clean and standardize data across all years."""
        cleaned_data = {}
        
        for year, df in data_by_year.items():
            logger.info(f"Cleaning data for year {year}...")
            
            # Create a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # 1. Standardize column names
            cleaned_df.columns = self._standardize_column_names(cleaned_df.columns)
            
            # 2. Handle missing values
            cleaned_df = self._handle_missing_values(cleaned_df)
            
            # 3. Fix data types
            cleaned_df = self._fix_data_types(cleaned_df)
            
            # 4. Handle outliers
            cleaned_df = self._handle_outliers(cleaned_df)
            
            # 5. Add derived columns
            cleaned_df = self._add_derived_columns(cleaned_df, year)
            
            cleaned_data[year] = cleaned_df
            logger.info(f"Year {year} cleaning complete: {len(cleaned_df):,} rows")
        
        return cleaned_data
    
    def _standardize_column_names(self, columns: pd.Index) -> List[str]:
        """Standardize column names for consistency."""
        # Create mapping for Korean column names to English
        column_mapping = {
            '기준_년분기_코드': 'quarter_code',
            '행정동_코드': 'district_code', 
            '행정동_코드_명': 'district_name',
            '서비스_업종_코드': 'business_type_code',
            '서비스_업종_코드_명': 'business_type_name',
            '당월_매출_금액': 'monthly_revenue',
            '당월_매출_건수': 'monthly_transactions',
            '주중_매출_금액': 'weekday_revenue',
            '주말_매출_금액': 'weekend_revenue',
        }
        
        # Add day-specific mappings
        days = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        day_names_en = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for korean_day, english_day in zip(days, day_names_en):
            column_mapping[f'{korean_day}_매출_금액'] = f'{english_day}_revenue'
            column_mapping[f'{korean_day}_매출_건수'] = f'{english_day}_transactions'
        
        # Add time period mappings
        time_periods = ['00~06', '06~11', '11~14', '14~17', '17~21', '21~24']
        time_names = ['night', 'morning', 'lunch', 'afternoon', 'evening', 'late_night']
        for korean_time, english_time in zip(time_periods, time_names):
            column_mapping[f'시간대_{korean_time}_매출_금액'] = f'{english_time}_revenue'
            # Handle inconsistent column naming in the data
            column_mapping[f'시간대_건수~{korean_time.split("~")[1]}_매출_건수'] = f'{english_time}_transactions'
        
        # Add demographic mappings
        column_mapping.update({
            '남성_매출_금액': 'male_revenue',
            '여성_매출_금액': 'female_revenue',
            '남성_매출_건수': 'male_transactions', 
            '여성_매출_건수': 'female_transactions'
        })
        
        # Add age group mappings
        age_groups = ['10', '20', '30', '40', '50', '60_이상']
        age_names = ['age_10s', 'age_20s', 'age_30s', 'age_40s', 'age_50s', 'age_60plus']
        for korean_age, english_age in zip(age_groups, age_names):
            column_mapping[f'연령대_{korean_age}_매출_금액'] = f'{english_age}_revenue'
            column_mapping[f'연령대_{korean_age}_매출_건수'] = f'{english_age}_transactions'
        
        # Apply mapping
        standardized_columns = []
        for col in columns:
            if col in column_mapping:
                standardized_columns.append(column_mapping[col])
            else:
                # Keep original column name if no mapping found
                standardized_columns.append(col)
        
        return standardized_columns
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately."""
        # Fill numeric columns with 0 (makes sense for sales data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Fill categorical columns with 'Unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for consistency."""
        # Revenue and transaction columns should be numeric
        revenue_cols = [col for col in df.columns if 'revenue' in col or 'transactions' in col]
        for col in revenue_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Code columns should be string
        code_cols = [col for col in df.columns if 'code' in col]
        for col in code_cols:
            df[col] = df[col].astype(str)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme outliers in revenue data."""
        revenue_columns = [col for col in df.columns if 'revenue' in col]
        
        for col in revenue_columns:
            # Use IQR method to cap outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds (3 * IQR method)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap outliers
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0:
                logger.info(f"Capping {outlier_count} outliers in {col}")
                df[col] = df[col].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Add derived columns for analysis."""
        # Add quarter information
        if 'quarter_code' in df.columns:
            df['year'] = year
            df['quarter'] = df['quarter_code'].astype(str).str[-1].astype(int)
        
        # Add total revenue calculations
        revenue_cols = [col for col in df.columns if 'revenue' in col and col != 'monthly_revenue']
        if revenue_cols:
            df['calculated_total_revenue'] = df[revenue_cols].sum(axis=1)
        
        # Add revenue ratios
        if 'weekday_revenue' in df.columns and 'weekend_revenue' in df.columns:
            total_week_revenue = df['weekday_revenue'] + df['weekend_revenue']
            df['weekday_ratio'] = np.where(total_week_revenue > 0, 
                                         df['weekday_revenue'] / total_week_revenue, 0)
        
        # Add processing timestamp
        df['processed_timestamp'] = datetime.now()
        
        return df
    
    def _prepare_features(self, cleaned_data: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Prepare features for ML pipeline."""
        logger.info("Preparing features for ML pipeline...")
        
        prepared_data = {}
        for year, df in cleaned_data.items():
            # Sort by district and business type for consistent ordering
            df_sorted = df.sort_values(['district_code', 'business_type_code', 'quarter_code'])
            
            # Add feature engineering placeholders
            df_sorted['feature_engineering_ready'] = True
            
            prepared_data[year] = df_sorted
            
        logger.info("Feature preparation complete")
        return prepared_data
    
    def _save_processed_data(self, processed_data: Dict[int, pd.DataFrame]) -> List[str]:
        """Save processed data to files."""
        saved_files = []
        
        # Ensure processed data directory exists
        self.data_paths['processed'].mkdir(parents=True, exist_ok=True)
        
        for year, df in processed_data.items():
            # Save individual year files
            year_file = self.data_paths['processed'] / f'seoul_sales_{year}.csv'
            df.to_csv(year_file, index=False, encoding='utf-8')
            saved_files.append(str(year_file))
            logger.info(f"Saved {year} data: {len(df):,} rows to {year_file}")
        
        # Save combined data
        combined_df = pd.concat(processed_data.values(), ignore_index=True)
        combined_file = self.data_paths['processed'] / 'seoul_sales_combined.csv'
        combined_df.to_csv(combined_file, index=False, encoding='utf-8')
        saved_files.append(str(combined_file))
        logger.info(f"Saved combined data: {len(combined_df):,} rows to {combined_file}")
        
        return saved_files
    
    def _generate_preprocessing_report(self, results: Dict) -> Path:
        """Generate comprehensive preprocessing report."""
        report_path = self.data_paths['processed'] / 'preprocessing_report.json'
        
        # Convert datetime objects to strings for JSON serialization
        json_results = results.copy()
        json_results['start_time'] = results['start_time'].isoformat()
        json_results['end_time'] = results['end_time'].isoformat()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Preprocessing report saved to {report_path}")
        return report_path


def main():
    """Main function to run preprocessing pipeline."""
    try:
        preprocessor = SeoulDataPreprocessor()
        results = preprocessor.run_preprocessing_pipeline()
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Duration: {results['duration']:.1f} seconds")
        print(f"Years processed: {results['data_summary']['years_loaded']}")
        print(f"Total records: {results['data_summary']['total_records']:,}")
        print(f"Files saved: {len(results['data_summary']['saved_files'])}")
        print(f"Report: {results['report_path']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()