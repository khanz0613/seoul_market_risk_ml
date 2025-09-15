"""
Data Loading Module for Seoul Market Risk ML System
Handles CSV file loading with proper encoding detection and conversion.
"""

import pandas as pd
import chardet
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)


class SeoulDataLoader:
    """Seoul Commercial Area Data Loader with encoding handling."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        self.raw_data_path = Path(self.data_config['raw_data_path'])
        
    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        
        confidence = result['confidence']
        encoding = result['encoding']
        
        logger.info(f"Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2f})")
        
        # Special handling for 2019 data (known to be EUC-KR)
        if '2019' in str(file_path) and confidence < 0.8:
            logger.warning(f"Low confidence detection for 2019 file, using EUC-KR")
            return 'euc-kr'
            
        return encoding.lower() if encoding else 'utf-8'
    
    def load_single_csv(self, file_path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        """Load a single CSV file with proper encoding."""
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        
        try:
            logger.info(f"Loading {file_path.name} with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            
            # Add metadata
            df['source_file'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            df['load_timestamp'] = datetime.now()
            
            logger.info(f"Successfully loaded {len(df):,} rows from {file_path.name}")
            return df
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error for {file_path.name}: {e}")
            # Fallback to EUC-KR for problematic files
            if encoding != 'euc-kr':
                logger.info("Retrying with EUC-KR encoding...")
                return self.load_single_csv(file_path, 'euc-kr')
            raise
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            raise
    
    def _extract_year_from_filename(self, filename: str) -> int:
        """Extract year from filename."""
        import re
        year_match = re.search(r'(\d{4})', filename)
        if year_match:
            return int(year_match.group(1))
        else:
            logger.warning(f"Could not extract year from filename: {filename}")
            return None
    
    def load_all_csv_files(self) -> Dict[int, pd.DataFrame]:
        """Load all CSV files and return dictionary by year."""
        csv_files = list(self.raw_data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_data_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        data_by_year = {}
        for file_path in sorted(csv_files):
            try:
                df = self.load_single_csv(file_path)
                year = df['year'].iloc[0]
                data_by_year[year] = df
                
                logger.info(f"Year {year}: {len(df):,} rows, {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                continue
        
        return data_by_year
    
    def get_schema_info(self, data_by_year: Dict[int, pd.DataFrame]) -> Dict:
        """Analyze schema consistency across years."""
        schema_info = {
            'years': list(data_by_year.keys()),
            'total_rows': sum(len(df) for df in data_by_year.values()),
            'columns_by_year': {},
            'common_columns': None,
            'schema_changes': []
        }
        
        # Analyze columns for each year
        all_columns_sets = []
        for year, df in data_by_year.items():
            columns = list(df.columns)
            schema_info['columns_by_year'][year] = {
                'columns': columns,
                'count': len(columns),
                'dtypes': df.dtypes.to_dict()
            }
            all_columns_sets.append(set(columns))
        
        # Find common columns
        if all_columns_sets:
            schema_info['common_columns'] = list(set.intersection(*all_columns_sets))
        
        # Detect schema changes
        if len(all_columns_sets) > 1:
            base_columns = all_columns_sets[0]
            for i, year_columns in enumerate(all_columns_sets[1:], 1):
                year = list(data_by_year.keys())[i]
                added = year_columns - base_columns
                removed = base_columns - year_columns
                
                if added or removed:
                    schema_info['schema_changes'].append({
                        'year': year,
                        'added_columns': list(added),
                        'removed_columns': list(removed)
                    })
        
        logger.info(f"Schema analysis complete: {len(schema_info['common_columns'])} common columns")
        return schema_info
    
    def validate_data_quality(self, data_by_year: Dict[int, pd.DataFrame]) -> Dict:
        """Validate data quality across all years."""
        quality_report = {
            'total_records': 0,
            'issues_by_year': {},
            'overall_issues': {
                'missing_values': 0,
                'duplicate_rows': 0,
                'invalid_dates': 0,
                'negative_sales': 0
            }
        }
        
        for year, df in data_by_year.items():
            year_issues = {
                'row_count': len(df),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'columns_with_nulls': df.isnull().any().sum(),
                'negative_sales_columns': 0
            }
            
            # Check for negative sales values
            sales_columns = [col for col in df.columns if '매출' in col and '금액' in col]
            for col in sales_columns:
                if (df[col] < 0).any():
                    year_issues['negative_sales_columns'] += 1
            
            quality_report['issues_by_year'][year] = year_issues
            quality_report['total_records'] += len(df)
        
        # Aggregate overall issues
        for year_data in quality_report['issues_by_year'].values():
            quality_report['overall_issues']['missing_values'] += year_data['missing_values']
            quality_report['overall_issues']['duplicate_rows'] += year_data['duplicate_rows']
            quality_report['overall_issues']['negative_sales'] += year_data['negative_sales_columns']
        
        logger.info(f"Data quality validation complete for {quality_report['total_records']:,} total records")
        return quality_report


def main():
    """Main function for testing data loading."""
    loader = SeoulDataLoader()
    
    try:
        # Load all data
        data_by_year = loader.load_all_csv_files()
        
        # Analyze schema
        schema_info = loader.get_schema_info(data_by_year)
        print("\n=== SCHEMA ANALYSIS ===")
        print(f"Years: {schema_info['years']}")
        print(f"Total rows: {schema_info['total_rows']:,}")
        print(f"Common columns: {len(schema_info['common_columns'])}")
        
        # Validate data quality
        quality_report = loader.validate_data_quality(data_by_year)
        print("\n=== DATA QUALITY REPORT ===")
        print(f"Total records: {quality_report['total_records']:,}")
        print(f"Missing values: {quality_report['overall_issues']['missing_values']:,}")
        print(f"Duplicate rows: {quality_report['overall_issues']['duplicate_rows']:,}")
        
        # Save processed data
        for year, df in data_by_year.items():
            output_path = Path(f"data/processed/seoul_sales_{year}.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Saved {year} data to {output_path}")
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise


if __name__ == "__main__":
    main()