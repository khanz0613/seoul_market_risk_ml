"""
External Data Integration Module for Seoul Market Risk ML System
Handles weather, holidays, and economic indicators data from Korean APIs.
"""

import pandas as pd
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import time

from ..utils.config_loader import load_config, get_data_paths


logger = logging.getLogger(__name__)


class ExternalDataIntegrator:
    """Integrates external data sources for market analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.api_config = self.config.get('external_apis', {})
        self.data_paths = get_data_paths(self.config)
        
        # Ensure external data directory exists
        self.data_paths['external'].mkdir(parents=True, exist_ok=True)
        
    def fetch_all_external_data(self, start_year: int = 2019, end_year: int = 2024) -> Dict[str, pd.DataFrame]:
        """Fetch all external data sources."""
        logger.info(f"Fetching external data for years {start_year}-{end_year}")
        
        external_data = {}
        
        try:
            # 1. Fetch weather data
            logger.info("Fetching weather data...")
            weather_data = self._fetch_weather_data(start_year, end_year)
            external_data['weather'] = weather_data
            
            # 2. Fetch holidays data
            logger.info("Fetching holidays data...")
            holidays_data = self._fetch_holidays_data(start_year, end_year)
            external_data['holidays'] = holidays_data
            
            # 3. Fetch economic indicators
            logger.info("Fetching economic indicators...")
            economic_data = self._fetch_economic_data(start_year, end_year)
            external_data['economic'] = economic_data
            
            # 4. Save external data
            self._save_external_data(external_data)
            
            logger.info("External data integration completed successfully")
            return external_data
            
        except Exception as e:
            logger.error(f"External data integration failed: {e}")
            raise
    
    def _fetch_weather_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch weather data from Korean Meteorological Administration."""
        # Since actual API requires authentication, we'll create mock data
        # In production, this would use the real API
        logger.info("Generating mock weather data (replace with actual API in production)")
        
        dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='D')
        
        # Generate realistic weather patterns for Seoul
        weather_data = []
        for date in dates:
            # Simulate seasonal temperature patterns
            day_of_year = date.timetuple().tm_yday
            base_temp = 12 + 15 * np.cos((day_of_year - 200) * 2 * np.pi / 365)  # Seasonal variation
            
            # Add some random variation
            temperature = base_temp + np.random.normal(0, 5)
            
            # Simulate precipitation (more likely in summer)
            precip_prob = 0.3 if 150 < day_of_year < 250 else 0.15
            precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0
            
            weather_data.append({
                'date': date,
                'temperature_avg': round(temperature, 1),
                'temperature_max': round(temperature + np.random.uniform(3, 8), 1),
                'temperature_min': round(temperature - np.random.uniform(3, 8), 1),
                'precipitation': round(precipitation, 1),
                'humidity': round(50 + 30 * np.cos((day_of_year - 50) * 2 * np.pi / 365) + np.random.normal(0, 10), 1),
                'wind_speed': round(np.random.exponential(3), 1),
                'weather_condition': self._get_weather_condition(precipitation, temperature)
            })
        
        weather_df = pd.DataFrame(weather_data)
        logger.info(f"Generated weather data: {len(weather_df)} daily records")
        return weather_df
    
    def _get_weather_condition(self, precipitation: float, temperature: float) -> str:
        """Determine weather condition based on precipitation and temperature."""
        if precipitation > 20:
            return 'heavy_rain'
        elif precipitation > 5:
            return 'rain'
        elif precipitation > 0:
            return 'light_rain'
        elif temperature > 30:
            return 'hot'
        elif temperature < 0:
            return 'cold'
        elif temperature < 5:
            return 'very_cold'
        else:
            return 'clear'
    
    def _fetch_holidays_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch Korean holiday data."""
        logger.info("Generating Korean holidays data")
        
        holidays = []
        
        # Korean holidays with fixed dates
        fixed_holidays = [
            ('01-01', 'New Year'),
            ('03-01', 'Independence Movement Day'),
            ('05-01', 'Labor Day'),
            ('05-05', 'Children\'s Day'),
            ('06-06', 'Memorial Day'),
            ('08-15', 'Liberation Day'),
            ('10-03', 'National Foundation Day'),
            ('10-09', 'Hangeul Day'),
            ('12-25', 'Christmas')
        ]
        
        # Add fixed holidays for each year
        for year in range(start_year, end_year + 1):
            for month_day, name in fixed_holidays:
                holidays.append({
                    'date': pd.to_datetime(f'{year}-{month_day}'),
                    'holiday_name': name,
                    'holiday_type': 'public',
                    'is_long_weekend': False  # Simplified
                })
        
        # Add lunar calendar holidays (simplified approximations)
        lunar_holidays = self._generate_lunar_holidays(start_year, end_year)
        holidays.extend(lunar_holidays)
        
        holidays_df = pd.DataFrame(holidays).sort_values('date').reset_index(drop=True)
        
        # Add additional features
        holidays_df['month'] = holidays_df['date'].dt.month
        holidays_df['day_of_week'] = holidays_df['date'].dt.dayofweek
        holidays_df['quarter'] = holidays_df['date'].dt.quarter
        
        logger.info(f"Generated {len(holidays_df)} holiday records")
        return holidays_df
    
    def _generate_lunar_holidays(self, start_year: int, end_year: int) -> List[Dict]:
        """Generate approximate lunar holiday dates."""
        # This is a simplified approximation - in production, use proper lunar calendar library
        lunar_holidays = []
        
        # Approximate dates for major lunar holidays
        lunar_holiday_approx = {
            2019: [('02-05', 'Lunar New Year'), ('09-13', 'Chuseok')],
            2020: [('01-25', 'Lunar New Year'), ('10-01', 'Chuseok')],
            2021: [('02-12', 'Lunar New Year'), ('09-21', 'Chuseok')],
            2022: [('02-01', 'Lunar New Year'), ('09-10', 'Chuseok')],
            2023: [('01-22', 'Lunar New Year'), ('09-29', 'Chuseok')],
            2024: [('02-10', 'Lunar New Year'), ('09-17', 'Chuseok')]
        }
        
        for year in range(start_year, end_year + 1):
            if year in lunar_holiday_approx:
                for date_str, name in lunar_holiday_approx[year]:
                    lunar_holidays.append({
                        'date': pd.to_datetime(f'{year}-{date_str}'),
                        'holiday_name': name,
                        'holiday_type': 'lunar',
                        'is_long_weekend': True  # These are typically 3-day holidays
                    })
        
        return lunar_holidays
    
    def _fetch_economic_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch economic indicators from Bank of Korea."""
        logger.info("Generating mock economic indicators data")
        
        # Generate quarterly economic data
        quarters = []
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                quarter_start = pd.to_datetime(f'{year}-{quarter*3-2:02d}-01')
                quarters.append(quarter_start)
        
        economic_data = []
        for i, date in enumerate(quarters):
            # Simulate economic trends with some realism
            base_gdp_growth = 2.5 + 0.5 * np.sin(i * 0.2) + np.random.normal(0, 0.5)  # Cyclical growth
            
            # COVID impact (2020-2021)
            if date.year == 2020:
                base_gdp_growth -= 3.0
            elif date.year == 2021:
                base_gdp_growth -= 1.0
            
            economic_data.append({
                'date': date,
                'quarter': f"{date.year}Q{date.quarter}",
                'gdp_growth_rate': round(base_gdp_growth, 2),
                'inflation_rate': round(2.0 + 0.5 * np.sin(i * 0.3) + np.random.normal(0, 0.3), 2),
                'unemployment_rate': round(3.5 + 0.5 * np.cos(i * 0.25) + np.random.normal(0, 0.2), 2),
                'consumer_confidence': round(100 + 10 * np.sin(i * 0.15) + np.random.normal(0, 5), 1),
                'bank_lending_rate': round(2.5 + 1.0 * np.sin(i * 0.1) + np.random.normal(0, 0.1), 2),
                'seoul_housing_price_index': round(100 * (1.02 ** (i/4)) + np.random.normal(0, 2), 1)
            })
        
        economic_df = pd.DataFrame(economic_data)
        logger.info(f"Generated {len(economic_df)} quarterly economic records")
        return economic_df
    
    def _save_external_data(self, external_data: Dict[str, pd.DataFrame]) -> None:
        """Save external data to files."""
        for data_type, df in external_data.items():
            # Save CSV
            csv_path = self.data_paths['external'] / f'{data_type}_data.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Saved {data_type} data: {len(df)} records to {csv_path}")
            
            # Save JSON metadata
            metadata = {
                'data_type': data_type,
                'record_count': len(df),
                'date_range': {
                    'start': df['date'].min().isoformat() if 'date' in df.columns else None,
                    'end': df['date'].max().isoformat() if 'date' in df.columns else None
                },
                'columns': list(df.columns),
                'generated_timestamp': datetime.now().isoformat()
            }
            
            metadata_path = self.data_paths['external'] / f'{data_type}_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_external_data(self) -> Dict[str, pd.DataFrame]:
        """Load previously saved external data."""
        external_data = {}
        
        data_types = ['weather', 'holidays', 'economic']
        for data_type in data_types:
            csv_path = self.data_paths['external'] / f'{data_type}_data.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=['date'] if data_type != 'economic' else ['date'])
                external_data[data_type] = df
                logger.info(f"Loaded {data_type} data: {len(df)} records")
            else:
                logger.warning(f"External data file not found: {csv_path}")
        
        return external_data
    
    def merge_with_sales_data(self, sales_df: pd.DataFrame, external_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge external data with sales data."""
        logger.info("Merging external data with sales data")
        
        # Ensure sales data has proper date column
        if 'quarter_code' in sales_df.columns:
            # Convert quarter code to date (e.g., "20191" -> 2019-01-01)
            sales_df['quarter_date'] = pd.to_datetime(
                sales_df['quarter_code'].astype(str).str[:4] + '-' + 
                ((sales_df['quarter_code'].astype(str).str[4:].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
            )
        
        merged_df = sales_df.copy()
        
        # Merge weather data (aggregate to quarter)
        if 'weather' in external_data:
            weather_df = external_data['weather'].copy()
            weather_df['quarter_date'] = pd.to_datetime(weather_df['date']).dt.to_period('Q').dt.start_time
            
            weather_agg = weather_df.groupby('quarter_date').agg({
                'temperature_avg': 'mean',
                'precipitation': 'sum',
                'humidity': 'mean'
            }).reset_index()
            
            merged_df = merged_df.merge(weather_agg, on='quarter_date', how='left')
        
        # Merge holidays data (count holidays per quarter)
        if 'holidays' in external_data:
            holidays_df = external_data['holidays'].copy()
            holidays_df['quarter_date'] = pd.to_datetime(holidays_df['date']).dt.to_period('Q').dt.start_time
            
            holiday_count = holidays_df.groupby('quarter_date').size().reset_index(name='holiday_count')
            merged_df = merged_df.merge(holiday_count, on='quarter_date', how='left')
            merged_df['holiday_count'] = merged_df['holiday_count'].fillna(0)
        
        # Merge economic data
        if 'economic' in external_data:
            economic_df = external_data['economic'].rename(columns={'date': 'quarter_date'})
            merged_df = merged_df.merge(economic_df.drop('quarter', axis=1), on='quarter_date', how='left')
        
        logger.info(f"External data merged: {len(merged_df)} records with {len(merged_df.columns)} columns")
        return merged_df


# Import numpy for data generation
import numpy as np


def main():
    """Main function for testing external data integration."""
    integrator = ExternalDataIntegrator()
    
    try:
        # Fetch external data
        external_data = integrator.fetch_all_external_data()
        
        print("\n=== EXTERNAL DATA SUMMARY ===")
        for data_type, df in external_data.items():
            print(f"{data_type.upper()} DATA:")
            print(f"  Records: {len(df):,}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Columns: {list(df.columns)}")
            print()
        
        # Test loading
        loaded_data = integrator.load_external_data()
        print(f"Successfully loaded {len(loaded_data)} external data types")
        
    except Exception as e:
        logger.error(f"External data integration failed: {e}")
        raise


if __name__ == "__main__":
    main()