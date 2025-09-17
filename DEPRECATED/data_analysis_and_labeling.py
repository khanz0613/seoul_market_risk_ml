#!/usr/bin/env python3
"""
Seoul Commercial District Data Analysis & ML Labeling
======================================================

Analyzes 6 CSV files (2019-2024) containing 342,555 Seoul commercial district records
and creates risk labels for supervised ML training.

Author: Seoul Market Risk ML System
Date: 2025-09-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SeoulCommercialDataAnalyzer:
    """Comprehensive analyzer for Seoul commercial district data"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.raw_data = None
        self.analysis_results = {}

    def load_all_data(self) -> pd.DataFrame:
        """Load and combine all CSV files from 2019-2024"""
        print("📊 Loading Seoul Commercial District Data (2019-2024)")
        print("=" * 60)

        all_dataframes = []

        # Get all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        csv_files.sort()  # Sort by filename for consistent order

        for file_path in csv_files:
            if file_path.name.startswith('.'):  # Skip hidden files
                continue

            print(f"Loading: {file_path.name}")

            try:
                # Try different encodings
                for encoding in ['utf-8', 'euc-kr', 'cp949']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError("Could not decode file with any encoding")

                # Add year column based on filename
                year = self._extract_year_from_filename(file_path.name)
                df['데이터연도'] = year
                df['파일명'] = file_path.name

                print(f"  → {len(df):,} records loaded (Year: {year})")
                all_dataframes.append(df)

            except Exception as e:
                print(f"  ❌ Error loading {file_path.name}: {e}")

        # Combine all dataframes
        if all_dataframes:
            self.raw_data = pd.concat(all_dataframes, ignore_index=True, sort=False)
            print(f"\n🎯 Total Combined Records: {len(self.raw_data):,}")
            print(f"📈 Data Shape: {self.raw_data.shape}")
        else:
            raise ValueError("No data files could be loaded")

        return self.raw_data

    def _extract_year_from_filename(self, filename: str) -> int:
        """Extract year from Seoul data filename"""
        import re
        year_match = re.search(r'(\d{4})', filename)
        return int(year_match.group(1)) if year_match else 2024

    def analyze_data_structure(self) -> Dict:
        """Comprehensive data structure analysis"""
        print("\n🔍 Data Structure Analysis")
        print("=" * 40)

        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_all_data() first")

        analysis = {
            'total_records': len(self.raw_data),
            'columns': list(self.raw_data.columns),
            'column_count': len(self.raw_data.columns),
            'data_types': self.raw_data.dtypes.to_dict(),
            'memory_usage': self.raw_data.memory_usage(deep=True).sum(),
            'year_distribution': self.raw_data['데이터연도'].value_counts().to_dict(),
        }

        # Print key findings
        print(f"Total Records: {analysis['total_records']:,}")
        print(f"Columns: {analysis['column_count']}")
        print(f"Memory Usage: {analysis['memory_usage'] / 1024**2:.1f} MB")

        print("\n📅 Year Distribution:")
        for year, count in sorted(analysis['year_distribution'].items()):
            print(f"  {year}: {count:,} records")

        print(f"\n📋 Column Names ({len(analysis['columns'])} total):")
        for i, col in enumerate(analysis['columns'][:20], 1):  # Show first 20
            print(f"  {i:2d}. {col}")
        if len(analysis['columns']) > 20:
            print(f"      ... and {len(analysis['columns']) - 20} more columns")

        self.analysis_results['structure'] = analysis
        return analysis

    def analyze_business_types(self) -> Dict:
        """Analyze business type distribution"""
        print("\n🏪 Business Type Analysis")
        print("=" * 30)

        # Find business type column (could be different names)
        business_cols = [col for col in self.raw_data.columns
                        if any(keyword in col.lower() for keyword in ['업종', 'business', '사업'])]

        if not business_cols:
            print("❌ No business type column found")
            return {}

        business_col = business_cols[0]  # Use first match
        print(f"Business Type Column: {business_col}")

        business_dist = self.raw_data[business_col].value_counts()

        print(f"\nTop 15 Business Types (Total: {len(business_dist)} types):")
        for i, (business, count) in enumerate(business_dist.head(15).items(), 1):
            pct = (count / len(self.raw_data)) * 100
            print(f"  {i:2d}. {business}: {count:,} ({pct:.1f}%)")

        analysis = {
            'business_column': business_col,
            'total_types': len(business_dist),
            'distribution': business_dist.to_dict(),
            'top_10': business_dist.head(10).to_dict()
        }

        self.analysis_results['business_types'] = analysis
        return analysis

    def analyze_regions(self) -> Dict:
        """Analyze regional distribution"""
        print("\n🗺️ Regional Analysis")
        print("=" * 20)

        # Find region-related columns
        region_cols = [col for col in self.raw_data.columns
                      if any(keyword in col.lower() for keyword in ['구', '동', '시', 'district', 'region'])]

        if not region_cols:
            print("❌ No region columns found")
            return {}

        print(f"Region Columns Found: {region_cols}")

        analysis = {}

        for col in region_cols[:3]:  # Analyze top 3 region columns
            print(f"\n📍 {col} Distribution:")
            region_dist = self.raw_data[col].value_counts()

            print(f"  Total {col}: {len(region_dist)}")
            print("  Top 10:")
            for i, (region, count) in enumerate(region_dist.head(10).items(), 1):
                pct = (count / len(self.raw_data)) * 100
                print(f"    {i:2d}. {region}: {count:,} ({pct:.1f}%)")

            analysis[col] = {
                'total_count': len(region_dist),
                'distribution': region_dist.to_dict(),
                'top_10': region_dist.head(10).to_dict()
            }

        self.analysis_results['regions'] = analysis
        return analysis

    def analyze_financial_columns(self) -> Dict:
        """Analyze financial/revenue related columns"""
        print("\n💰 Financial Data Analysis")
        print("=" * 28)

        # Find financial columns
        financial_keywords = ['매출', '수입', '매상', 'revenue', '금액', '원', '비용', 'cost']
        financial_cols = [col for col in self.raw_data.columns
                         if any(keyword in col.lower() for keyword in financial_keywords)]

        if not financial_cols:
            print("❌ No financial columns found")
            return {}

        print(f"Financial Columns Found ({len(financial_cols)}):")
        for col in financial_cols:
            print(f"  • {col}")

        analysis = {}

        # Analyze numeric financial columns
        for col in financial_cols[:5]:  # Analyze top 5
            if self.raw_data[col].dtype in ['int64', 'float64']:
                stats = self.raw_data[col].describe()

                print(f"\n📊 {col} Statistics:")
                print(f"  Count: {stats['count']:,.0f}")
                print(f"  Mean: {stats['mean']:,.0f}")
                print(f"  Median: {stats['50%']:,.0f}")
                print(f"  Min: {stats['min']:,.0f}")
                print(f"  Max: {stats['max']:,.0f}")

                analysis[col] = {
                    'type': 'numeric',
                    'stats': stats.to_dict(),
                    'non_null_count': self.raw_data[col].notna().sum(),
                    'null_count': self.raw_data[col].isna().sum()
                }

        self.analysis_results['financial'] = analysis
        return analysis

    def create_risk_labels(self) -> pd.Series:
        """
        Create 5-level risk labels for supervised ML training

        Risk Levels:
        1 = 매우여유 (Very Comfortable)
        2 = 여유 (Comfortable)
        3 = 보통 (Average)
        4 = 위험 (Risky)
        5 = 매우위험 (Very Risky)
        """
        print("\n🎯 Creating Risk Labels for ML Training")
        print("=" * 45)

        if self.raw_data is None:
            raise ValueError("Data not loaded")

        # Find key financial columns for risk assessment
        revenue_cols = [col for col in self.raw_data.columns
                       if '매출' in col or 'revenue' in col.lower()]

        if not revenue_cols:
            print("❌ No revenue column found for risk labeling")
            return pd.Series([3] * len(self.raw_data))  # Default to "보통"

        revenue_col = revenue_cols[0]  # Use primary revenue column
        print(f"Using revenue column: {revenue_col}")

        # Calculate risk based on multiple factors
        risk_labels = []

        # Get business and region for benchmarking
        business_col = None
        region_col = None

        for col in self.raw_data.columns:
            if '업종' in col and business_col is None:
                business_col = col
            if '구' in col and region_col is None:
                region_col = col

        print(f"Business column: {business_col}")
        print(f"Region column: {region_col}")

        for idx, row in self.raw_data.iterrows():
            if idx % 50000 == 0:  # Progress indicator
                print(f"  Processing: {idx:,}/{len(self.raw_data):,}")

            revenue = row[revenue_col]
            business = row[business_col] if business_col else "기타"
            region = row[region_col] if region_col else "기타"
            year = row['데이터연도']

            # Calculate risk score based on multiple factors
            risk_score = self._calculate_risk_score(revenue, business, region, year)

            # Convert risk score to 5-level classification
            if risk_score <= 0.2:
                risk_level = 1  # 매우여유
            elif risk_score <= 0.4:
                risk_level = 2  # 여유
            elif risk_score <= 0.6:
                risk_level = 3  # 보통
            elif risk_score <= 0.8:
                risk_level = 4  # 위험
            else:
                risk_level = 5  # 매우위험

            risk_labels.append(risk_level)

        risk_series = pd.Series(risk_labels, name='risk_label')

        # Print distribution
        risk_dist = risk_series.value_counts().sort_index()
        print(f"\n📊 Risk Label Distribution:")
        risk_names = {1: "매우여유", 2: "여유", 3: "보통", 4: "위험", 5: "매우위험"}

        for level, count in risk_dist.items():
            pct = (count / len(risk_series)) * 100
            print(f"  {level}={risk_names[level]}: {count:,} ({pct:.1f}%)")

        return risk_series

    def _calculate_risk_score(self, revenue: float, business: str, region: str, year: int) -> float:
        """Calculate risk score between 0.0 and 1.0"""
        try:
            # Handle missing/invalid revenue
            if pd.isna(revenue) or revenue <= 0:
                return 0.6  # Default moderate risk

            # Industry benchmark (simplified)
            industry_avg = self._get_industry_benchmark(business)
            industry_ratio = revenue / industry_avg if industry_avg > 0 else 1.0

            # Regional benchmark (simplified)
            regional_avg = self._get_regional_benchmark(region)
            regional_ratio = revenue / regional_avg if regional_avg > 0 else 1.0

            # Time factor (economic cycles)
            time_factor = self._get_time_factor(year)

            # Combined risk score calculation
            base_score = max(0.0, min(1.0, 0.5 - (industry_ratio - 1.0) * 0.2))
            regional_adjustment = max(-0.2, min(0.2, (1.0 - regional_ratio) * 0.1))
            time_adjustment = time_factor * 0.1

            final_score = max(0.0, min(1.0, base_score + regional_adjustment + time_adjustment))

            return final_score

        except:
            return 0.6  # Default moderate risk on error

    def _get_industry_benchmark(self, business: str) -> float:
        """Get industry average revenue (simplified)"""
        # Industry benchmarks (estimated values for demo)
        industry_benchmarks = {
            "한식음식점": 15000000,
            "카페": 8000000,
            "편의점": 25000000,
            "미용실": 5000000,
            "의류": 12000000,
            "치킨": 18000000,
            "피자": 16000000,
            "패스트푸드": 20000000,
            "기타": 10000000
        }

        # Find best match
        for key in industry_benchmarks:
            if key in str(business):
                return industry_benchmarks[key]

        return industry_benchmarks["기타"]

    def _get_regional_benchmark(self, region: str) -> float:
        """Get regional average revenue (simplified)"""
        # Regional benchmarks (estimated values for demo)
        regional_benchmarks = {
            "강남구": 18000000,
            "서초구": 16000000,
            "마포구": 14000000,
            "종로구": 15000000,
            "영등포구": 13000000,
            "기타": 12000000
        }

        # Find best match
        for key in regional_benchmarks:
            if key in str(region):
                return regional_benchmarks[key]

        return regional_benchmarks["기타"]

    def _get_time_factor(self, year: int) -> float:
        """Get economic time factor"""
        # Economic factors by year (simplified)
        time_factors = {
            2019: 0.0,    # Pre-COVID baseline
            2020: 0.4,    # COVID impact
            2021: 0.3,    # Recovery phase
            2022: 0.1,    # Recovery continues
            2023: -0.1,   # Normalization
            2024: -0.2    # Growth phase
        }

        return time_factors.get(year, 0.0)

    def save_analysis_results(self, output_dir: str = "ml_analysis_results") -> None:
        """Save analysis results and labeled data"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n💾 Saving Analysis Results to {output_dir}/")
        print("=" * 50)

        # Save labeled dataset for ML training
        if self.raw_data is not None:
            # Add risk labels
            risk_labels = self.create_risk_labels()
            labeled_data = self.raw_data.copy()
            labeled_data['risk_label'] = risk_labels

            # Save full labeled dataset
            labeled_file = output_path / "seoul_commercial_labeled_dataset.csv"
            labeled_data.to_csv(labeled_file, index=False, encoding='utf-8')
            print(f"✅ Labeled Dataset: {labeled_file}")
            print(f"   Records: {len(labeled_data):,}")
            print(f"   Size: {labeled_file.stat().st_size / 1024**2:.1f} MB")

        # Save analysis summary
        summary_file = output_path / "data_analysis_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Seoul Commercial District Data Analysis Summary\n")
            f.write("=" * 50 + "\n\n")

            for analysis_type, results in self.analysis_results.items():
                f.write(f"{analysis_type.upper()} ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"{results}\n\n")

        print(f"✅ Analysis Summary: {summary_file}")
        print(f"🎯 Ready for ML model training!")

def main():
    """Main execution function"""
    print("🚀 Seoul Commercial District ML Data Analysis")
    print("=" * 55)

    # Initialize analyzer
    analyzer = SeoulCommercialDataAnalyzer()

    try:
        # Step 1: Load all data
        data = analyzer.load_all_data()

        # Step 2: Comprehensive analysis
        analyzer.analyze_data_structure()
        analyzer.analyze_business_types()
        analyzer.analyze_regions()
        analyzer.analyze_financial_columns()

        # Step 3: Save results for ML training
        analyzer.save_analysis_results()

        print(f"\n✅ Analysis Complete!")
        print(f"📊 Total Records Analyzed: {len(data):,}")
        print(f"🎯 Ready for Feature Engineering & ML Training")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()