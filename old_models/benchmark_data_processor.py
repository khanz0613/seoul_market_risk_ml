#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/raw 기반 업종/지역별 벤치마크 데이터 처리기
핵심 기능: 사용자 지출 구조를 동일 조건 평균과 비교
"""

import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BenchmarkDataProcessor:
    def __init__(self):
        """벤치마크 데이터 처리기 초기화"""
        self.benchmarks = {}
        self.industry_coefficients = self._create_industry_coefficients()
        self.load_all_benchmarks()

    def _create_industry_coefficients(self) -> Dict:
        """업종별 재무 계수 (Altman Z-Score 계산용)"""
        return {
            # 음식점업
            '한식음식점': {
                'asset_turnover': 2.5,      # 자산회전율
                'debt_ratio': 0.65,         # 부채비율
                'profit_margin': 0.08,      # 순이익률
                'retention_ratio': 0.6,     # 이익 유보율
                'expense_structure': {      # 지출 구조 평균
                    'labor_ratio': 0.35,    # 인건비 비율
                    'material_ratio': 0.30, # 식자재 비율
                    'rent_ratio': 0.15,     # 임대료 비율
                    'others_ratio': 0.20    # 기타 비율
                }
            },
            '중식음식점': {
                'asset_turnover': 2.3,
                'debt_ratio': 0.67,
                'profit_margin': 0.07,
                'retention_ratio': 0.55,
                'expense_structure': {
                    'labor_ratio': 0.32,
                    'material_ratio': 0.35,
                    'rent_ratio': 0.15,
                    'others_ratio': 0.18
                }
            },
            '일식음식점': {
                'asset_turnover': 2.1,
                'debt_ratio': 0.60,
                'profit_margin': 0.12,
                'retention_ratio': 0.7,
                'expense_structure': {
                    'labor_ratio': 0.38,
                    'material_ratio': 0.35,
                    'rent_ratio': 0.12,
                    'others_ratio': 0.15
                }
            },
            '양식음식점': {
                'asset_turnover': 2.0,
                'debt_ratio': 0.62,
                'profit_margin': 0.10,
                'retention_ratio': 0.65,
                'expense_structure': {
                    'labor_ratio': 0.40,
                    'material_ratio': 0.32,
                    'rent_ratio': 0.13,
                    'others_ratio': 0.15
                }
            },
            '카페': {
                'asset_turnover': 3.0,
                'debt_ratio': 0.55,
                'profit_margin': 0.15,
                'retention_ratio': 0.8,
                'expense_structure': {
                    'labor_ratio': 0.45,
                    'material_ratio': 0.25,
                    'rent_ratio': 0.18,
                    'others_ratio': 0.12
                }
            },
            '커피전문점': {
                'asset_turnover': 2.8,
                'debt_ratio': 0.58,
                'profit_margin': 0.12,
                'retention_ratio': 0.75,
                'expense_structure': {
                    'labor_ratio': 0.42,
                    'material_ratio': 0.28,
                    'rent_ratio': 0.17,
                    'others_ratio': 0.13
                }
            },
            '치킨전문점': {
                'asset_turnover': 3.2,
                'debt_ratio': 0.70,
                'profit_margin': 0.06,
                'retention_ratio': 0.5,
                'expense_structure': {
                    'labor_ratio': 0.30,
                    'material_ratio': 0.40,
                    'rent_ratio': 0.15,
                    'others_ratio': 0.15
                }
            },
            '기타': {
                'asset_turnover': 2.5,
                'debt_ratio': 0.65,
                'profit_margin': 0.08,
                'retention_ratio': 0.6,
                'expense_structure': {
                    'labor_ratio': 0.35,
                    'material_ratio': 0.30,
                    'rent_ratio': 0.15,
                    'others_ratio': 0.20
                }
            }
        }

    def load_all_benchmarks(self):
        """data/raw 파일들에서 벤치마크 데이터 계산"""
        print("📊 벤치마크 데이터 로딩 중...")

        # CSV 파일들 로드
        files = glob.glob('data/raw/서울시*상권분석서비스*.csv')
        if not files:
            print("⚠️ data/raw 폴더에 CSV 파일이 없습니다.")
            return

        combined_data = []
        for file in files:
            try:
                df = pd.read_csv(file, encoding='utf-8')
                combined_data.append(df)
            except Exception as e:
                print(f"❌ 파일 로드 실패: {file} - {e}")
                continue

        if not combined_data:
            print("❌ 로드된 데이터가 없습니다.")
            return

        # 데이터 합치기
        df = pd.concat(combined_data, ignore_index=True)
        print(f"✅ 총 {len(df):,}개 레코드 로드")

        # 벤치마크 계산
        self._calculate_industry_benchmarks(df)
        self._calculate_location_benchmarks(df)

        print(f"✅ 벤치마크 계산 완료 - 업종: {len(self.benchmarks.get('industry', {}))}, 지역: {len(self.benchmarks.get('location', {}))}")

    def _calculate_industry_benchmarks(self, df: pd.DataFrame):
        """업종별 평균 매출/지출 벤치마크 계산"""
        if '서비스_업종_코드_명' not in df.columns:
            print("⚠️ 업종 정보 컬럼이 없습니다.")
            return

        industry_stats = df.groupby('서비스_업종_코드_명').agg({
            '당월_매출_금액': ['mean', 'median', 'std', 'count'],
            '추정지출금액': ['mean', 'median', 'std', 'count']
        }).round(0)

        # 평탄화하고 딕셔너리로 변환
        industry_benchmarks = {}
        for industry in industry_stats.index:
            if industry_stats.loc[industry, ('당월_매출_금액', 'count')] >= 10:  # 최소 10개 샘플
                industry_benchmarks[industry] = {
                    'revenue_mean': industry_stats.loc[industry, ('당월_매출_금액', 'mean')],
                    'revenue_median': industry_stats.loc[industry, ('당월_매출_금액', 'median')],
                    'revenue_std': industry_stats.loc[industry, ('당월_매출_금액', 'std')],
                    'expense_mean': industry_stats.loc[industry, ('추정지출금액', 'mean')],
                    'expense_median': industry_stats.loc[industry, ('추정지출금액', 'median')],
                    'expense_std': industry_stats.loc[industry, ('추정지출금액', 'std')],
                    'sample_count': industry_stats.loc[industry, ('당월_매출_금액', 'count')]
                }

        self.benchmarks['industry'] = industry_benchmarks

    def _calculate_location_benchmarks(self, df: pd.DataFrame):
        """지역별 평균 매출/지출 벤치마크 계산"""
        if '행정동_코드_명' not in df.columns:
            print("⚠️ 지역 정보 컬럼이 없습니다.")
            return

        location_stats = df.groupby('행정동_코드_명').agg({
            '당월_매출_금액': ['mean', 'median', 'std', 'count'],
            '추정지출금액': ['mean', 'median', 'std', 'count']
        }).round(0)

        # 평탄화하고 딕셔너리로 변환
        location_benchmarks = {}
        for location in location_stats.index:
            if location_stats.loc[location, ('당월_매출_금액', 'count')] >= 5:  # 최소 5개 샘플
                location_benchmarks[location] = {
                    'revenue_mean': location_stats.loc[location, ('당월_매출_금액', 'mean')],
                    'revenue_median': location_stats.loc[location, ('당월_매출_금액', 'median')],
                    'revenue_std': location_stats.loc[location, ('당월_매출_금액', 'std')],
                    'expense_mean': location_stats.loc[location, ('추정지출금액', 'mean')],
                    'expense_median': location_stats.loc[location, ('추정지출금액', 'median')],
                    'expense_std': location_stats.loc[location, ('추정지출금액', 'std')],
                    'sample_count': location_stats.loc[location, ('당월_매출_금액', 'count')]
                }

        self.benchmarks['location'] = location_benchmarks

    def get_industry_benchmark(self, business_type: str) -> Optional[Dict]:
        """업종별 벤치마크 반환"""
        print(f"🔍 업종 벤치마크 검색: {business_type}")
        print(f"  사용 가능한 업종: {list(self.benchmarks.get('industry', {}).keys())[:5]}...")

        # 키워드 매칭
        for industry, benchmark in self.benchmarks.get('industry', {}).items():
            if business_type in industry or industry in business_type:
                print(f"  ✅ 매칭된 업종: {industry}")
                return benchmark

        # 매칭되는 업종이 없으면 일반 음식점 평균 반환
        food_industries = [k for k in self.benchmarks.get('industry', {}).keys() if '음식점' in k]
        print(f"  음식점 업종들: {food_industries[:3]}...")

        if food_industries:
            # 음식점 업종들의 평균 계산
            avg_benchmark = {}
            for key in ['revenue_mean', 'expense_mean']:
                values = [self.benchmarks['industry'][ind][key] for ind in food_industries
                         if key in self.benchmarks['industry'][ind]]
                avg_benchmark[key] = np.mean(values) if values else 10000000
            print(f"  📊 평균 벤치마크 계산: {avg_benchmark}")
            return avg_benchmark

        print("  ❌ 벤치마크를 찾을 수 없음")
        return None

    def get_location_benchmark(self, location: str) -> Optional[Dict]:
        """지역별 벤치마크 반환"""
        # 키워드 매칭
        for loc, benchmark in self.benchmarks.get('location', {}).items():
            if location in loc or loc in location:
                return benchmark

        # 매칭되는 지역이 없으면 전체 평균 반환
        if self.benchmarks.get('location'):
            avg_benchmark = {}
            for key in ['revenue_mean', 'expense_mean']:
                values = [bench[key] for bench in self.benchmarks['location'].values()
                         if key in bench]
                avg_benchmark[key] = np.mean(values) if values else 10000000
            return avg_benchmark

        return None

    def compare_user_expenses(self,
                            monthly_revenue: int,
                            monthly_expenses: Dict[str, int],
                            business_type: str,
                            location: str) -> Dict:
        """
        사용자 지출 구조를 업종/지역 평균과 비교 (핵심 기능!)
        """
        print("🔍 업종/지역 대비 지출 구조 분석 중...")

        # 벤치마크 데이터 가져오기
        industry_benchmark = self.get_industry_benchmark(business_type)
        location_benchmark = self.get_location_benchmark(location)

        # 업종별 표준 지출 구조 가져오기
        industry_coeff = self.industry_coefficients.get(business_type)
        if not industry_coeff:
            # 가장 유사한 업종 찾기
            for key in self.industry_coefficients.keys():
                if key in business_type or business_type in key:
                    industry_coeff = self.industry_coefficients[key]
                    break
            else:
                industry_coeff = self.industry_coefficients['기타']

        total_expenses = sum(monthly_expenses.values())

        # ⭐ 중요: data/raw는 지역 전체 업종 총합이므로 비율로 비교
        # 벤치마크 매출 대비 지출 비율 계산
        if industry_benchmark and industry_benchmark.get('revenue_mean', 0) > 0:
            benchmark_expense_ratio = industry_benchmark['expense_mean'] / industry_benchmark['revenue_mean']
            print(f"  📊 업종 평균 지출 비율: {benchmark_expense_ratio*100:.1f}%")

            # 사용자 매출에 비례한 예상 지출 계산
            benchmark_total_expense = monthly_revenue * benchmark_expense_ratio
            print(f"  📊 사용자 매출 기준 예상 지출: {benchmark_total_expense:,.0f}원")
        else:
            benchmark_expense_ratio = 0.85  # 기본 가정: 85% 지출률
            benchmark_total_expense = monthly_revenue * benchmark_expense_ratio
            print(f"  📊 추정 지출 (매출의 85%): {benchmark_total_expense:,.0f}원")

        # 업종 평균 지출 구조로 세부 분해
        benchmark_expenses = {
            'labor_cost': benchmark_total_expense * industry_coeff['expense_structure']['labor_ratio'],
            'food_materials': benchmark_total_expense * industry_coeff['expense_structure']['material_ratio'],
            'rent': benchmark_total_expense * industry_coeff['expense_structure']['rent_ratio'],
            'others': benchmark_total_expense * industry_coeff['expense_structure']['others_ratio']
        }
        print(f"  📋 사용자 매출 기준 벤치마크: {benchmark_expenses}")

        # 사용자 vs 벤치마크 비교
        expense_comparison = {}
        total_comparison = (total_expenses / benchmark_total_expense * 100) if benchmark_total_expense > 0 else 100

        for expense_type, user_amount in monthly_expenses.items():
            benchmark_amount = benchmark_expenses.get(expense_type, benchmark_total_expense * 0.25)
            if benchmark_amount > 0:
                ratio = user_amount / benchmark_amount * 100
                expense_comparison[expense_type] = {
                    'user_amount': user_amount,
                    'benchmark_amount': benchmark_amount,
                    'ratio_percent': ratio,
                    'status': self._get_expense_status(ratio),
                    'message': f"업종 평균 대비 {ratio:.0f}%"
                }
            else:
                expense_comparison[expense_type] = {
                    'user_amount': user_amount,
                    'benchmark_amount': 0,
                    'ratio_percent': 100,
                    'status': 'normal',
                    'message': "평균 수준"
                }

        # 종합 분석
        result = {
            'business_type': business_type,
            'location': location,
            'benchmark_info': {
                'industry_revenue_avg': industry_benchmark['revenue_mean'] if industry_benchmark else None,
                'industry_expense_avg': industry_benchmark['expense_mean'] if industry_benchmark else None,
                'location_revenue_avg': location_benchmark['revenue_mean'] if location_benchmark else None,
                'location_expense_avg': location_benchmark['expense_mean'] if location_benchmark else None
            },
            'total_comparison': {
                'user_total': total_expenses,
                'benchmark_total': benchmark_total_expense,
                'ratio_percent': total_comparison,
                'status': self._get_expense_status(total_comparison),
                'message': f"업종 평균 대비 {total_comparison:.0f}%"
            },
            'expense_breakdown': expense_comparison,
            'recommendations': self._generate_expense_recommendations(expense_comparison, total_comparison)
        }

        return result

    def _get_expense_status(self, ratio: float) -> str:
        """지출 비율에 따른 상태 분류"""
        if ratio < 80:
            return 'low'
        elif ratio < 120:
            return 'normal'
        elif ratio < 150:
            return 'high'
        else:
            return 'very_high'

    def _generate_expense_recommendations(self, expense_comparison: Dict, total_ratio: float) -> List[str]:
        """지출 분석 기반 권장사항 생성"""
        recommendations = []

        # 총 지출 수준 평가
        if total_ratio > 150:
            recommendations.append("💸 총 지출이 업종 평균 대비 매우 높습니다. 전반적인 비용 절감이 필요합니다.")
        elif total_ratio > 120:
            recommendations.append("📊 총 지출이 업종 평균보다 높습니다. 주요 비용 항목을 점검해보세요.")
        elif total_ratio < 80:
            recommendations.append("✨ 총 지출이 업종 평균보다 낮습니다. 효율적인 운영을 하고 있습니다.")

        # 세부 지출 항목별 권장사항
        for expense_type, data in expense_comparison.items():
            ratio = data['ratio_percent']
            expense_name = {
                'labor_cost': '인건비',
                'food_materials': '식자재비',
                'rent': '임대료',
                'others': '기타 지출'
            }.get(expense_type, expense_type)

            if ratio > 150:
                recommendations.append(f"🔴 {expense_name}가 평균 대비 매우 높습니다 ({ratio:.0f}%). 구조적 개선이 필요합니다.")
            elif ratio > 120:
                recommendations.append(f"🟠 {expense_name}가 평균보다 높습니다 ({ratio:.0f}%). 절감 방안을 검토하세요.")
            elif ratio < 70:
                recommendations.append(f"🟢 {expense_name}가 평균보다 낮습니다 ({ratio:.0f}%). 좋은 관리 상태입니다.")

        if not recommendations:
            recommendations.append("✅ 전반적으로 업종 평균 수준의 지출 구조를 유지하고 있습니다.")

        return recommendations

    def estimate_financial_metrics(self,
                                 monthly_revenue: int,
                                 monthly_expenses: Dict[str, int],
                                 total_available_assets: int,
                                 business_type: str) -> Dict:
        """
        Altman Z-Score 계산을 위한 재무 지표 추정
        """
        # 업종 계수 가져오기
        industry_coeff = self.industry_coefficients.get(business_type, self.industry_coefficients['기타'])

        # 월 순이익 계산
        total_expenses = sum(monthly_expenses.values())
        monthly_profit = monthly_revenue - total_expenses
        annual_profit = monthly_profit * 12

        # 재무 지표 추정
        asset_turnover = industry_coeff['asset_turnover']
        debt_ratio = industry_coeff['debt_ratio']

        # 총자산 추정 (매출 기반)
        estimated_total_assets = max(total_available_assets, (monthly_revenue * 12) / asset_turnover)

        # 부채 추정
        estimated_total_debt = estimated_total_assets * debt_ratio

        # 자기자본 추정
        estimated_equity = estimated_total_assets - estimated_total_debt

        # 이익잉여금 추정 (연간 순이익의 누적)
        retention_ratio = industry_coeff['retention_ratio']
        estimated_retained_earnings = max(0, annual_profit * retention_ratio * 2)  # 2년치 가정

        # 운전자본 추정 (당좌자산 - 당좌부채)
        estimated_working_capital = total_available_assets * 0.8  # 가용자산의 80%를 운전자본으로 가정

        return {
            'total_assets': estimated_total_assets,
            'total_debt': estimated_total_debt,
            'equity_book_value': estimated_equity,
            'retained_earnings': estimated_retained_earnings,
            'working_capital': estimated_working_capital,
            'annual_profit': annual_profit,
            'monthly_profit': monthly_profit,
            'estimation_basis': {
                'asset_turnover_used': asset_turnover,
                'debt_ratio_used': debt_ratio,
                'retention_ratio_used': retention_ratio
            }
        }

if __name__ == "__main__":
    # 테스트
    processor = BenchmarkDataProcessor()

    # 샘플 지출 비교 테스트
    test_comparison = processor.compare_user_expenses(
        monthly_revenue=15000000,
        monthly_expenses={
            'labor_cost': 6000000,   # 높음
            'food_materials': 3000000,  # 보통
            'rent': 2000000,         # 보통
            'others': 1500000        # 보통
        },
        business_type="한식음식점",
        location="강남구"
    )

    print("\n🔍 지출 구조 분석 결과:")
    print(f"📊 총 지출: {test_comparison['total_comparison']['message']}")

    for expense_type, data in test_comparison['expense_breakdown'].items():
        print(f"  💰 {expense_type}: {data['message']} (상태: {data['status']})")

    print("\n💡 권장사항:")
    for rec in test_comparison['recommendations']:
        print(f"  {rec}")

    # 재무 지표 추정 테스트
    financial_metrics = processor.estimate_financial_metrics(
        monthly_revenue=15000000,
        monthly_expenses={'labor_cost': 6000000, 'food_materials': 3000000, 'rent': 2000000, 'others': 1500000},
        total_available_assets=50000000,
        business_type="한식음식점"
    )

    print(f"\n📈 추정 재무지표:")
    print(f"  총자산: {financial_metrics['total_assets']:,.0f}원")
    print(f"  총부채: {financial_metrics['total_debt']:,.0f}원")
    print(f"  자기자본: {financial_metrics['equity_book_value']:,.0f}원")
    print(f"  연간순이익: {financial_metrics['annual_profit']:,.0f}원")