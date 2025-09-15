"""
ML Feature Engineering Pipeline
하이브리드 모델용 피처 엔지니어링
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class HybridFeatureEngineer:
    """하이브리드 모델용 피처 엔지니어링"""

    def __init__(self):
        self.expense_ratio = 0.7544867193

    def load_and_combine_data(self, data_dir: str = "data/raw") -> pd.DataFrame:
        """모든 CSV 파일 로드 및 결합"""
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*상권분석서비스*.csv"))

        all_data = []
        for file in csv_files:
            df = pd.read_csv(file)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"전체 데이터: {len(combined_df):,}행")
        return combined_df

    def create_business_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """사업자별 시계열 데이터 생성"""
        # 사업자 고유 식별자 생성
        df['business_id'] = df['행정동_코드'].astype(str) + '_' + df['서비스_업종_코드'].astype(str)

        # 시간 정보 추출
        df['year'] = df['기준_년분기_코드'].astype(str).str[:4].astype(int)
        df['quarter'] = df['기준_년분기_코드'].astype(str).str[4:].astype(int)
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + (df['quarter']*3).astype(str) + '-01')

        # 매출, 지출 데이터 정리
        df['revenue'] = df['당월_매출_금액'].fillna(0)
        df['expenses'] = df.get('추정지출금액', df['revenue'] * self.expense_ratio)
        df['profit'] = df['revenue'] - df['expenses']
        df['profit_margin'] = np.where(df['revenue'] > 0, df['profit'] / df['revenue'], 0)

        return df.sort_values(['business_id', 'date'])

    def calculate_altman_features(self, business_df: pd.DataFrame) -> Dict[str, float]:
        """사업자별 Altman Z-Score 기반 피처"""

        # 최근 데이터 기준 재무 지표 추정
        recent_data = business_df.tail(4)  # 최근 4분기

        avg_revenue = recent_data['revenue'].mean()
        avg_expenses = recent_data['expenses'].mean()
        avg_profit = recent_data['profit'].mean()

        # 추정 재무상태표 (간소화)
        estimated_assets = avg_revenue * 2.0  # 매출의 2배를 자산으로 추정
        estimated_working_capital = avg_profit * 6  # 6개월치 순이익을 운전자본으로
        estimated_debt = estimated_assets * 0.4  # 자산의 40%를 부채로 추정
        estimated_equity = estimated_assets - estimated_debt

        # Altman Z-Score 구성요소
        features = {
            'working_capital_ratio': estimated_working_capital / estimated_assets if estimated_assets > 0 else 0,
            'retained_earnings_ratio': (avg_profit * len(business_df)) / estimated_assets if estimated_assets > 0 else 0,
            'ebit_ratio': avg_profit / estimated_assets if estimated_assets > 0 else 0,
            'equity_debt_ratio': estimated_equity / estimated_debt if estimated_debt > 0 else 10,
            'asset_turnover': avg_revenue / estimated_assets if estimated_assets > 0 else 0
        }

        # Z-Score 계산 (소상공인용 가중치)
        zscore = (6.56 * features['working_capital_ratio'] +
                 3.26 * features['retained_earnings_ratio'] +
                 6.72 * features['ebit_ratio'] +
                 1.05 * features['equity_debt_ratio'])

        features['altman_zscore'] = zscore

        return features

    def calculate_operational_features(self, business_df: pd.DataFrame) -> Dict[str, float]:
        """영업 안정성 피처"""

        revenue_series = business_df['revenue'].values

        features = {}

        # 1. 매출 성장성
        if len(revenue_series) >= 2:
            growth_rates = []
            for i in range(1, len(revenue_series)):
                if revenue_series[i-1] > 0:
                    growth = (revenue_series[i] - revenue_series[i-1]) / revenue_series[i-1]
                    growth_rates.append(growth)

            features['avg_growth_rate'] = np.mean(growth_rates) if growth_rates else 0
            features['growth_volatility'] = np.std(growth_rates) if len(growth_rates) > 1 else 0
        else:
            features['avg_growth_rate'] = 0
            features['growth_volatility'] = 0

        # 2. 매출 변동성
        if len(revenue_series) > 1 and np.mean(revenue_series) > 0:
            features['revenue_cv'] = np.std(revenue_series) / np.mean(revenue_series)
        else:
            features['revenue_cv'] = 0

        # 3. 사업 지속성
        features['business_quarters'] = len(business_df)
        features['non_zero_revenue_quarters'] = sum(1 for r in revenue_series if r > 0)
        features['revenue_consistency'] = features['non_zero_revenue_quarters'] / features['business_quarters']

        # 4. 계절성 지표
        if len(business_df) >= 4:
            quarterly_avg = business_df.groupby(business_df['date'].dt.quarter)['revenue'].mean()
            if len(quarterly_avg) > 1:
                features['seasonality_strength'] = quarterly_avg.std() / quarterly_avg.mean()
            else:
                features['seasonality_strength'] = 0
        else:
            features['seasonality_strength'] = 0

        return features

    def calculate_industry_features(self, business_df: pd.DataFrame, all_data: pd.DataFrame) -> Dict[str, float]:
        """업종 비교 피처"""

        industry_code = business_df['서비스_업종_코드'].iloc[0]

        # 같은 업종 데이터 추출
        industry_data = all_data[all_data['서비스_업종_코드'] == industry_code]

        # 개인 vs 업종 평균 비교
        personal_avg_revenue = business_df['revenue'].mean()
        personal_profit_margin = business_df['profit_margin'].mean()

        industry_avg_revenue = industry_data['revenue'].mean()
        industry_avg_profit_margin = industry_data['profit_margin'].mean()

        features = {
            'revenue_vs_industry': personal_avg_revenue / industry_avg_revenue if industry_avg_revenue > 0 else 1,
            'profit_margin_vs_industry': personal_profit_margin / industry_avg_profit_margin if industry_avg_profit_margin > 0 else 1,
            'industry_percentile_revenue': (industry_data['revenue'] < personal_avg_revenue).mean() * 100,
            'industry_percentile_profit': (industry_data['profit_margin'] < personal_profit_margin).mean() * 100
        }

        return features

    def create_hybrid_features_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 하이브리드 피처셋 생성"""

        feature_list = []
        business_groups = df.groupby('business_id')

        total_businesses = len(business_groups)

        for i, (business_id, business_df) in enumerate(business_groups):
            if i % 1000 == 0:
                logger.info(f"처리 중: {i}/{total_businesses}")

            # 최소 2분기 이상 데이터가 있는 사업자만 처리
            if len(business_df) < 2:
                continue

            try:
                # 각 피처 그룹 계산
                altman_features = self.calculate_altman_features(business_df)
                operational_features = self.calculate_operational_features(business_df)
                industry_features = self.calculate_industry_features(business_df, df)

                # 기본 정보
                basic_features = {
                    'business_id': business_id,
                    'industry_code': business_df['서비스_업종_코드'].iloc[0],
                    'district_code': business_df['행정동_코드'].iloc[0],
                    'quarters_active': len(business_df),
                    'latest_revenue': business_df['revenue'].iloc[-1],
                    'latest_profit_margin': business_df['profit_margin'].iloc[-1]
                }

                # 모든 피처 합치기
                all_features = {**basic_features, **altman_features, **operational_features, **industry_features}
                feature_list.append(all_features)

            except Exception as e:
                logger.warning(f"피처 생성 실패 {business_id}: {e}")
                continue

        features_df = pd.DataFrame(feature_list)
        logger.info(f"피처 데이터셋 생성 완료: {len(features_df)}개 사업자")

        return features_df

    def create_risk_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """위험도 라벨 생성 (하이브리드 모델 기준)"""

        # 각 구성요소 점수 계산
        features_df = features_df.copy()

        # 1. 재무건전성 점수 (Z-Score 기반)
        features_df['financial_health_score'] = np.where(
            features_df['altman_zscore'] >= 2.99, 90 + np.minimum(10, (features_df['altman_zscore'] - 2.99) * 5),
            np.where(features_df['altman_zscore'] >= 1.81,
                    60 + ((features_df['altman_zscore'] - 1.81) / (2.99 - 1.81)) * 30,
                    np.maximum(0, (features_df['altman_zscore'] / 1.81) * 60))
        )

        # 2. 영업안정성 점수
        growth_score = np.clip(50 + features_df['avg_growth_rate'] * 1000, 0, 100)
        volatility_score = np.clip(100 - features_df['revenue_cv'] * 250, 0, 100)
        continuity_score = features_df['revenue_consistency'] * 100

        features_df['operational_stability_score'] = (growth_score * 0.44 + volatility_score * 0.33 + continuity_score * 0.23)

        # 3. 업종내 위치 점수
        features_df['industry_position_score'] = (
            features_df['industry_percentile_revenue'] * 0.4 +
            features_df['industry_percentile_profit'] * 0.6
        )

        # 4. 총 위험도 점수 (가중평균)
        features_df['total_risk_score'] = (
            features_df['financial_health_score'] * 0.40 +
            features_df['operational_stability_score'] * 0.45 +
            features_df['industry_position_score'] * 0.15
        )

        # 5. 위험도 등급
        features_df['risk_level'] = pd.cut(
            features_df['total_risk_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['매우위험', '위험군', '적정', '좋음', '매우좋음']
        )

        # 6. 이진 분류 라벨 (위험 여부)
        features_df['is_risky'] = (features_df['total_risk_score'] <= 40).astype(int)

        return features_df