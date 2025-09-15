#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Seoul Market Risk ML Preprocessing Pipeline
목표: MAE 대폭 감소 + 최대 데이터 활용
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SeoulMarketPreprocessor:
    def __init__(self):
        """서울 상권 데이터 전처리기 초기화"""
        self.district_clusters = self._create_district_clusters()
        self.business_encoders = {}

    def _create_district_clusters(self):
        """
        서울 행정동을 경제적 특성에 따라 6개 클러스터로 분류
        실제 부동산 가치와 상권 특성을 반영한 전략적 클러스터링
        """
        return {
            # 🏆 TIER 1: 초고급 상권 (강남 4구 + CBD)
            'premium': [
                # 강남구 핵심
                '역삼1동', '논현1동', '압구정동', '청담동', '삼성1동', '대치1동', '대치2동', '대치4동',
                '역삼2동', '논현2동', '신사동', '삼성2동', '개포2동', '일원1동', '일원2동', '수서동',
                # 서초구 핵심
                '서초1동', '서초2동', '서초3동', '서초4동', '반포1동', '반포2동', '반포3동', '반포4동',
                '잠원동', '방배1동', '방배2동', '방배3동', '양재1동', '양재2동', '내곡동',
                # 송파구 핵심
                '문정1동', '문정2동', '석촌동', '삼전동', '가락1동', '가락2동', '송파1동', '송파2동',
                '풍납1동', '풍납2동', '거여1동', '거여2동', '마천1동', '마천2동', '오금동', '오륜동',
                # 종로 CBD
                '종로1·2·3·4가동', '종로5·6가동', '명동', '을지로동', '회현동', '중림동',
                # 여의도 CBD
                '여의동', '당산1동', '당산2동', '영등포동'
            ],

            # 🥈 TIER 2: 고급 상권 (홍대/이태원/한강변 고급지)
            'upscale': [
                # 마포/홍대 상권
                '서교동', '합정동', '상수동', '연남동', '성산1동', '성산2동', '망원1동', '망원2동',
                '염리동', '신수동', '아현동', '공덕동', '용강동', '대흥동', '도화동',
                # 용산/이태원 상권
                '이태원1동', '이태원2동', '한남동', '보광동', '용산2가동', '남영동', '청파동',
                '원효로1동', '원효로2동', '효창동', '용문동', '한강로동',
                # 성동구 고급지역
                '성수1가1동', '성수1가2동', '성수2가1동', '성수2가3동', '왕십리2동', '금강로동',
                # 강서 고급 신도시
                '화곡본동', '화곡1동', '화곡2동', '화곡3동', '화곡4동', '화곡6동', '화곡8동',
                '등촌1동', '등촌2동', '등촌3동', '염창동', '발산1동', '우장산동'
            ],

            # 🥉 TIER 3: 중급 상권 (강북 중심지 + 신흥 상권)
            'midtier': [
                # 강북 중심지
                '종로1·2·3·4가동', '제기동', '청운효자동', '사직동', '삼청동', '부암동',
                '평창동', '무악동', '교남동', '가회동', '종로5·6가동', '창신1동', '창신2동', '창신3동',
                # 성북/동대문 중심
                '성북동', '삼선동', '동선동', '돈암1동', '돈암2동', '안암동', '보문동', '정릉1동',
                '길음1동', '길음2동', '월곡1동', '월곡2동', '장위1동', '장위2동', '장위3동', '석관동',
                # 노원/도봉 중심
                '노원1동', '노원2동', '노원3동', '상계1동', '상계2동', '상계3동', '상계4동',
                '상계5동', '중계본동', '중계1동', '중계2동', '중계4동', '하계1동', '하계2동',
                # 은평 중심
                '은평구', '갈현1동', '갈현2동', '구산동', '대조동', '불광1동', '불광2동',
                '녹번동', '홍제1동', '홍제2동', '홍제3동', '신사1동', '신사2동', '증산동'
            ],

            # 📍 TIER 4: 일반 상권 (주거 중심 + 대학가)
            'standard': [
                # 강동구 전체
                '강일동', '상일동', '명일1동', '명일2동', '고덕1동', '고덕2동', '암사1동', '암사2동', '암사3동',
                '천호1동', '천호2동', '천호3동', '성내1동', '성내2동', '성내3동', '길동', '둔촌1동', '둔촌2동',
                # 중랑구 전체
                '면목본동', '면목2동', '면목3동', '면목4동', '면목5동', '면목7동', '면목8동',
                '상봉1동', '상봉2동', '중화1동', '중화2동', '묵1동', '묵2동', '망우본동', '망우3동',
                # 서대문 대학가
                '충현동', '천연동', '신촌동', '연희동', '홍제동', '홍은1동', '홍은2동', '남가좌1동',
                '남가좌2동', '북가좌1동', '북가좌2동', '냉천동'
            ],

            # 🏘️ TIER 5: 준주거 상권 (외곽 주거지역)
            'residential': [
                # 관악구 대부분
                '청룡동', '청림동', '행운동', '낙성대동', '인헌동', '남현동', '서원동', '신원동',
                '서림동', '미성동', '난곡동', '인사동', '성현동', '중앙동', '상도1동', '상도2동',
                '상도3동', '상도4동', '흑석동', '노량진1동', '노량진2동', '대방동', '신대방1동', '신대방2동',
                # 금천구 전체
                '가산동', '독산1동', '독산2동', '독산3동', '독산4동', '시흥1동', '시흥2동',
                '시흥3동', '시흥4동', '시흥5동',
                # 구로구 대부분
                '신도림동', '구로1동', '구로2동', '구로3동', '구로4동', '구로5동', '가리봉동',
                '개봉1동', '개봉2동', '개봉3동', '오류1동', '오류2동', '항동'
            ],

            # 🌿 TIER 6: 외곽 상권 (변두리/신도시)
            'suburban': [
                # 강북구 전체
                '삼양동', '미아동', '송중동', '송천동', '삼각산동', '번1동', '번2동', '번3동',
                '수유1동', '수유2동', '수유3동', '우이동', '인수동',
                # 도봉구 전체
                '쌍문1동', '쌍문2동', '쌍문3동', '쌍문4동', '방학1동', '방학2동', '방학3동',
                '창1동', '창2동', '창3동', '창4동', '창5동', '도봉1동', '도봉2동',
                # 동대문구 외곽
                '전농1동', '전농2동', '답십리1동', '답십리2동', '장안1동', '장안2동',
                '청량리동', '회기동', '휘경1동', '휘경2동', '이문1동', '이문2동'
            ]
        }

    def load_all_data(self):
        """모든 연도 데이터를 통합하여 로드 (최대 활용)"""
        print("📥 모든 데이터 파일 로딩 중...")

        files = glob.glob('data/raw/서울시*상권분석서비스*.csv')
        dataframes = []

        for file in files:
            print(f"  로딩: {os.path.basename(file)}")
            df = pd.read_csv(file)

            # 연도 추출
            year = None
            if '2019' in file:
                year = 2019
            elif '2020' in file:
                year = 2020
            elif '2021' in file:
                year = 2021
            elif '2022' in file:
                year = 2022
            elif '2023' in file:
                year = 2023
            elif '2024' in file:
                year = 2024

            if year:
                df['연도'] = year
                dataframes.append(df)

        # 전체 데이터 통합
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"✅ 총 {len(combined_df):,}개 레코드 로드 완료")

        return combined_df

    def add_geographic_features(self, df):
        """지역 클러스터링 피처 추가"""
        print("🗺️ 지역 클러스터링 피처 생성...")

        # 클러스터 매핑
        district_to_cluster = {}
        for cluster_name, districts in self.district_clusters.items():
            for district in districts:
                district_to_cluster[district] = cluster_name

        # 클러스터 할당 (매핑되지 않은 지역은 'standard'로 분류)
        df['지역_클러스터'] = df['행정동_코드_명'].map(district_to_cluster).fillna('standard')

        # 클러스터별 통계
        cluster_counts = df['지역_클러스터'].value_counts()
        print("📊 클러스터별 데이터 분포:")
        for cluster, count in cluster_counts.items():
            print(f"  {cluster}: {count:,}개 ({count/len(df)*100:.1f}%)")

        return df

    def apply_target_transformation(self, df):
        """타겟 변수 로그 변환 적용"""
        print("📈 타겟 변수 로그 변환 적용...")

        # 원본 백업
        df['원본_추정지출금액'] = df['추정지출금액'].copy()

        # 0 값 처리 후 로그 변환
        df['추정지출금액_log'] = np.log1p(df['추정지출금액'])

        # 통계 출력
        print(f"  원본 평균: {df['원본_추정지출금액'].mean():,.0f}원")
        print(f"  로그 평균: {df['추정지출금액_log'].mean():.2f}")
        print(f"  원본 표준편차: {df['원본_추정지출금액'].std():,.0f}")
        print(f"  로그 표준편차: {df['추정지출금액_log'].std():.2f}")

        return df

    def create_enhanced_features(self, df):
        """향상된 피처 엔지니어링"""
        print("🛠️ 향상된 피처 생성...")

        # 1. 시간대별 집중도 피처
        time_columns = [col for col in df.columns if '시간대_' in col and '매출_금액' in col]
        if time_columns:
            df['낮시간_비율'] = (df['시간대_11~14_매출_금액'] + df['시간대_14~17_매출_금액']) / df['당월_매출_금액']
            df['밤시간_비율'] = (df['시간대_21~24_매출_금액'] + df['시간대_00~06_매출_금액']) / df['당월_매출_금액']

        # 2. 성별 선호도 피처
        if '남성_매출_금액' in df.columns and '여성_매출_금액' in df.columns:
            total_gender = df['남성_매출_금액'] + df['여성_매출_금액']
            df['남성_비율'] = df['남성_매출_금액'] / total_gender
            df['성별_편중도'] = abs(df['남성_비율'] - 0.5)

        # 3. 연령대 다양성 피처
        age_columns = [col for col in df.columns if '연령대_' in col and '매출_금액' in col]
        if age_columns:
            age_total = df[age_columns].sum(axis=1)
            for col in age_columns:
                age_name = col.replace('연령대_', '').replace('_매출_금액', '')
                df[f'{age_name}_비율'] = df[col] / age_total

        # 4. 주중/주말 패턴
        if '주중_매출_금액' in df.columns and '주말_매출_금액' in df.columns:
            df['주말_비율'] = df['주말_매출_금액'] / df['당월_매출_금액']
            df['평일_집중도'] = df['주중_매출_금액'] / df['주말_매출_금액']

        # 5. 클러스터별 상대적 성과
        cluster_medians = df.groupby('지역_클러스터')['당월_매출_금액'].median()
        df['클러스터_대비_성과'] = df.apply(
            lambda row: row['당월_매출_금액'] / cluster_medians[row['지역_클러스터']], axis=1
        )

        print(f"  생성된 피처 수: {len([col for col in df.columns if any(x in col for x in ['비율', '집중도', '성과'])])}")

        return df

    def encode_categorical_features(self, df):
        """범주형 변수 인코딩"""
        print("🔤 범주형 변수 인코딩...")

        categorical_features = ['지역_클러스터', '서비스_업종_코드_명']

        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature])
                self.business_encoders[feature] = le
                print(f"  {feature}: {len(le.classes_)} 고유값")

        return df

    def filter_and_clean_data(self, df):
        """데이터 필터링 및 정리 (최대한 보존)"""
        print("🧹 데이터 정리 중...")

        initial_count = len(df)

        # 1. 극단적 이상치만 제거 (99.9%ile 이상)
        spending_99_9 = df['추정지출금액'].quantile(0.999)
        df = df[df['추정지출금액'] <= spending_99_9]

        # 2. 0원 데이터는 1원으로 대체 (로그 변환 대비)
        df['추정지출금액'] = df['추정지출금액'].replace(0, 1)

        # 3. 필수 컬럼 누락 데이터만 제거
        essential_columns = ['추정지출금액', '행정동_코드_명', '서비스_업종_코드_명']
        df = df.dropna(subset=essential_columns)

        final_count = len(df)
        retention_rate = (final_count / initial_count) * 100

        print(f"  데이터 보존율: {retention_rate:.1f}% ({initial_count:,} → {final_count:,})")

        return df

    def prepare_training_data(self, df):
        """학습용 데이터 준비"""
        print("🎯 학습용 데이터 준비...")

        # 피처 선택 (강화된 피처셋)
        feature_columns = [
            # 기본 매출 정보
            '당월_매출_금액', '당월_매출_건수',
            # 지역 정보 (핵심!)
            '지역_클러스터_encoded',
            # 업종 정보
            '서비스_업종_코드_명_encoded',
            # 시간대 패턴
            '낮시간_비율', '밤시간_비율', '주말_비율',
            # 고객 특성
            '남성_비율', '성별_편중도',
            # 상대적 성과
            '클러스터_대비_성과',
            # 연도 트렌드
            '연도'
        ]

        # 존재하는 컬럼만 선택
        available_features = [col for col in feature_columns if col in df.columns]

        # 연령대 비율 컬럼 추가
        age_ratio_cols = [col for col in df.columns if '_비율' in col and '연령대' in col]
        available_features.extend(age_ratio_cols)

        X = df[available_features].copy()
        y = df['추정지출금액_log'].copy()  # 로그 변환된 타겟 사용

        # NaN 값 처리
        X = X.fillna(X.median())

        print(f"  최종 피처 수: {len(available_features)}")
        print(f"  학습 데이터 크기: {len(X):,}")
        print(f"  피처 목록: {available_features}")

        return X, y, available_features

    def run_full_pipeline(self):
        """전체 전처리 파이프라인 실행"""
        print("🚀 향상된 전처리 파이프라인 시작...")
        print("=" * 50)

        # 1. 모든 데이터 로드
        df = self.load_all_data()

        # 2. 지역 클러스터링
        df = self.add_geographic_features(df)

        # 3. 타겟 변환
        df = self.apply_target_transformation(df)

        # 4. 피처 엔지니어링
        df = self.create_enhanced_features(df)

        # 5. 범주형 인코딩
        df = self.encode_categorical_features(df)

        # 6. 데이터 정리
        df = self.filter_and_clean_data(df)

        # 7. 학습 데이터 준비
        X, y, feature_names = self.prepare_training_data(df)

        print("=" * 50)
        print("✅ 전처리 완료!")
        print(f"📊 최종 데이터셋: {len(X):,} 샘플, {len(feature_names)} 피처")

        return X, y, feature_names, df

if __name__ == "__main__":
    # 전처리 실행
    preprocessor = SeoulMarketPreprocessor()
    X, y, features, processed_df = preprocessor.run_full_pipeline()

    # 결과 저장
    import joblib

    print("\n💾 결과 저장 중...")

    # 전처리된 데이터 저장
    joblib.dump(X, 'data/processed/X_enhanced.joblib')
    joblib.dump(y, 'data/processed/y_enhanced.joblib')
    joblib.dump(features, 'data/processed/features_enhanced.joblib')

    # 전체 처리된 데이터프레임 저장 (분석용)
    processed_df.to_csv('data/processed/seoul_market_enhanced.csv', index=False)

    print("✅ 저장 완료!")
    print("\n🎯 다음 단계: python enhanced_train_models.py 실행")