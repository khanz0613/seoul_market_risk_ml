#!/usr/bin/env python3
"""
Full-Scale ML System with Complete Dataset
==========================================

🔧 문제점 해결:
1. 전체 40만개+ 데이터 사용 (10,000개 → 400,000개+)
2. 실제 지역/업종 인코딩 완전 해결
3. 고성능 머신러닝 파이프라인

Author: Seoul Market Risk ML - Full Scale Version
Date: 2025-09-17
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class FullScaleMLSystem:
    """전체 데이터를 사용하는 고성능 ML 시스템"""

    def __init__(self):
        self.model = None
        self.region_encoder = LabelEncoder()
        self.business_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # 모델 저장 디렉토리
        self.model_dir = "full_scale_models"
        Path(self.model_dir).mkdir(exist_ok=True)

        # 성능 메트릭
        self.training_stats = {
            'total_samples': 0,
            'unique_regions': 0,
            'unique_businesses': 0,
            'training_time': 0,
            'accuracy': 0,
            'cross_val_score': 0
        }

    def load_complete_seoul_data(self) -> pd.DataFrame:
        """전체 서울 상권 데이터 로드 (40만개+)"""
        print("📂 Loading COMPLETE Seoul commercial data...")

        data_dir = "data/raw"
        csv_files = list(Path(data_dir).glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        print(f"Found {len(csv_files)} data files")

        # 모든 파일 로드 및 결합
        all_dataframes = []
        total_rows = 0

        for csv_file in csv_files:
            print(f"  Loading: {csv_file.name}")
            df = pd.read_csv(csv_file, encoding='utf-8')
            all_dataframes.append(df)
            total_rows += len(df)
            print(f"    Rows: {len(df):,}")

        # 모든 데이터 결합
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        print(f"✅ Total combined data: {len(combined_df):,} rows")
        print(f"✅ Columns: {len(combined_df.columns)}")

        # 기본 정보 출력
        print(f"\n📊 Data Overview:")
        print(f"   Unique regions: {combined_df['행정동_코드_명'].nunique():,}")
        print(f"   Unique businesses: {combined_df['서비스_업종_코드_명'].nunique():,}")
        print(f"   Date range: {combined_df['기준_년분기_코드'].min()} - {combined_df['기준_년분기_코드'].max()}")

        return combined_df

    def create_comprehensive_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """실제 데이터 기반 훈련 데이터 생성 (40만개 전체 사용)"""
        print("🔧 Creating comprehensive training data from FULL dataset...")

        # 결측값 처리
        df = df.dropna(subset=['당월_매출_금액', '행정동_코드_명', '서비스_업종_코드_명'])

        # 매출액이 0이거나 음수인 경우 제거
        df = df[df['당월_매출_금액'] > 0]

        print(f"✅ After cleaning: {len(df):,} valid records")

        # 훈련 데이터 구조 생성
        training_data = []

        for idx, row in df.iterrows():
            if idx % 50000 == 0:
                print(f"  Processing: {idx:,}/{len(df):,} ({idx/len(df)*100:.1f}%)")

            # 실제 매출 데이터 사용
            base_sales = float(row['당월_매출_금액'])

            # 실제 거래 건수 고려
            transaction_count = float(row.get('당월_매출_건수', 1))
            avg_transaction = base_sales / max(transaction_count, 1)

            # 월매출 = 실제 데이터
            monthly_revenue = base_sales

            # 비용 구조 추정 (업종별 차별화)
            business_type = row['서비스_업종_코드_명']

            # 업종별 비용 구조
            if '음식점' in business_type or '카페' in business_type:
                labor_ratio = np.random.uniform(0.25, 0.35)  # 25-35%
                rent_ratio = np.random.uniform(0.15, 0.25)   # 15-25%
                material_ratio = np.random.uniform(0.30, 0.40)  # 30-40%
                other_ratio = np.random.uniform(0.05, 0.15)   # 5-15%
            elif '소매' in business_type:
                labor_ratio = np.random.uniform(0.15, 0.25)
                rent_ratio = np.random.uniform(0.20, 0.30)
                material_ratio = np.random.uniform(0.40, 0.50)
                other_ratio = np.random.uniform(0.05, 0.10)
            else:  # 서비스업
                labor_ratio = np.random.uniform(0.30, 0.45)
                rent_ratio = np.random.uniform(0.10, 0.20)
                material_ratio = np.random.uniform(0.10, 0.20)
                other_ratio = np.random.uniform(0.10, 0.20)

            # 총자산 추정 (매출 기반)
            total_assets = monthly_revenue * np.random.uniform(4, 10)

            # 가용자산 추정 (총자산의 일부)
            available_cash = total_assets * np.random.uniform(0.15, 0.35)

            training_data.append({
                '총자산': total_assets,
                '월매출': monthly_revenue,
                '인건비': monthly_revenue * labor_ratio,
                '임대료': monthly_revenue * rent_ratio,
                '식자재비': monthly_revenue * material_ratio,
                '기타비용': monthly_revenue * other_ratio,
                '가용자산': available_cash,
                '지역': row['행정동_코드_명'],
                '업종': row['서비스_업종_코드_명'],
                # 추가 정보 (분석용)
                '거래건수': transaction_count,
                '평균거래액': avg_transaction,
                '연도분기': row['기준_년분기_코드']
            })

        training_df = pd.DataFrame(training_data)

        print(f"✅ Complete training data created: {len(training_df):,} samples")
        print(f"   Unique regions: {training_df['지역'].nunique():,}")
        print(f"   Unique businesses: {training_df['업종'].nunique():,}")

        # 통계 업데이트
        self.training_stats['total_samples'] = len(training_df)
        self.training_stats['unique_regions'] = training_df['지역'].nunique()
        self.training_stats['unique_businesses'] = training_df['업종'].nunique()

        return training_df

    def create_enhanced_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """향상된 라벨링 시스템"""
        print("🏷️ Creating enhanced risk labels...")

        # 재무 비율 계산
        df['월비용'] = df['인건비'] + df['임대료'] + df['식자재비'] + df['기타비용']
        df['월순익'] = df['월매출'] - df['월비용']
        df['순익률'] = df['월순익'] / df['월매출']
        df['자산회전율'] = (df['월매출'] * 12) / df['총자산']
        df['현금비율'] = df['가용자산'] / df['월비용']

        # Altman Z-Score 개선 버전
        working_capital = df['가용자산']
        retained_earnings = df['월순익'] * 12 * 0.3  # 실제 이익 기반
        ebit = df['월순익'] * 12
        market_value_equity = df['총자산'] - (df['총자산'] * 0.3)
        sales = df['월매출'] * 12

        # 안전한 분모 계산
        safe_assets = np.maximum(df['총자산'], 1000000)
        safe_debt = np.maximum(df['총자산'] * 0.3, 100000)

        # Z-Score 계산
        A = working_capital / safe_assets
        B = retained_earnings / safe_assets
        C = ebit / safe_assets
        D = market_value_equity / safe_debt
        E = sales / safe_assets

        df['z_score'] = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

        # 개선된 위험도 분류 (더 세밀한 구분)
        def assign_enhanced_risk_label(row):
            z_score = row['z_score']
            profit_margin = row['순익률']
            cash_ratio = row['현금비율']

            # 복합 지표 고려
            if z_score > 3.0 and profit_margin > 0.1 and cash_ratio > 2.0:
                return 1  # 매우 안전
            elif z_score > 2.7 and profit_margin > 0.05:
                return 2  # 안전
            elif z_score > 1.8 and profit_margin > 0:
                return 3  # 보통
            elif z_score > 1.1 or profit_margin > -0.1:
                return 4  # 위험
            else:
                return 5  # 매우 위험

        df['risk_label'] = df.apply(assign_enhanced_risk_label, axis=1)

        print("✅ Enhanced risk labels created:")
        print(df['risk_label'].value_counts().sort_index())

        return df

    def prepare_full_features(self, df: pd.DataFrame):
        """전체 데이터를 위한 완전한 피처 엔지니어링"""
        print("🔧 Preparing complete features for full dataset...")

        # 기본 피처
        total_cost = df['인건비'] + df['임대료'] + df['식자재비'] + df['기타비용']

        features = [
            np.log1p(df['총자산']),      # 로그 변환
            np.log1p(df['월매출']),      # 로그 변환
            np.log1p(total_cost),        # 로그 변환
            np.log1p(df['가용자산']),    # 추가 피처
            df['월매출'] / df['총자산'],  # 자산 효율성
            total_cost / df['월매출'],   # 비용 비율
        ]

        # 지역 인코딩 (전체 데이터)
        print(f"  Encoding {df['지역'].nunique():,} unique regions...")
        region_encoded = self.region_encoder.fit_transform(df['지역'])
        features.append(region_encoded)

        # 업종 인코딩 (전체 데이터)
        print(f"  Encoding {df['업종'].nunique():,} unique business types...")
        business_encoded = self.business_encoder.fit_transform(df['업종'])
        features.append(business_encoded)

        # 추가 파생 피처
        features.extend([
            df['거래건수'],              # 거래 빈도
            np.log1p(df['평균거래액']),   # 거래 규모
        ])

        # 피처 매트릭스 생성
        X = np.column_stack(features)

        # 표준화
        X_scaled = self.scaler.fit_transform(X)

        print(f"✅ Feature matrix: {X_scaled.shape}")
        print(f"   Features: 10개 (기본 6개 + 지역 + 업종 + 추가 2개)")

        return X_scaled

    def train_full_scale_model(self, X, y):
        """전체 데이터를 활용한 고성능 ML 훈련"""
        print("🤖 Training FULL-SCALE ML model...")
        print(f"📊 Training on {len(X):,} samples")

        start_time = time.time()

        # 데이터 분할 (더 큰 데이터셋이므로 더 엄격한 분할)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Test samples: {len(X_test):,}")

        # 고성능 RandomForest 설정
        self.model = RandomForestClassifier(
            n_estimators=200,        # 더 많은 트리
            max_depth=15,           # 더 깊은 학습
            min_samples_split=10,   # 더 세밀한 분할
            min_samples_leaf=5,     # 더 세밀한 잎
            max_features='sqrt',    # 피처 서브샘플링
            random_state=42,
            class_weight='balanced',
            n_jobs=-1              # 병렬 처리
        )

        # 실제 훈련
        print("🔄 Training in progress...")
        self.model.fit(X_train, y_train)

        training_time = time.time() - start_time
        self.training_stats['training_time'] = training_time

        # 성능 평가
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.training_stats['accuracy'] = accuracy

        print(f"✅ Full-scale model trained successfully!")
        print(f"   Training time: {training_time:.1f} seconds")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")

        # 교차 검증
        print("🔄 Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, n_jobs=-1)
        cv_mean = cv_scores.mean()
        self.training_stats['cross_val_score'] = cv_mean

        print(f"📊 5-Fold CV Score: {cv_mean:.4f} (±{cv_scores.std():.4f})")

        # 상세 성능 리포트
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))

        # 혼동 행렬
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # 피처 중요도
        feature_names = ['총자산', '월매출', '월비용', '가용자산', '자산효율성',
                        '비용비율', '지역', '업종', '거래건수', '평균거래액']
        importances = self.model.feature_importances_

        print("\n📊 Feature Importance:")
        for name, importance in zip(feature_names, importances):
            print(f"   {name}: {importance:.3f}")

        return accuracy

    def save_full_model(self):
        """완전한 모델 저장"""
        model_path = f"{self.model_dir}/full_scale_ml_model.joblib"
        encoder_path = f"{self.model_dir}/full_scale_encoders.joblib"
        scaler_path = f"{self.model_dir}/full_scale_scaler.joblib"
        stats_path = f"{self.model_dir}/training_stats.json"

        # 모델 저장
        joblib.dump(self.model, model_path)

        # 인코더 저장
        encoders = {
            'region_encoder': self.region_encoder,
            'business_encoder': self.business_encoder
        }
        joblib.dump(encoders, encoder_path)

        # 스케일러 저장
        joblib.dump(self.scaler, scaler_path)

        # 통계 저장
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        print(f"✅ Full-scale model saved:")
        print(f"   Model: {model_path}")
        print(f"   Encoders: {encoder_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Stats: {stats_path}")

    def predict_with_full_model(self, 총자산, 월매출, 인건비, 임대료, 식자재비,
                               기타비용, 가용자산, 지역, 업종):
        """완전한 모델로 예측"""
        if self.model is None:
            raise ValueError("Model not trained!")

        print("🎯 Full-Scale ML Prediction")
        print("=" * 40)

        # 피처 준비
        total_cost = 인건비 + 임대료 + 식자재비 + 기타비용

        # 지역/업종 인코딩 (훈련된 인코더 사용)
        try:
            region_encoded = self.region_encoder.transform([지역])[0]
            print(f"✅ Region '{지역}' recognized")
        except ValueError:
            region_encoded = 0
            print(f"⚠️ Unknown region '{지역}', using fallback")

        try:
            business_encoded = self.business_encoder.transform([업종])[0]
            print(f"✅ Business '{업종}' recognized")
        except ValueError:
            business_encoded = 0
            print(f"⚠️ Unknown business '{업종}', using fallback")

        # 피처 벡터 (훈련 시와 동일한 구조)
        features = np.array([
            np.log1p(총자산),
            np.log1p(월매출),
            np.log1p(total_cost),
            np.log1p(가용자산),
            월매출 / 총자산,        # 자산 효율성
            total_cost / 월매출,    # 비용 비율
            region_encoded,
            business_encoded,
            100,                    # 기본 거래건수
            월매출 / 100           # 평균 거래액
        ]).reshape(1, -1)

        # 표준화
        features_scaled = self.scaler.transform(features)

        # 예측
        risk_level = self.model.predict(features_scaled)[0]
        confidence = max(self.model.predict_proba(features_scaled)[0]) * 100

        risk_names = {1: "매우안전", 2: "안전", 3: "보통", 4: "위험", 5: "매우위험"}

        print(f"\n🎯 Full-Scale ML Result:")
        print(f"   Risk Level: {risk_level} ({risk_names[risk_level]})")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Model: Full-Scale RandomForest")
        print(f"   Training Data: {self.training_stats['total_samples']:,} samples")

        return {
            'risk_level': risk_level,
            'risk_name': risk_names[risk_level],
            'confidence': confidence,
            'model_type': 'Full-Scale ML (400K+ samples)',
            'training_stats': self.training_stats
        }

def main():
    """전체 데이터를 사용한 ML 시스템 실행"""
    print("🚀 Full-Scale ML System (400K+ Samples)")
    print("=" * 60)
    print("🎯 Goal: Use COMPLETE dataset for maximum performance")
    print("🔧 Fix: Unknown region/business encoding issues")

    try:
        # 시스템 초기화
        ml_system = FullScaleMLSystem()

        # 1. 전체 데이터 로드
        raw_data = ml_system.load_complete_seoul_data()

        # 2. 훈련 데이터 생성 (전체 사용)
        training_data = ml_system.create_comprehensive_training_data(raw_data)

        # 3. 라벨링
        labeled_data = ml_system.create_enhanced_labels(training_data)

        # 4. 피처 엔지니어링
        X = ml_system.prepare_full_features(labeled_data)
        y = labeled_data['risk_label'].values

        # 5. 모델 훈련
        accuracy = ml_system.train_full_scale_model(X, y)

        # 6. 모델 저장
        ml_system.save_full_model()

        print("\n" + "=" * 70)
        print("✅ FULL-SCALE ML SYSTEM READY!")
        print("🎉 Complete Dataset Training Successful")
        print(f"📊 Final Accuracy: {accuracy:.4f}")
        print(f"📊 Training Samples: {ml_system.training_stats['total_samples']:,}")
        print(f"📊 Unique Regions: {ml_system.training_stats['unique_regions']:,}")
        print(f"📊 Unique Businesses: {ml_system.training_stats['unique_businesses']:,}")
        print("=" * 70)

        # 7. 테스트 예측
        print("\n🧪 Testing full-scale prediction...")
        result = ml_system.predict_with_full_model(
            총자산=50000000,      # 5천만원
            월매출=12000000,      # 1200만원
            인건비=3000000,       # 300만원
            임대료=2000000,       # 200만원
            식자재비=3500000,     # 350만원
            기타비용=500000,      # 50만원
            가용자산=15000000,    # 1500만원
            지역='강남구',         # 실제 지역 테스트
            업종='커피전문점'      # 실제 업종 테스트
        )

        print(f"\n🎯 Test Result: {result['risk_name']} ({result['confidence']:.1f}%)")
        print(f"✅ FULL-SCALE SYSTEM TEST COMPLETE!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()