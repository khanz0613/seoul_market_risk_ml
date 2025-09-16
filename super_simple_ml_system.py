#!/usr/bin/env python3
"""
Super Simple ML 100% System
===========================

사용자 요구사항 완전 달성:
- 5개 입력만: 총자산, 월매출, 4개 비용, 업종, 지역
- ML 100%: 순수 머신러닝 예측만
- Altman Z-Score: 라벨링에만 사용 (예측에 사용 안함)
- 간단하고 직관적

Author: Seoul Market Risk ML - Super Simple Version
Date: 2025-09-17
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from pathlib import Path
import json

class SuperSimpleMLSystem:
    """초간단 ML 시스템 - 5개 입력으로 위험도 예측"""

    def __init__(self):
        self.model = None
        self.region_encoder = LabelEncoder()
        self.business_encoder = LabelEncoder()
        self.model_dir = "simple_models"

        # 모델 디렉토리 생성
        Path(self.model_dir).mkdir(exist_ok=True)

    def create_simple_labels(self, df):
        """간단한 Altman Z-Score 기반 라벨링"""
        print("🏷️ Creating simple risk labels using basic Altman Z-Score...")

        # 기본 재무 비율 계산
        df['working_capital'] = df['총자산(원)'] * 0.3  # 추정
        df['retained_earnings'] = df['총자산(원)'] * 0.1  # 추정
        df['ebit'] = df['월매출(원)'] * 12 * 0.1  # 추정
        df['market_value'] = df['총자산(원)'] * 0.8  # 추정
        df['sales'] = df['월매출(원)'] * 12

        # Altman Z-Score 계산 (간단 버전)
        df['z_score'] = (
            1.2 * (df['working_capital'] / df['총자산(원)']) +
            1.4 * (df['retained_earnings'] / df['총자산(원)']) +
            3.3 * (df['ebit'] / df['총자산(원)']) +
            0.6 * (df['market_value'] / df['총자산(원)']) +
            1.0 * (df['sales'] / df['총자산(원)'])
        )

        # 위험도 라벨 생성 (1=매우안전, 5=매우위험)
        def assign_risk_label(z_score):
            if z_score > 3.0:
                return 1  # 매우 안전
            elif z_score > 2.7:
                return 2  # 안전
            elif z_score > 1.8:
                return 3  # 보통
            elif z_score > 1.1:
                return 4  # 위험
            else:
                return 5  # 매우 위험

        df['risk_label'] = df['z_score'].apply(assign_risk_label)

        print(f"✅ Risk labels created:")
        print(df['risk_label'].value_counts().sort_index())

        return df

    def prepare_features(self, df):
        """5개 입력만 사용한 간단한 피처 준비"""
        print("🔧 Preparing simple features (5 inputs only)...")

        # 5개 핵심 피처만 사용
        features = []

        # 1. 총자산 (정규화)
        features.append(np.log1p(df['총자산(원)']))

        # 2. 월매출 (정규화)
        features.append(np.log1p(df['월매출(원)']))

        # 3. 총 월비용 (4개 비용의 합)
        total_cost = (df['인건비(원)'] + df['임대료(원)'] +
                     df['식자재비(원)'] + df['기타비용(원)'])
        features.append(np.log1p(total_cost))

        # 4. 지역 (인코딩)
        region_encoded = self.region_encoder.fit_transform(df['지역'])
        features.append(region_encoded)

        # 5. 업종 (인코딩)
        business_encoded = self.business_encoder.fit_transform(df['업종'])
        features.append(business_encoded)

        # 피처 매트릭스 생성
        X = np.column_stack(features)

        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"   Features: [총자산, 월매출, 총비용, 지역, 업종]")

        return X

    def train_simple_model(self, X, y):
        """간단한 ML 모델 훈련"""
        print("🤖 Training simple ML model...")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 간단한 RandomForest 모델
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )

        # 모델 훈련
        self.model.fit(X_train, y_train)

        # 예측 및 평가
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ Model trained successfully!")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        # 성능 리포트
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy

    def save_model(self):
        """모델 저장"""
        model_path = f"{self.model_dir}/simple_ml_model.joblib"
        encoder_path = f"{self.model_dir}/encoders.joblib"

        # 모델 저장
        joblib.dump(self.model, model_path)

        # 인코더 저장
        encoders = {
            'region_encoder': self.region_encoder,
            'business_encoder': self.business_encoder
        }
        joblib.dump(encoders, encoder_path)

        print(f"✅ Model saved: {model_path}")
        print(f"✅ Encoders saved: {encoder_path}")

    def load_model(self):
        """모델 로드"""
        model_path = f"{self.model_dir}/simple_ml_model.joblib"
        encoder_path = f"{self.model_dir}/encoders.joblib"

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.model = joblib.load(model_path)
            encoders = joblib.load(encoder_path)
            self.region_encoder = encoders['region_encoder']
            self.business_encoder = encoders['business_encoder']
            print("✅ Model loaded successfully!")
            return True
        else:
            print("❌ Model files not found!")
            return False

    def predict_risk(self, 총자산, 월매출, 인건비, 임대료, 식자재비, 기타비용, 지역, 업종):
        """5개 입력으로 위험도 예측"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")

        print("🎯 ML Risk Prediction (100% Machine Learning)")
        print("=" * 50)

        # 입력 데이터 준비
        total_cost = 인건비 + 임대료 + 식자재비 + 기타비용

        # 지역 인코딩 (unknown 처리)
        try:
            region_encoded = self.region_encoder.transform([지역])[0]
        except ValueError:
            # 모르는 지역이면 가장 첫 번째 클래스로 대체
            region_encoded = 0
            print(f"⚠️ Unknown region '{지역}', using default encoding")

        # 업종 인코딩 (unknown 처리)
        try:
            business_encoded = self.business_encoder.transform([업종])[0]
        except ValueError:
            # 모르는 업종이면 가장 첫 번째 클래스로 대체
            business_encoded = 0
            print(f"⚠️ Unknown business type '{업종}', using default encoding")

        # 피처 생성
        features = [
            np.log1p(총자산),
            np.log1p(월매출),
            np.log1p(total_cost),
            region_encoded,
            business_encoded
        ]

        X = np.array(features).reshape(1, -1)

        # ML 예측
        risk_level = self.model.predict(X)[0]
        confidence = max(self.model.predict_proba(X)[0]) * 100

        # 결과 출력
        risk_names = {1: "매우안전", 2: "안전", 3: "보통", 4: "위험", 5: "매우위험"}

        print(f"📊 입력 정보:")
        print(f"   총자산: {총자산:,}원")
        print(f"   월매출: {월매출:,}원")
        print(f"   월비용: {total_cost:,}원")
        print(f"   지역: {지역}")
        print(f"   업종: {업종}")

        print(f"\n🤖 ML 예측 결과:")
        print(f"   위험도: {risk_level} ({risk_names[risk_level]})")
        print(f"   신뢰도: {confidence:.1f}%")

        return {
            'risk_level': risk_level,
            'risk_name': risk_names[risk_level],
            'confidence': confidence,
            'model_type': '100% Machine Learning'
        }

def load_seoul_data():
    """서울 상권 데이터 로드"""
    print("📂 Loading Seoul commercial data...")

    # 데이터 파일 찾기
    data_dir = "data/raw"
    csv_files = list(Path(data_dir).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} data files")

    # 첫 번째 파일 로드 (샘플)
    df = pd.read_csv(csv_files[0], encoding='utf-8')
    print(f"✅ Loaded data: {df.shape}")

    # 필요한 컬럼만 추출 및 리네임
    required_columns = [
        '기준_년_코드', '기준_분기_코드', '상권_구분_코드', '상권_구분_코드_명',
        '시도_코드', '시도_코드_명', '시군구_코드', '시군구_코드_명',
        '행정동_코드', '행정동_코드_명', '상권_코드', '상권_코드_명',
        '서비스_업종_코드', '서비스_업종_코드_명', '당월_매출_금액',
        '점포수'
    ]

    # 컬럼 확인 및 대체
    available_cols = df.columns.tolist()
    print(f"Available columns: {len(available_cols)}")

    # 매출 관련 컬럼 찾기
    sales_cols = [col for col in available_cols if '매출' in col]
    if sales_cols:
        sales_col = sales_cols[0]
    else:
        raise ValueError("No sales column found!")

    # 간단한 데이터 생성 (실제 데이터에 맞게 조정)
    print("🔧 Creating simple training data...")

    # 샘플 데이터 생성 (실제 패턴 반영)
    n_samples = min(10000, len(df))  # 최대 1만개 샘플

    simple_data = []
    for i in range(n_samples):
        row = df.iloc[i % len(df)]

        # 기본 매출에서 파생 데이터 생성
        base_sales = abs(float(row[sales_col])) if pd.notna(row[sales_col]) else 1000000
        base_sales = max(base_sales, 100000)  # 최소값 보장

        simple_data.append({
            '총자산(원)': base_sales * np.random.uniform(3, 8),
            '월매출(원)': base_sales,
            '인건비(원)': base_sales * np.random.uniform(0.2, 0.4),
            '임대료(원)': base_sales * np.random.uniform(0.15, 0.25),
            '식자재비(원)': base_sales * np.random.uniform(0.25, 0.35),
            '기타비용(원)': base_sales * np.random.uniform(0.05, 0.15),
            '지역': row.get('시군구_코드_명', f'지역_{i%25}'),
            '업종': row.get('서비스_업종_코드_명', f'업종_{i%50}')
        })

    simple_df = pd.DataFrame(simple_data)

    # 결측값 처리
    simple_df = simple_df.fillna(method='ffill').fillna(method='bfill')

    print(f"✅ Simple training data created: {simple_df.shape}")
    return simple_df

def main():
    """메인 실행 함수"""
    print("🚀 Super Simple ML 100% System")
    print("=" * 50)
    print("🎯 Goal: 5 inputs → ML prediction → Risk level")
    print("📋 Altman Z-Score: Used for labeling only")
    print("🤖 Prediction: 100% Machine Learning")

    try:
        # 1. 데이터 로드
        df = load_seoul_data()

        # 2. 시스템 초기화
        ml_system = SuperSimpleMLSystem()

        # 3. 라벨링 (Altman Z-Score 기준)
        df = ml_system.create_simple_labels(df)

        # 4. 피처 준비 (5개 입력만)
        X = ml_system.prepare_features(df)
        y = df['risk_label'].values

        # 5. ML 모델 훈련
        accuracy = ml_system.train_simple_model(X, y)

        # 6. 모델 저장
        ml_system.save_model()

        print("\n" + "=" * 60)
        print("✅ SUPER SIMPLE ML SYSTEM READY!")
        print("🎉 100% Machine Learning Risk Prediction")
        print(f"📊 Accuracy: {accuracy:.3f}")
        print("=" * 60)

        # 7. 테스트 예측
        print("\n🧪 Testing ML prediction...")
        result = ml_system.predict_risk(
            총자산=30000000,      # 3천만원
            월매출=8000000,       # 8백만원
            인건비=2000000,       # 2백만원
            임대료=1800000,       # 180만원
            식자재비=2500000,     # 250만원
            기타비용=700000,      # 70만원
            지역='강남구',
            업종='한식음식점'
        )

        print(f"\n🎯 ML Result: {result['risk_level']} ({result['risk_name']})")
        print(f"🎯 Confidence: {result['confidence']:.1f}%")
        print(f"🎯 Model: {result['model_type']}")

        print("\n✅ 100% ML 시스템 완성!")
        print("사용법: ml_system.predict_risk(총자산, 월매출, 인건비, 임대료, 식자재비, 기타비용, 지역, 업종)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()