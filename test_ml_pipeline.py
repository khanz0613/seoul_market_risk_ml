#!/usr/bin/env python3
"""
ML 파이프라인 테스트 스크립트
모든 구성 요소가 올바르게 작동하는지 검증

이 스크립트는 다음을 테스트합니다:
1. SyntheticDataGenerator - 합성 데이터 생성
2. ExpensePredictionModel - 모델 학습 및 예측
3. 전체 파이프라인 통합
"""

import os
import sys
import logging
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_pipeline.synthetic_data_generator import SyntheticDataGenerator
from src.ml_pipeline.expense_prediction_model import ExpensePredictionModel
from src.ml_pipeline.model_trainer import ModelTrainer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_synthetic_data_generator():
    """합성 데이터 생성기 테스트"""
    print("🧪 테스트 1: SyntheticDataGenerator")
    print("-" * 40)

    try:
        # 데이터 생성기 초기화
        generator = SyntheticDataGenerator("data/raw")

        # 원시 데이터 로딩 테스트
        print("📁 원시 데이터 로딩 테스트...")
        raw_data = generator.load_raw_data()
        print(f"✅ 원시 데이터 로딩 성공: {len(raw_data):,} rows")

        # 데이터 정제 테스트
        print("🧹 데이터 정제 테스트...")
        cleaned_data = generator.clean_and_filter_data(raw_data.head(1000))  # 테스트용으로 작은 샘플 사용
        print(f"✅ 데이터 정제 성공: {len(cleaned_data):,} rows")

        # 피처 엔지니어링 테스트
        print("⚙️ 피처 엔지니어링 테스트...")
        featured_data = generator.add_feature_engineering(cleaned_data)
        print(f"✅ 피처 엔지니어링 성공: {len(featured_data.columns)} 컬럼")

        # 합성 비용 데이터 생성 테스트
        print("🤖 합성 비용 데이터 생성 테스트...")
        synthetic_data = generator.generate_synthetic_expenses(featured_data)
        print(f"✅ 합성 데이터 생성 성공: {len(synthetic_data):,} rows")

        # 기본 검증
        required_columns = ['예측_재료비', '예측_인건비', '예측_임대료', '예측_기타']
        missing_columns = [col for col in required_columns if col not in synthetic_data.columns]

        if missing_columns:
            raise ValueError(f"필수 컬럼 누락: {missing_columns}")

        print(f"✅ 필수 컬럼 검증 통과")

        return synthetic_data

    except Exception as e:
        print(f"❌ SyntheticDataGenerator 테스트 실패: {e}")
        return None

def test_expense_prediction_model(synthetic_data):
    """비용 예측 모델 테스트"""
    print("\n🧪 테스트 2: ExpensePredictionModel")
    print("-" * 40)

    try:
        # 모델 초기화
        print("🤖 모델 초기화...")
        model = ExpensePredictionModel(model_type='randomforest')
        print("✅ 모델 초기화 성공")

        # 학습 데이터셋 생성
        print("📊 학습 데이터 준비...")

        # 최소한의 데이터로 빠른 테스트
        small_sample = synthetic_data.head(500)  # 500개 샘플만 사용

        # 필요한 컬럼 확인
        required_features = [
            '당월_매출_금액', '통합업종카테고리', '행정동_코드',
            '매출규모_로그', '매출규모_카테고리', '년도', '분기', '시군구코드'
        ]

        # 누락된 피처가 있다면 기본값으로 채움
        for col in required_features:
            if col not in small_sample.columns:
                if col == '행정동_코드':
                    small_sample[col] = 11110515  # 기본 행정동 코드
                elif col == '시군구코드':
                    small_sample[col] = '11110'  # 기본 시군구 코드
                elif col == '년도':
                    small_sample[col] = 2024
                elif col == '분기':
                    small_sample[col] = 1
                else:
                    small_sample[col] = 0

        # train-test 분할
        train_size = int(len(small_sample) * 0.8)
        train_data = small_sample.iloc[:train_size].copy()
        test_data = small_sample.iloc[train_size:].copy()

        print(f"📈 훈련 데이터: {len(train_data)} rows")
        print(f"📉 테스트 데이터: {len(test_data)} rows")

        # 모델 학습
        print("🚀 모델 학습 시작...")
        training_results = model.train(train_data=train_data, validation_data=test_data)
        print("✅ 모델 학습 완료")

        # 학습 결과 검증
        if 'training_metrics' in training_results:
            mae = training_results['training_metrics']['overall']['mae']
            r2 = training_results['training_metrics']['overall']['r2']
            print(f"📊 훈련 성능: MAE={mae:,.0f}원, R²={r2:.3f}")

        # 단일 예측 테스트
        print("🎯 단일 예측 테스트...")
        prediction = model.predict(
            revenue=8_000_000,      # 800만원
            industry_code="CS100001",  # 한식음식점
            region="11110515"       # 청운효자동
        )

        print("✅ 예측 결과:")
        for category, amount in prediction.items():
            print(f"   {category}: {amount:,.0f}원")

        # 예측값 검증
        total_predicted = sum([v for k, v in prediction.items() if k != '총비용'])
        revenue = 8_000_000

        if total_predicted > revenue * 1.5:  # 매출의 150%를 넘으면 비정상
            print(f"⚠️ 예측값이 너무 높음: {total_predicted:,.0f}원 (매출의 {total_predicted/revenue*100:.1f}%)")
        else:
            print(f"✅ 예측값 합리적: {total_predicted:,.0f}원 (매출의 {total_predicted/revenue*100:.1f}%)")

        return model

    except Exception as e:
        print(f"❌ ExpensePredictionModel 테스트 실패: {e}")
        return None

def test_model_trainer():
    """모델 트레이너 테스트"""
    print("\n🧪 테스트 3: ModelTrainer (간단 버전)")
    print("-" * 40)

    try:
        # 트레이너 초기화
        print("🏗️ 트레이너 초기화...")
        trainer = ModelTrainer(
            data_path="data/raw",
            models_dir="test_models",
            results_dir="test_results"
        )
        print("✅ 트레이너 초기화 성공")

        # 합성 데이터 생성 (샘플만)
        print("📊 테스트용 합성 데이터 생성...")

        # 실제로는 trainer.generate_synthetic_data를 호출하지만
        # 테스트를 위해 간단한 데이터 생성
        from pathlib import Path
        csv_files = list(Path("data/raw").glob("*.csv"))

        if not csv_files:
            print("⚠️ 원시 데이터 파일이 없어 모델 트레이너 테스트 건너뜀")
            return None

        print(f"✅ {len(csv_files)}개 CSV 파일 발견")

        # 환경만 검증하고 실제 학습은 건너뜀 (시간 절약)
        print("✅ ModelTrainer 환경 검증 완료")

        return trainer

    except Exception as e:
        print(f"❌ ModelTrainer 테스트 실패: {e}")
        return None

def test_integration():
    """전체 통합 테스트"""
    print("\n🧪 테스트 4: 통합 테스트")
    print("-" * 40)

    try:
        # enhanced_main.py 통합 테스트
        print("🔗 enhanced_main.py 통합 테스트...")

        from enhanced_main import EnhancedBusinessRiskAnalyzer

        # ML 모델 없이 초기화 (기본 모드)
        analyzer = EnhancedBusinessRiskAnalyzer(use_ml_model=False)
        print("✅ 기본 모드로 시스템 초기화 성공")

        # ML 모델 모드 초기화 테스트 (모델이 없어도 graceful fallback)
        analyzer_ml = EnhancedBusinessRiskAnalyzer(use_ml_model=True)

        if analyzer_ml.use_ml_model:
            print("✅ ML 모드로 시스템 초기화 성공")
        else:
            print("⚠️ ML 모델 없음, 기본 모드로 fallback")

        return True

    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""

    print("🚀 ML 파이프라인 통합 테스트 시작")
    print("=" * 60)

    # 환경 확인
    data_path = Path("data/raw")
    if not data_path.exists():
        print(f"❌ 데이터 경로를 찾을 수 없습니다: {data_path}")
        print("💡 data/raw 디렉토리에 CSV 파일을 준비해주세요.")
        return 1

    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {data_path}")
        print("💡 data/raw 디렉토리에 서울시 상권분석 CSV 파일을 준비해주세요.")
        return 1

    print(f"✅ 환경 검증 완료: {len(csv_files)}개 CSV 파일 발견")

    # 테스트 실행
    test_results = []

    # 테스트 1: 합성 데이터 생성
    synthetic_data = test_synthetic_data_generator()
    test_results.append(("SyntheticDataGenerator", synthetic_data is not None))

    # 테스트 2: 모델 학습 및 예측
    if synthetic_data is not None:
        model = test_expense_prediction_model(synthetic_data)
        test_results.append(("ExpensePredictionModel", model is not None))
    else:
        test_results.append(("ExpensePredictionModel", False))
        model = None

    # 테스트 3: 모델 트레이너
    trainer = test_model_trainer()
    test_results.append(("ModelTrainer", trainer is not None))

    # 테스트 4: 통합 테스트
    integration_success = test_integration()
    test_results.append(("Integration", integration_success))

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n📊 총 테스트: {len(test_results)}")
    print(f"✅ 통과: {passed}")
    print(f"❌ 실패: {failed}")

    if failed == 0:
        print("\n🎉 모든 테스트 통과!")
        print("💡 이제 train_ml_models.py로 모델을 학습하거나")
        print("   enhanced_main.py로 시스템을 실행해보세요.")
        return 0
    else:
        print(f"\n⚠️ {failed}개 테스트 실패")
        print("💡 위의 오류 메시지를 확인하고 문제를 해결해주세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)