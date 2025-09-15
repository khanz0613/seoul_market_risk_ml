#!/usr/bin/env python3
"""
ML 모델 학습 실행 스크립트
소상공인 비용 예측 모델 학습

사용법:
    python train_ml_models.py [options]

옵션:
    --model-types: 학습할 모델 타입 (기본값: randomforest,gradient_boosting)
    --data-path: 원시 데이터 경로 (기본값: data/raw)
    --models-dir: 모델 저장 경로 (기본값: models)
    --quick: 빠른 테스트용 (작은 데이터셋)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_pipeline.model_trainer import ModelTrainer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="소상공인 비용 예측 ML 모델 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-types",
        type=str,
        default="randomforest,gradient_boosting",
        help="학습할 모델 타입 (쉼표로 구분)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw",
        help="원시 데이터 경로"
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="모델 저장 디렉토리"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="training_results",
        help="학습 결과 저장 디렉토리"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="빠른 테스트 모드 (RandomForest만 학습)"
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="성능 플롯 생성 건너뛰기"
    )

    return parser.parse_args()

def validate_environment(args):
    """실행 환경 검증"""
    logger.info("=== 실행 환경 검증 ===")

    # 데이터 디렉토리 확인
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {data_path}")

    # CSV 파일 확인
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {data_path}")

    logger.info(f"✅ 데이터 경로 확인: {data_path}")
    logger.info(f"✅ CSV 파일 개수: {len(csv_files)}")

    # 필요한 디렉토리 생성
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"✅ 모델 저장 경로: {args.models_dir}")
    logger.info(f"✅ 결과 저장 경로: {args.results_dir}")

def main():
    """메인 실행 함수"""

    print("🚀 소상공인 비용 예측 ML 모델 학습 시작")
    print("=" * 60)

    # 인수 파싱
    args = parse_arguments()

    # 환경 검증
    try:
        validate_environment(args)
    except Exception as e:
        logger.error(f"환경 검증 실패: {e}")
        print(f"❌ 환경 검증 실패: {e}")
        return 1

    # 모델 타입 처리
    if args.quick:
        model_types = ['randomforest']
        logger.info("🏃‍♂️ 빠른 테스트 모드: RandomForest만 학습")
    else:
        model_types = [m.strip() for m in args.model_types.split(',')]
        logger.info(f"🎯 학습할 모델: {model_types}")

    # 지원되는 모델 타입 확인
    supported_models = ['randomforest', 'gradient_boosting']
    invalid_models = [m for m in model_types if m not in supported_models]
    if invalid_models:
        logger.error(f"지원하지 않는 모델 타입: {invalid_models}")
        logger.info(f"지원되는 모델: {supported_models}")
        return 1

    # 모델 트레이너 초기화
    try:
        trainer = ModelTrainer(
            data_path=args.data_path,
            models_dir=args.models_dir,
            results_dir=args.results_dir
        )
        logger.info("✅ 모델 트레이너 초기화 완료")

    except Exception as e:
        logger.error(f"트레이너 초기화 실패: {e}")
        return 1

    # 전체 파이프라인 실행
    try:
        logger.info("🚀 ML 학습 파이프라인 시작")

        results = trainer.run_full_pipeline(
            model_types=model_types,
            evaluate_performance=True
        )

        # 결과 요약 출력
        print("\n" + "=" * 60)
        print("📊 학습 결과 요약")
        print("=" * 60)

        if 'best_model' in results:
            print(f"🏆 최고 성능 모델: {results['best_model']}")

        if 'data_size' in results:
            train_size = results['data_size']['training']
            test_size = results['data_size']['testing']
            print(f"📈 훈련 데이터: {train_size:,} rows")
            print(f"📉 테스트 데이터: {test_size:,} rows")

        # 개별 모델 성능
        if 'individual_results' in results:
            print(f"\n🎯 개별 모델 성능:")
            for model_name, model_result in results['individual_results'].items():
                if 'error' in model_result:
                    print(f"  ❌ {model_name}: 학습 실패")
                else:
                    # 검증 성능 추출
                    val_metrics = model_result.get('validation_metrics', {})
                    if val_metrics:
                        mae = val_metrics.get('overall', {}).get('mae', 0)
                        r2 = val_metrics.get('overall', {}).get('r2', 0)
                        print(f"  ✅ {model_name}: MAE={mae:,.0f}원, R²={r2:.3f}")
                    else:
                        print(f"  ⚠️ {model_name}: 검증 데이터 없음")

        # 성능 목표 달성 여부
        print(f"\n🎯 성능 목표 달성도:")

        best_model_result = None
        if 'best_model' in results and results['best_model'] in results.get('individual_results', {}):
            best_model_result = results['individual_results'][results['best_model']]

        if best_model_result and 'validation_metrics' in best_model_result:
            val_metrics = best_model_result['validation_metrics']['overall']
            mae = val_metrics.get('mae', float('inf'))
            r2 = val_metrics.get('r2', 0)

            # MAE < 500,000원 목표
            mae_goal = 500_000
            mae_status = "✅" if mae < mae_goal else "❌"
            print(f"  {mae_status} MAE < 50만원: {mae:,.0f}원 (목표: {mae_goal:,.0f}원)")

            # R² > 0.7 목표
            r2_goal = 0.7
            r2_status = "✅" if r2 > r2_goal else "❌"
            print(f"  {r2_status} R² > 0.7: {r2:.3f} (목표: {r2_goal:.1f})")

        else:
            print("  ❓ 성능 지표를 확인할 수 없습니다")

        # 학습된 모델 경로
        print(f"\n💾 학습된 모델 위치:")
        models_dir = Path(args.models_dir)
        model_files = list(models_dir.glob("*.joblib"))
        for model_file in model_files:
            print(f"  📁 {model_file}")

        print(f"\n📋 상세 결과:")
        results_dir = Path(args.results_dir)
        result_files = list(results_dir.glob("training_results_*.json"))
        if result_files:
            latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
            print(f"  📄 {latest_result}")

        print("\n" + "=" * 60)
        print("🎉 ML 모델 학습 완료!")
        print("=" * 60)

        # 다음 단계 안내
        print("\n📖 다음 단계:")
        print("1. python enhanced_main.py  # AI 예측 모드로 시스템 실행")
        print("2. 또는 직접 ExpensePredictionModel을 import하여 사용")

        return 0

    except Exception as e:
        logger.error(f"학습 파이프라인 실행 실패: {e}")
        print(f"\n❌ 학습 실패: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)