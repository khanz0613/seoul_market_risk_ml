#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Seoul Market Risk ML Training Pipeline
목표: 로그 변환된 타겟으로 MAE 대폭 감소
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedMarketRiskTrainer:
    def __init__(self):
        """향상된 모델 훈련기 초기화"""
        self.models = {}
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_preprocessed_data(self):
        """전처리된 데이터 로드"""
        print("📥 전처리된 데이터 로딩...")

        try:
            X = joblib.load('data/processed/X_enhanced.joblib')
            y = joblib.load('data/processed/y_enhanced.joblib')
            features = joblib.load('data/processed/features_enhanced.joblib')

            print(f"✅ 데이터 로드 완료: {len(X):,} 샘플, {len(features)} 피처")
            return X, y, features

        except FileNotFoundError:
            print("❌ 전처리된 데이터가 없습니다. enhanced_preprocessing.py를 먼저 실행하세요.")
            raise

    def create_models(self):
        """향상된 모델 설정"""
        print("🤖 모델 설정...")

        self.models = {
            'enhanced_randomforest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'enhanced_gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'regularized_linear': Ridge(
                alpha=100.0,
                random_state=42
            )
        }

        print(f"  생성된 모델 수: {len(self.models)}")

    def train_and_evaluate_model(self, model_name, model, X_train, X_test, y_train, y_test):
        """모델 훈련 및 평가"""
        print(f"🎯 {model_name} 훈련 중...")

        # 모델 훈련
        model.fit(X_train, y_train)

        # 예측
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 로그 공간에서의 메트릭
        train_mae_log = mean_absolute_error(y_train, y_train_pred)
        test_mae_log = mean_absolute_error(y_test, y_test_pred)
        test_r2_log = r2_score(y_test, y_test_pred)

        # 원본 공간으로 변환하여 실제 MAE 계산
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)
        y_train_pred_orig = np.expm1(y_train_pred)
        y_test_pred_orig = np.expm1(y_test_pred)

        train_mae_orig = mean_absolute_error(y_train_orig, y_train_pred_orig)
        test_mae_orig = mean_absolute_error(y_test_orig, y_test_pred_orig)
        test_r2_orig = r2_score(y_test_orig, y_test_pred_orig)

        # 교차 검증 (로그 공간에서)
        cv_scores = cross_val_score(model, X_train, y_train,
                                   cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        results = {
            'model_name': model_name,
            'log_space_metrics': {
                'train_mae': train_mae_log,
                'test_mae': test_mae_log,
                'test_r2': test_r2_log,
                'cv_mae_mean': cv_mae,
                'cv_mae_std': cv_std
            },
            'original_space_metrics': {
                'train_mae': train_mae_orig,
                'test_mae': test_mae_orig,
                'test_r2': test_r2_orig
            }
        }

        # 모델 저장
        model_path = f'models/enhanced_{model_name}_{self.timestamp}.joblib'
        joblib.dump(model, model_path)
        results['model_path'] = model_path

        print(f"  원본공간 MAE: {test_mae_orig:,.0f}원")
        print(f"  원본공간 R²: {test_r2_orig:.3f}")
        print(f"  로그공간 MAE: {test_mae_log:.3f}")

        return results

    def compare_models(self):
        """모델 성능 비교"""
        print("📊 모델 성능 비교...")

        comparison = []
        for model_name, result in self.results.items():
            comparison.append({
                'model': model_name,
                'original_mae': result['original_space_metrics']['test_mae'],
                'original_r2': result['original_space_metrics']['test_r2'],
                'log_mae': result['log_space_metrics']['test_mae'],
                'log_r2': result['log_space_metrics']['test_r2']
            })

        # 원본 공간 MAE 기준으로 정렬
        comparison.sort(key=lambda x: x['original_mae'])

        print("\n🏆 모델 순위 (원본공간 MAE 기준):")
        for i, comp in enumerate(comparison, 1):
            print(f"  {i}. {comp['model']}")
            print(f"     원본 MAE: {comp['original_mae']:,.0f}원")
            print(f"     원본 R²: {comp['original_r2']:.3f}")
            print(f"     로그 MAE: {comp['log_mae']:.3f}")
            print()

        return comparison

    def save_results(self, comparison):
        """결과 저장"""
        print("💾 결과 저장...")

        # 결과 디렉토리 생성
        os.makedirs('training_results', exist_ok=True)

        # 상세 결과 저장
        detailed_results = {
            'timestamp': self.timestamp,
            'enhanced_preprocessing': True,
            'log_transformation': True,
            'geographic_clustering': True,
            'models_trained': list(self.results.keys()),
            'detailed_results': self.results,
            'performance_comparison': comparison,
            'best_model': comparison[0]['model'],
            'improvement_notes': {
                'target_transformation': 'log1p applied to reduce skewness',
                'geographic_features': '6-tier Seoul district clustering',
                'feature_engineering': 'time patterns, demographics, relative performance',
                'data_utilization': 'All available years (2020-2024) combined'
            }
        }

        # JSON 저장
        result_file = f'training_results/enhanced_training_results_{self.timestamp}.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)

        # 간단한 요약 저장
        summary_file = f'training_results/enhanced_summary_{self.timestamp}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("🎯 Enhanced Seoul Market Risk ML Training Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Improvements Applied: ✅\n")
            f.write(f"  - Log transformation: ✅\n")
            f.write(f"  - Geographic clustering: ✅ (6 tiers)\n")
            f.write(f"  - Enhanced features: ✅\n")
            f.write(f"  - Maximum data usage: ✅\n\n")

            f.write("🏆 Model Performance (Original Space):\n")
            for i, comp in enumerate(comparison, 1):
                f.write(f"{i}. {comp['model']}\n")
                f.write(f"   MAE: {comp['original_mae']:,.0f}원\n")
                f.write(f"   R²: {comp['original_r2']:.3f}\n\n")

            f.write(f"💡 Best Model: {comparison[0]['model']}\n")
            f.write(f"🎯 Target MAE (500K): {'✅ 달성' if comparison[0]['original_mae'] <= 500000 else '❌ 미달성'}\n")

        print(f"✅ 상세결과: {result_file}")
        print(f"✅ 요약: {summary_file}")

    def run_training_pipeline(self):
        """전체 훈련 파이프라인 실행"""
        print("🚀 Enhanced Training Pipeline 시작...")
        print("=" * 60)

        # 1. 데이터 로드
        X, y, features = self.load_preprocessed_data()

        # 2. 훈련/테스트 분할
        print("🔄 데이터 분할...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        print(f"  훈련: {len(X_train):,}, 테스트: {len(X_test):,}")

        # 3. 모델 생성
        self.create_models()

        # 4. 모델 훈련 및 평가
        print("\n🎯 모델 훈련 시작...")
        for model_name, model in self.models.items():
            result = self.train_and_evaluate_model(
                model_name, model, X_train, X_test, y_train, y_test
            )
            self.results[model_name] = result

        # 5. 성능 비교
        print("\n" + "=" * 60)
        comparison = self.compare_models()

        # 6. 결과 저장
        self.save_results(comparison)

        print("=" * 60)
        print("✅ Enhanced Training 완료!")

        # 개선 효과 출력
        best_mae = comparison[0]['original_mae']
        target_mae = 500000
        previous_mae = 33785880  # 이전 결과

        print(f"\n📈 개선 효과:")
        print(f"  이전 MAE: {previous_mae:,.0f}원")
        print(f"  현재 MAE: {best_mae:,.0f}원")
        print(f"  개선율: {((previous_mae - best_mae) / previous_mae * 100):.1f}%")
        print(f"  목표 달성: {'✅' if best_mae <= target_mae else '❌'}")

        return comparison

if __name__ == "__main__":
    # 전처리된 데이터 디렉토리 생성
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 훈련 실행
    trainer = EnhancedMarketRiskTrainer()
    results = trainer.run_training_pipeline()

    print(f"\n🎯 다음 단계:")
    print(f"  1. 결과 확인: training_results/ 폴더")
    print(f"  2. 최적 모델: models/ 폴더")
    print(f"  3. 추가 튜닝이 필요하면 하이퍼파라미터 조정")