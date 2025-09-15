"""
Model Trainer Pipeline
ML 모델 학습 파이프라인

전체 학습 프로세스 관리:
1. 합성 데이터 생성
2. 모델 학습
3. 성능 평가
4. 모델 저장
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from pathlib import Path
import json

# 프로젝트 경로 추가
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml_pipeline.synthetic_data_generator import SyntheticDataGenerator
from src.ml_pipeline.expense_prediction_model import ExpensePredictionModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """ML 모델 학습 파이프라인 관리자"""

    def __init__(self,
                 data_path: str = "data/raw",
                 models_dir: str = "models",
                 results_dir: str = "training_results"):
        """
        Initialize the model trainer

        Args:
            data_path: Path to raw data files
            models_dir: Directory to save trained models
            results_dir: Directory to save training results
        """
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_generator = SyntheticDataGenerator(data_path)
        self.trained_models = {}
        self.training_results = {}

        logger.info(f"ModelTrainer 초기화 완료")
        logger.info(f"  데이터 경로: {data_path}")
        logger.info(f"  모델 저장 경로: {self.models_dir}")
        logger.info(f"  결과 저장 경로: {self.results_dir}")

    def generate_synthetic_data(self,
                              save_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic training data

        Args:
            save_data: Whether to save the generated data

        Returns:
            Tuple of (training_data, test_data)
        """
        logger.info("=== 합성 데이터 생성 단계 ===")

        # Generate synthetic data
        synthetic_data_path = None
        if save_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            synthetic_data_path = self.results_dir / f"synthetic_data_{timestamp}.csv"

        synthetic_data = self.data_generator.generate_training_data(
            save_path=synthetic_data_path
        )

        # Create ML dataset
        train_data, test_data = self.data_generator.create_ml_dataset(test_size=0.2)

        logger.info(f"합성 데이터 생성 완료:")
        logger.info(f"  훈련 데이터: {len(train_data):,} rows")
        logger.info(f"  테스트 데이터: {len(test_data):,} rows")

        return train_data, test_data

    def train_single_model(self,
                          model_type: str,
                          train_data: pd.DataFrame,
                          test_data: pd.DataFrame,
                          model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train a single model

        Args:
            model_type: Type of model to train ('randomforest', 'gradient_boosting')
            train_data: Training dataset
            test_data: Test dataset
            model_name: Optional custom model name

        Returns:
            Training results dictionary
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}"

        logger.info(f"=== {model_type} 모델 학습 시작 ===")

        # Initialize model
        model = ExpensePredictionModel(model_type=model_type)

        # Train the model
        training_results = model.train(
            train_data=train_data,
            validation_data=test_data
        )

        # Save the trained model
        model_path = self.models_dir / f"{model_name}.joblib"
        model.save_model(str(model_path))

        # Store results
        self.trained_models[model_name] = model
        self.training_results[model_name] = training_results

        # Save performance plots
        plot_path = self.results_dir / f"{model_name}_performance.png"
        model.plot_performance(save_path=str(plot_path))

        logger.info(f"모델 학습 완료: {model_name}")

        return training_results

    def train_all_models(self,
                        model_types: List[str] = None,
                        save_data: bool = True) -> Dict[str, Any]:
        """
        Train all specified model types

        Args:
            model_types: List of model types to train
            save_data: Whether to save generated data

        Returns:
            Comprehensive training results
        """
        if model_types is None:
            model_types = ['randomforest', 'gradient_boosting']

        logger.info("=== 전체 모델 학습 파이프라인 시작 ===")
        logger.info(f"학습할 모델 타입: {model_types}")

        start_time = datetime.now()

        # Step 1: Generate synthetic data
        train_data, test_data = self.generate_synthetic_data(save_data=save_data)

        # Step 2: Train each model type
        all_results = {}

        for model_type in model_types:
            try:
                results = self.train_single_model(
                    model_type=model_type,
                    train_data=train_data,
                    test_data=test_data
                )
                all_results[model_type] = results

            except Exception as e:
                logger.error(f"{model_type} 모델 학습 실패: {e}")
                all_results[model_type] = {'error': str(e)}

        # Step 3: Compare models and select best
        best_model_name = self.select_best_model(all_results)

        # Step 4: Save comprehensive results
        comprehensive_results = {
            'training_timestamp': start_time.isoformat(),
            'training_duration': str(datetime.now() - start_time),
            'data_size': {
                'training': len(train_data),
                'testing': len(test_data)
            },
            'models_trained': model_types,
            'individual_results': all_results,
            'best_model': best_model_name,
            'model_comparison': self.compare_models(all_results)
        }

        # Save results to file
        results_file = self.results_dir / f"training_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"전체 학습 완료 - 결과 저장: {results_file}")
        logger.info(f"최고 성능 모델: {best_model_name}")

        return comprehensive_results

    def select_best_model(self, results: Dict[str, Any]) -> str:
        """
        Select the best performing model based on validation metrics

        Args:
            results: Training results for all models

        Returns:
            Name of the best model
        """
        best_model = None
        best_score = float('inf')  # Lower is better for MAE

        for model_name, result in results.items():
            if 'error' in result:
                continue

            # Use validation MAE as primary metric
            validation_metrics = result.get('validation_metrics', {})
            if validation_metrics:
                mae = validation_metrics.get('overall', {}).get('mae', float('inf'))
            else:
                # Fall back to training MAE if no validation
                mae = result.get('training_metrics', {}).get('overall', {}).get('mae', float('inf'))

            if mae < best_score:
                best_score = mae
                best_model = model_name

        logger.info(f"최고 성능 모델 선정: {best_model} (MAE: {best_score:,.0f}원)")

        return best_model

    def compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create model comparison summary

        Args:
            results: Training results for all models

        Returns:
            Model comparison summary
        """
        comparison = {
            'performance_ranking': [],
            'metric_comparison': {}
        }

        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if not valid_results:
            return comparison

        # Extract key metrics for comparison
        metrics_data = []

        for model_name, result in valid_results.items():
            training_metrics = result.get('training_metrics', {}).get('overall', {})
            validation_metrics = result.get('validation_metrics', {}).get('overall', {})

            model_metrics = {
                'model': model_name,
                'train_mae': training_metrics.get('mae', 0),
                'train_r2': training_metrics.get('r2', 0),
                'val_mae': validation_metrics.get('mae', training_metrics.get('mae', 0)),
                'val_r2': validation_metrics.get('r2', training_metrics.get('r2', 0))
            }
            metrics_data.append(model_metrics)

        # Sort by validation MAE (lower is better)
        metrics_data.sort(key=lambda x: x['val_mae'])

        # Create ranking
        comparison['performance_ranking'] = [
            {
                'rank': i + 1,
                'model': model['model'],
                'validation_mae': model['val_mae'],
                'validation_r2': model['val_r2']
            }
            for i, model in enumerate(metrics_data)
        ]

        # Create detailed metric comparison
        comparison['metric_comparison'] = {
            'mae': {model['model']: model['val_mae'] for model in metrics_data},
            'r2': {model['model']: model['val_r2'] for model in metrics_data}
        }

        return comparison

    def load_best_model(self, results_file: Optional[str] = None) -> ExpensePredictionModel:
        """
        Load the best performing model

        Args:
            results_file: Optional specific results file to use

        Returns:
            Loaded ExpensePredictionModel
        """
        if results_file is None:
            # Find the most recent results file
            results_files = list(self.results_dir.glob("training_results_*.json"))
            if not results_files:
                raise FileNotFoundError("훈련 결과 파일을 찾을 수 없습니다.")

            results_file = max(results_files, key=lambda x: x.stat().st_mtime)

        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        best_model_name = results['best_model']

        # Find the model file
        model_files = list(self.models_dir.glob(f"{best_model_name}*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"최고 성능 모델 파일을 찾을 수 없습니다: {best_model_name}")

        model_file = model_files[0]

        # Load and return the model
        model = ExpensePredictionModel()
        model.load_model(str(model_file))

        logger.info(f"최고 성능 모델 로딩 완료: {model_file}")

        return model

    def run_full_pipeline(self,
                         model_types: List[str] = None,
                         evaluate_performance: bool = True) -> Dict[str, Any]:
        """
        Run the complete ML training pipeline

        Args:
            model_types: Model types to train
            evaluate_performance: Whether to run comprehensive evaluation

        Returns:
            Complete pipeline results
        """
        logger.info("=== 전체 ML 파이프라인 실행 ===")

        # Step 1: Train all models
        training_results = self.train_all_models(model_types=model_types)

        # Step 2: Load best model and evaluate
        if evaluate_performance:
            try:
                best_model = self.load_best_model()
                logger.info("✅ 최고 성능 모델 로딩 완료")

            except Exception as e:
                logger.warning(f"성능 평가 실패: {e}")
                training_results['performance_evaluation'] = {'error': str(e)}

        logger.info("=== 전체 파이프라인 완료 ===")

        return training_results


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()

    # Run complete pipeline
    results = trainer.run_full_pipeline(
        model_types=['randomforest', 'gradient_boosting']
    )

    print("파이프라인 실행 완료!")
    print(f"최고 성능 모델: {results['best_model']}")

    # Load and test the best model
    best_model = trainer.load_best_model()

    # Test prediction
    test_prediction = best_model.predict(
        revenue=8_000_000,
        industry_code="CS100001",
        region="11110"
    )

    print("\n테스트 예측 결과:")
    for category, amount in test_prediction.items():
        print(f"  {category}: {amount:,.0f}원")