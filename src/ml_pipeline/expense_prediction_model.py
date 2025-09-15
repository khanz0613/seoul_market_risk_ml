"""
Expense Prediction Model
소상공인 비용 예측 ML 모델

다중 출력 회귀를 통해 4개 비용 카테고리 동시 예측:
- 재료비 (Materials)
- 인건비 (Labor)
- 임대료 (Rent)
- 기타 (Other)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정하여 GUI 없이 이미지만 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트 설정"""
    try:
        # macOS의 경우 Apple SD Gothic Neo 사용
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        try:
            # 다른 시스템에서 사용할 수 있는 폰트들
            available_fonts = [font.name for font in fm.fontManager.ttflist]
            korean_fonts = [
                'Malgun Gothic', 'NanumGothic', 'NanumBarunGothic',
                'AppleSDGothicNeo-Regular', 'Apple SD Gothic Neo'
            ]

            for font in korean_fonts:
                if font in available_fonts:
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False
                    break
            else:
                # 한글 폰트가 없으면 영어로 표시하고 백엔드 모드로 진행
                pass
        except:
            # 폰트 설정 실패해도 계속 진행
            pass

set_korean_font()
import seaborn as sns

# 프로젝트 경로 추가
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_processing.industry_mapper import IndustryMapper

logger = logging.getLogger(__name__)

class ExpensePredictionModel:
    """소상공인 비용 예측 ML 모델"""

    def __init__(self,
                 model_type: str = 'randomforest',
                 random_state: int = 42):
        """
        Initialize the expense prediction model

        Args:
            model_type: Type of ML model ('randomforest', 'gradient_boosting')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_encoders = {}
        self.feature_scaler = None
        self.is_trained = False
        self.feature_columns = None
        self.target_columns = ['예측_재료비', '예측_인건비', '예측_임대료', '예측_기타']
        self.industry_mapper = IndustryMapper()

        # Model performance metrics
        self.training_metrics = {}
        self.validation_metrics = {}

        # Initialize the model
        self.model = self._create_model()

        logger.info(f"ExpensePredictionModel 초기화 완료 - 모델타입: {model_type}")

    def _create_model(self) -> MultiOutputRegressor:
        """Create the underlying ML model"""

        if self.model_type == 'randomforest':
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=1  # 메모리 누수 방지
            )
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            base_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")

        # Wrap in MultiOutputRegressor for multiple target prediction
        return MultiOutputRegressor(base_model)

    def prepare_features(self,
                        data: pd.DataFrame,
                        is_training: bool = True) -> pd.DataFrame:
        """
        Prepare features for ML training/prediction

        Args:
            data: Input data with features
            is_training: Whether this is for training (affects encoding)

        Returns:
            Processed feature DataFrame
        """
        logger.info(f"피처 전처리 시작 - 훈련모드: {is_training}")

        # Select feature columns
        if self.feature_columns is None:
            self.feature_columns = [
                '당월_매출_금액',
                '통합업종카테고리',
                '행정동_코드',
                '매출규모_로그',
                '매출규모_카테고리',
                '년도',
                '분기',
                '시군구코드'
            ]

        # Ensure all feature columns exist
        available_columns = [col for col in self.feature_columns if col in data.columns]
        if len(available_columns) != len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_columns)
            logger.warning(f"누락된 피처 컬럼: {missing}")

        X = data[available_columns].copy()

        # Handle categorical variables
        categorical_columns = ['통합업종카테고리', '매출규모_카테고리', '시군구코드']

        for col in categorical_columns:
            if col in X.columns:
                if is_training:
                    # Create and fit encoder during training
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col].astype(str))
                    self.feature_encoders[col] = encoder
                else:
                    # Use existing encoder for prediction
                    if col in self.feature_encoders:
                        encoder = self.feature_encoders[col]
                        # Handle unknown categories
                        known_classes = set(encoder.classes_)
                        X[col] = X[col].astype(str).apply(
                            lambda x: x if x in known_classes else encoder.classes_[0]
                        )
                        X[col] = encoder.transform(X[col])
                    else:
                        logger.warning(f"인코더를 찾을 수 없음: {col}")

        # Scale numerical features
        numerical_columns = ['당월_매출_금액', '매출규모_로그', '년도', '분기', '행정동_코드']
        numerical_columns = [col for col in numerical_columns if col in X.columns]

        if numerical_columns:
            if is_training:
                self.feature_scaler = StandardScaler()
                X[numerical_columns] = self.feature_scaler.fit_transform(X[numerical_columns])
            else:
                if self.feature_scaler is not None:
                    X[numerical_columns] = self.feature_scaler.transform(X[numerical_columns])

        logger.info(f"피처 전처리 완료 - 피처 수: {len(X.columns)}")

        return X

    def train(self,
              train_data: pd.DataFrame,
              validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the expense prediction model

        Args:
            train_data: Training dataset with features and targets
            validation_data: Optional validation dataset

        Returns:
            Training metrics and performance summary
        """
        logger.info("=== 모델 학습 시작 ===")
        logger.info(f"훈련 데이터 크기: {train_data.shape}")

        # Prepare features and targets
        X_train = self.prepare_features(train_data, is_training=True)
        y_train = train_data[self.target_columns].values

        logger.info(f"훈련 피처 크기: {X_train.shape}")
        logger.info(f"훈련 타겟 크기: {y_train.shape}")

        # Train the model
        logger.info("모델 학습 중...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, "Training")

        # Validation metrics if validation data provided
        if validation_data is not None:
            logger.info("검증 데이터 평가 중...")
            X_val = self.prepare_features(validation_data, is_training=False)
            y_val = validation_data[self.target_columns].values
            y_val_pred = self.model.predict(X_val)
            self.validation_metrics = self._calculate_metrics(y_val, y_val_pred, "Validation")

        # Cross-validation scores
        cv_scores = self._perform_cross_validation(X_train, y_train)

        logger.info("=== 모델 학습 완료 ===")

        # Return comprehensive training results
        return {
            'model_type': self.model_type,
            'training_data_size': len(train_data),
            'feature_count': len(X_train.columns),
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'cross_validation': cv_scores,
            'is_trained': self.is_trained
        }

    def predict(self,
                revenue: float,
                industry_code: str,
                region: Optional[str] = None,
                **kwargs) -> Dict[str, float]:
        """
        Predict expenses for a single business case

        Args:
            revenue: Monthly revenue (당월_매출_금액)
            industry_code: Industry code (서비스_업종_코드)
            region: Administrative region code (optional)
            **kwargs: Additional features

        Returns:
            Dictionary with predicted expenses for each category
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")

        # Create input DataFrame
        input_data = {
            '당월_매출_금액': revenue,
            '서비스_업종_코드': industry_code,
        }

        if region:
            input_data['행정동_코드'] = region

        # Add additional features from kwargs
        input_data.update(kwargs)

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Map industry code to category
        if industry_code in self.industry_mapper.INDUSTRY_MAPPING:
            input_df['통합업종카테고리'] = self.industry_mapper.INDUSTRY_MAPPING[industry_code]
        else:
            logger.warning(f"알 수 없는 업종 코드: {industry_code}")
            input_df['통합업종카테고리'] = '개인서비스업'  # Default

        # Add derived features
        input_df['매출규모_로그'] = np.log1p(revenue)

        # Revenue scale categories
        if revenue <= 1_000_000:
            scale = '소규모'
        elif revenue <= 5_000_000:
            scale = '중소규모'
        elif revenue <= 20_000_000:
            scale = '중규모'
        else:
            scale = '대규모'
        input_df['매출규모_카테고리'] = scale

        # Default values for missing features
        input_df['년도'] = input_df.get('년도', 2024)
        input_df['분기'] = input_df.get('분기', 1)

        if region:
            input_df['시군구코드'] = str(region)[:5]
        else:
            input_df['시군구코드'] = '11110'  # Default Seoul region

        # Prepare features
        X = self.prepare_features(input_df, is_training=False)

        # Make prediction
        prediction = self.model.predict(X)[0]  # Get first (and only) prediction

        # Return as dictionary
        result = {
            '재료비': max(0, prediction[0]),
            '인건비': max(0, prediction[1]),
            '임대료': max(0, prediction[2]),
            '기타': max(0, prediction[3])
        }

        # Add total
        result['총비용'] = sum(result.values())

        return result

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expenses for multiple cases

        Args:
            data: DataFrame with input features

        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")

        logger.info(f"배치 예측 시작 - 데이터 크기: {data.shape}")

        # Prepare features
        X = self.prepare_features(data, is_training=False)

        # Make predictions
        predictions = self.model.predict(X)

        # Create result DataFrame
        result_df = data.copy()

        for i, col in enumerate(self.target_columns):
            result_df[col] = np.maximum(0, predictions[:, i])

        # Add total expenses
        result_df['예측_총비용'] = (
            result_df['예측_재료비'] +
            result_df['예측_인건비'] +
            result_df['예측_임대료'] +
            result_df['예측_기타']
        )

        logger.info("배치 예측 완료")

        return result_df

    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          dataset_name: str) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""

        metrics = {}

        # Overall metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        metrics['overall'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        # Per-category metrics
        category_names = ['재료비', '인건비', '임대료', '기타']
        metrics['by_category'] = {}

        for i, category in enumerate(category_names):
            cat_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            cat_rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            cat_r2 = r2_score(y_true[:, i], y_pred[:, i])

            metrics['by_category'][category] = {
                'mae': cat_mae,
                'rmse': cat_rmse,
                'r2': cat_r2
            }

        logger.info(f"\n=== {dataset_name} 성능 지표 ===")
        logger.info(f"전체 MAE: {mae:,.0f}원")
        logger.info(f"전체 RMSE: {rmse:,.0f}원")
        logger.info(f"전체 R²: {r2:.3f}")

        logger.info(f"\n카테고리별 성능:")
        for category, cat_metrics in metrics['by_category'].items():
            logger.info(f"  {category}: MAE={cat_metrics['mae']:,.0f}원, R²={cat_metrics['r2']:.3f}")

        return metrics

    def _perform_cross_validation(self, X: pd.DataFrame, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation evaluation"""

        logger.info(f"{cv}-fold 교차 검증 수행 중...")

        # Since we're using MultiOutputRegressor, we need to handle CV differently
        cv_scores = {}

        try:
            # Use negative MAE as scoring (scikit-learn convention)
            # n_jobs=1로 설정하여 메모리 누수 방지
            scores = cross_val_score(
                self.model, X, y,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=1
            )

            cv_scores['mae_scores'] = -scores  # Convert back to positive
            cv_scores['mae_mean'] = np.mean(-scores)
            cv_scores['mae_std'] = np.std(-scores)

            logger.info(f"교차 검증 MAE: {cv_scores['mae_mean']:,.0f} ± {cv_scores['mae_std']:,.0f}원")

        except Exception as e:
            logger.warning(f"교차 검증 실패: {e}")
            cv_scores = {'error': str(e)}

        return cv_scores

    def save_model(self, filepath: str) -> bool:
        """
        Save trained model to file

        Args:
            filepath: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            logger.warning("학습되지 않은 모델은 저장할 수 없습니다.")
            return False

        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Save model and metadata
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'feature_encoders': self.feature_encoders,
                'feature_scaler': self.feature_scaler,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics
            }

            joblib.dump(model_data, filepath)
            logger.info(f"모델 저장 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from file

        Args:
            filepath: Path to the saved model

        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_columns = model_data['feature_columns']
            self.target_columns = model_data['target_columns']
            self.feature_encoders = model_data['feature_encoders']
            self.feature_scaler = model_data['feature_scaler']
            self.is_trained = model_data['is_trained']
            self.training_metrics = model_data.get('training_metrics', {})
            self.validation_metrics = model_data.get('validation_metrics', {})

            logger.info(f"모델 로딩 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            return False

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (for tree-based models)

        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            return None

        try:
            # For MultiOutputRegressor, we need to average importance across outputs
            if hasattr(self.model.estimators_[0], 'feature_importances_'):
                importances = []
                for estimator in self.model.estimators_:
                    importances.append(estimator.feature_importances_)

                avg_importance = np.mean(importances, axis=0)

                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)

                return importance_df

        except Exception as e:
            logger.warning(f"Feature importance 추출 실패: {e}")

        return None

    def plot_performance(self, save_path: Optional[str] = None):
        """Plot model performance metrics"""

        if not self.training_metrics:
            logger.warning("성능 지표가 없습니다.")
            return

        # 폰트 재설정 (plot 함수 내에서)
        set_korean_font()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ML Model Performance Metrics', fontsize=16)

        # Extract metrics for plotting - 영어 카테고리 이름 사용
        categories_ko = ['재료비', '인건비', '임대료', '기타']
        categories_en = ['Materials', 'Labor', 'Rent', 'Others']

        train_mae = [self.training_metrics['by_category'][cat]['mae'] for cat in categories_ko]
        train_r2 = [self.training_metrics['by_category'][cat]['r2'] for cat in categories_ko]

        if self.validation_metrics:
            val_mae = [self.validation_metrics['by_category'][cat]['mae'] for cat in categories_ko]
            val_r2 = [self.validation_metrics['by_category'][cat]['r2'] for cat in categories_ko]
        else:
            val_mae = val_r2 = None

        # MAE by category
        axes[0, 0].bar(categories_en, train_mae, alpha=0.7, label='Training')
        if val_mae:
            axes[0, 0].bar(categories_en, val_mae, alpha=0.7, label='Validation')
        axes[0, 0].set_title('Mean Absolute Error by Category')
        axes[0, 0].set_ylabel('MAE (KRW)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # R² by category
        axes[0, 1].bar(categories_en, train_r2, alpha=0.7, label='Training')
        if val_r2:
            axes[0, 1].bar(categories_en, val_r2, alpha=0.7, label='Validation')
        axes[0, 1].set_title('R² Score by Category')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Feature importance (if available)
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            top_features = importance_df.head(10)
            axes[1, 0].barh(top_features['feature'], top_features['importance'])
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance')

        # Overall metrics comparison
        metrics_names = ['MAE', 'R²']
        train_overall = [self.training_metrics['overall']['mae'], self.training_metrics['overall']['r2']]

        if self.validation_metrics:
            val_overall = [self.validation_metrics['overall']['mae'], self.validation_metrics['overall']['r2']]

            x = np.arange(len(metrics_names))
            width = 0.35

            axes[1, 1].bar(x - width/2, train_overall, width, label='Training', alpha=0.7)
            axes[1, 1].bar(x + width/2, val_overall, width, label='Validation', alpha=0.7)
            axes[1, 1].set_title('Overall Performance Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(metrics_names)
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"성능 플롯 저장: {save_path}")
        else:
            # 기본 저장 경로 생성
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"training_results/model_performance_{timestamp}.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            logger.info(f"성능 플롯 저장: {default_path}")

        # plt.show()를 주석 처리하여 그래프가 화면에 표시되지 않도록 함
        plt.close()  # 메모리 정리


if __name__ == "__main__":
    # Example usage
    model = ExpensePredictionModel(model_type='randomforest')

    # Example prediction
    prediction = model.predict(
        revenue=8_000_000,
        industry_code="CS100001",  # 한식음식점
        region="11110"
    )

    print("예측 결과:")
    for category, amount in prediction.items():
        print(f"  {category}: {amount:,.0f}원")