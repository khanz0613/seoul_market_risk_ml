#!/usr/bin/env python3
"""
Ensemble ML Model Training for Seoul Market Risk Prediction
==========================================================

Builds and trains ensemble of ML models:
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- Neural Network (MLP)
- Support Vector Machine

Target Performance:
- Classification accuracy > 85%
- F1-Score > 0.80 (weighted)
- AUC-ROC > 0.90 per class
- Prediction time < 1 second

Author: Seoul Market Risk ML System
Date: 2025-09-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import warnings
import signal
from contextlib import contextmanager
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve
)
import joblib

# Install required libraries if needed
try:
    import xgboost as xgb
except ImportError:
    print("Installing XGBoost...")
    import subprocess
    subprocess.run(['pip', 'install', 'xgboost'], check=True)
    import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    print("Installing LightGBM...")
    import subprocess
    subprocess.run(['pip', 'install', 'lightgbm'], check=True)
    import lightgbm as lgb

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout operations"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class EnsembleModelTrainer:
    """Train and evaluate ensemble of ML models for risk classification"""

    def __init__(self, data_dir: str = "ml_preprocessed_data"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.model_performance = {}
        self.class_weights = {}
        self.feature_names = []

        self._load_preprocessed_data()
        self._quick_validation_check()

    def _load_preprocessed_data(self) -> None:
        """Load preprocessed training data"""
        print("📂 Loading preprocessed data...")

        try:
            # Load datasets
            self.train_data = pd.read_csv(self.data_dir / "train_data.csv")
            self.val_data = pd.read_csv(self.data_dir / "validation_data.csv")
            self.test_data = pd.read_csv(self.data_dir / "test_data.csv")

            # Separate features and labels
            self.X_train = self.train_data.drop('risk_label', axis=1)
            self.y_train = self.train_data['risk_label'] - 1  # Convert 1-5 to 0-4

            self.X_val = self.val_data.drop('risk_label', axis=1)
            self.y_val = self.val_data['risk_label'] - 1  # Convert 1-5 to 0-4

            self.X_test = self.test_data.drop('risk_label', axis=1)
            self.y_test = self.test_data['risk_label'] - 1  # Convert 1-5 to 0-4

            self.feature_names = list(self.X_train.columns)

            # Load and convert class weights to 0-4 indexing
            original_class_weights = joblib.load(self.data_dir / "class_weights.joblib")
            self.class_weights = {}
            for old_key, weight in original_class_weights.items():
                new_key = int(old_key - 1)  # Convert 1-5 keys to 0-4 keys (force Python int)
                self.class_weights[new_key] = weight

            print(f"✅ Training data: {self.X_train.shape}")
            print(f"✅ Validation data: {self.X_val.shape}")
            print(f"✅ Test data: {self.X_test.shape}")
            print(f"✅ Features: {len(self.feature_names)}")

        except FileNotFoundError as e:
            print(f"❌ Preprocessed data not found: {e}")
            print("Run efficient_data_preprocessing.py first!")
            raise

    def _quick_validation_check(self) -> None:
        """Quick checks to catch issues before long training"""
        print("🔍 Quick validation checks...")

        # Check for obvious data issues
        if self.X_train.isnull().any().any():
            print("❌ Training data has NaN values!")
            raise ValueError("Fix NaN values before training")

        if len(np.unique(self.y_train)) < 2:
            print("❌ Not enough target classes!")
            raise ValueError("Need at least 2 classes for classification")

        # Quick test with small fast model
        print("🧪 Testing with small model...")
        from sklearn.ensemble import RandomForestClassifier
        test_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

        try:
            # Test on small subset
            subset_size = min(1000, len(self.X_train))
            test_model.fit(self.X_train[:subset_size], self.y_train[:subset_size])
            test_pred = test_model.predict(self.X_val[:100])
            test_proba = test_model.predict_proba(self.X_val[:100])

            # Check for basic functionality
            if len(np.unique(test_pred)) < 2:
                print("⚠️ Model predicts only one class - check data balance")

            min_proba = np.min(test_proba)
            if min_proba < 1e-10:
                print(f"⚠️ Very small probabilities detected: {min_proba:.2e}")

            print("✅ Basic model training works")

        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            raise RuntimeError("Fix basic issues before full training")

    def build_random_forest(self) -> RandomForestClassifier:
        """Build Random Forest classifier"""
        print("🌳 Building Random Forest...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        return model

    def build_xgboost(self) -> xgb.XGBClassifier:
        """Build XGBoost classifier"""
        print("🚀 Building XGBoost...")

        # XGBoost now uses 0-4 classes like all other models
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        return model

    def build_lightgbm(self) -> lgb.LGBMClassifier:
        """Build LightGBM classifier"""
        print("💡 Building LightGBM...")

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        return model

    def build_neural_network(self) -> MLPClassifier:
        """Build Neural Network classifier"""
        print("🧠 Building Neural Network...")

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            verbose=False
        )

        return model

    def build_svm(self) -> Pipeline:
        """Build SVM classifier with scaling"""
        print("⚖️ Building SVM with feature scaling...")

        # SVM with proper scaling and optimized hyperparameters
        svm = SVC(
            C=0.1,  # Lower C for better generalization
            kernel='rbf',
            gamma='scale',
            class_weight=self.class_weights,
            probability=True,
            random_state=42,
            verbose=False,
            max_iter=5000  # Limit iterations to prevent infinite training
        )

        # Create pipeline with scaling
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', svm)
        ])

        return model

    def train_individual_models(self) -> Dict[str, Any]:
        """Train all individual models"""
        print("\n🎯 Training Individual Models")
        print("=" * 40)
        print("💡 TIP: Watch for warnings - stop early if issues detected")

        models_to_build = {
            'random_forest': self.build_random_forest,
            'xgboost': self.build_xgboost,
            'lightgbm': self.build_lightgbm,
            'neural_network': self.build_neural_network,
            'svm': self.build_svm
        }

        trained_models = {}

        for model_name, model_builder in models_to_build.items():
            print(f"\n🔄 Training {model_name}...")

            start_time = time.time()

            try:
                # Build model
                model = model_builder()

                # All models use unified 0-4 class system
                model.fit(self.X_train, self.y_train)

                # Validate model
                val_pred = model.predict(self.X_val)

                training_time = time.time() - start_time

                val_accuracy = accuracy_score(self.y_val, val_pred)
                val_f1 = f1_score(self.y_val, val_pred, average='weighted')

                # Prediction speed test
                pred_start = time.time()
                _ = model.predict(self.X_val[:1000])  # Test on 1000 samples
                pred_time_per_1k = time.time() - pred_start
                pred_time_per_sample = pred_time_per_1k / 1000

                trained_models[model_name] = model

                self.model_performance[model_name] = {
                    'training_time': training_time,
                    'validation_accuracy': val_accuracy,
                    'validation_f1': val_f1,
                    'prediction_time_per_sample': pred_time_per_sample
                }

                print(f"✅ {model_name} trained successfully")
                print(f"   Training time: {training_time:.1f}s")
                print(f"   Validation accuracy: {val_accuracy:.3f}")
                print(f"   Validation F1-score: {val_f1:.3f}")
                print(f"   Prediction time: {pred_time_per_sample*1000:.2f}ms per sample")

            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
                if "memory" in str(e).lower() or "timeout" in str(e).lower():
                    print(f"🚨 Critical error with {model_name} - consider skipping similar models")
                continue

        print(f"\n✅ Trained {len(trained_models)} individual models")
        self.models = trained_models
        return trained_models

    def build_voting_ensemble(self) -> VotingClassifier:
        """Build voting ensemble from trained models"""
        print("\n🤝 Building Voting Ensemble...")

        if len(self.models) < 3:
            print("❌ Need at least 3 trained models for ensemble")
            return None

        # Select best performing models for ensemble
        model_scores = [(name, perf['validation_f1']) for name, perf in self.model_performance.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # Use top 5 models or all if less than 5
        top_models = model_scores[:min(5, len(model_scores))]

        print("🎯 Ensemble models:")
        ensemble_estimators = []
        for model_name, f1_score in top_models:
            # 🔍 확률 진단을 위해 각 모델 체크
            model = self.models[model_name]
            if hasattr(model, 'predict_proba'):
                try:
                    # 작은 샘플로 확률 테스트
                    test_sample = self.X_val[:100]
                    sample_proba = model.predict_proba(test_sample)
                    min_proba = np.min(sample_proba)
                    zero_count = np.sum(sample_proba < 1e-10)

                    print(f"   {model_name}: F1={f1_score:.3f}, 최소확률={min_proba:.2e}, 확률0개수={zero_count}")

                    if zero_count > sample_proba.size * 0.01:  # >1% zero probabilities
                        print(f"      ⚠️ {model_name} 과도한 확률 0 ({zero_count}/{sample_proba.size}) - soft voting에서 제외")
                        continue

                except Exception as e:
                    print(f"   {model_name}: F1={f1_score:.3f} (확률 테스트 실패: {e})")
                    continue
            else:
                print(f"   {model_name}: F1={f1_score:.3f} (확률 미지원)")

            ensemble_estimators.append((model_name, self.models[model_name]))

        if len(ensemble_estimators) < 2:
            print("❌ 안정적인 모델이 부족해서 앙상블 생성 불가")
            return None

        # 확률 안정적인 모델만으로 soft voting 시도
        voting_ensemble = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft',  # 확률 0 모델 제외했으니 다시 soft voting 시도
            n_jobs=1  # 안정성 위해 단일 프로세스
        )

        return voting_ensemble

    def evaluate_models(self) -> Dict[str, Dict]:
        """Comprehensive evaluation of all models"""
        print("\n📊 Comprehensive Model Evaluation")
        print("=" * 45)

        evaluation_results = {}

        # Add ensemble to models if available
        if len(self.models) >= 3:
            ensemble = self.build_voting_ensemble()
            if ensemble is not None:
                print("🔄 Training voting ensemble...")
                try:
                    with timeout_context(300):  # 5 minute timeout
                        ensemble.fit(self.X_train, self.y_train)
                    self.models['voting_ensemble'] = ensemble
                    print("✅ Voting ensemble trained successfully")
                except TimeoutError:
                    print("⚠️ Voting ensemble training timed out (5 minutes)")
                    print("   Skipping ensemble evaluation...")
                except Exception as e:
                    print(f"⚠️ Voting ensemble training failed: {e}")
                    print("   Skipping ensemble evaluation...")

        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name}...")

            try:
                # All models use consistent 0-4 class system
                test_pred = model.predict(self.X_test)
                test_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None

                # 🔍 확률 진단 - 0 확률 체크
                if test_proba is not None:
                    min_proba = np.min(test_proba)
                    max_proba = np.max(test_proba)
                    zero_probas = np.sum(test_proba < 1e-10)
                    print(f"   🎲 확률 범위: {min_proba:.6f} ~ {max_proba:.6f}")
                    print(f"   ⚠️  확률 0인 예측: {zero_probas}개 ({zero_probas/(test_proba.size)*100:.1f}%)")

                    if zero_probas > 0:
                        print(f"   🚨 경고: {model_name} 모델이 확률 0 반환! 모델 훈련 문제 가능성")

                        # 각 클래스별 확률 분포 확인
                        for class_idx in range(test_proba.shape[1]):
                            class_zero_count = np.sum(test_proba[:, class_idx] < 1e-10)
                            if class_zero_count > 0:
                                print(f"      클래스 {class_idx}: {class_zero_count}개 확률 0")

                # Calculate metrics
                accuracy = accuracy_score(self.y_test, test_pred)
                f1_weighted = f1_score(self.y_test, test_pred, average='weighted')
                f1_macro = f1_score(self.y_test, test_pred, average='macro')

                # AUC-ROC (multiclass)
                if test_proba is not None:
                    try:
                        auc_score = roc_auc_score(self.y_test, test_proba, multi_class='ovr', average='weighted')
                    except:
                        auc_score = 0.0
                else:
                    auc_score = 0.0

                # Prediction speed on full test set
                start_time = time.time()
                _ = model.predict(self.X_test)
                prediction_time = time.time() - start_time

                evaluation_results[model_name] = {
                    'test_accuracy': accuracy,
                    'test_f1_weighted': f1_weighted,
                    'test_f1_macro': f1_macro,
                    'test_auc_roc': auc_score,
                    'prediction_time_full_test': prediction_time,
                    'samples_per_second': len(self.X_test) / prediction_time
                }

                print(f"✅ {model_name} Results:")
                print(f"   Accuracy: {accuracy:.3f}")
                print(f"   F1 (weighted): {f1_weighted:.3f}")
                print(f"   F1 (macro): {f1_macro:.3f}")
                print(f"   AUC-ROC: {auc_score:.3f}")
                print(f"   Prediction speed: {len(self.X_test) / prediction_time:.0f} samples/sec")

                # Check target achievements
                meets_accuracy = accuracy > 0.85
                meets_f1 = f1_weighted > 0.80
                meets_auc = auc_score > 0.90
                fast_prediction = (len(self.X_test) / prediction_time) > 1000  # >1000 samples/sec

                print(f"   🎯 Target Achievement:")
                print(f"      Accuracy >85%: {'✅' if meets_accuracy else '❌'} ({accuracy:.1%})")
                print(f"      F1-Score >80%: {'✅' if meets_f1 else '❌'} ({f1_weighted:.1%})")
                print(f"      AUC-ROC >90%: {'✅' if meets_auc else '❌'} ({auc_score:.1%})")
                print(f"      Fast prediction: {'✅' if fast_prediction else '❌'}")

            except Exception as e:
                print(f"❌ Evaluation failed for {model_name}: {e}")

        return evaluation_results

    def save_models_and_results(self, evaluation_results: Dict, output_dir: str = "trained_models") -> None:
        """Save trained models and evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n💾 Saving models and results to {output_dir}/...")

        # Save individual models
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            print(f"✅ Saved {model_name} model")

        # Save evaluation results
        results_summary = {
            'model_performance': self.model_performance,
            'evaluation_results': evaluation_results,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights
        }

        import json
        with open(output_path / "model_evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)

        # Create performance summary
        performance_summary = []
        for model_name, results in evaluation_results.items():
            performance_summary.append({
                'Model': model_name,
                'Accuracy': f"{results['test_accuracy']:.3f}",
                'F1-Score': f"{results['test_f1_weighted']:.3f}",
                'AUC-ROC': f"{results['test_auc_roc']:.3f}",
                'Speed (samples/sec)': f"{results['samples_per_second']:.0f}"
            })

        summary_df = pd.DataFrame(performance_summary)
        summary_df.to_csv(output_path / "model_performance_summary.csv", index=False)

        print(f"✅ Saved models and evaluation results")
        print(f"🎯 Model training complete!")

    def get_best_model(self, evaluation_results: Dict) -> Tuple[str, Any]:
        """Identify the best performing model"""
        best_score = 0
        best_model_name = None

        for model_name, results in evaluation_results.items():
            # Composite score: accuracy + f1_weighted + auc_roc
            composite_score = (
                results['test_accuracy'] * 0.4 +
                results['test_f1_weighted'] * 0.4 +
                results['test_auc_roc'] * 0.2
            )

            if composite_score > best_score:
                best_score = composite_score
                best_model_name = model_name

        if best_model_name:
            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"   Composite Score: {best_score:.3f}")
            return best_model_name, self.models[best_model_name]
        else:
            return None, None

def main():
    """Main ensemble training pipeline"""
    print("🚀 Ensemble ML Model Training Pipeline")
    print("=" * 50)

    trainer = EnsembleModelTrainer()

    try:
        # Step 1: Train individual models
        trained_models = trainer.train_individual_models()

        if len(trained_models) == 0:
            print("❌ No models trained successfully!")
            return

        # Step 2: Evaluate all models
        evaluation_results = trainer.evaluate_models()

        # Step 3: Identify best model
        best_name, best_model = trainer.get_best_model(evaluation_results)

        # Step 4: Save results
        trainer.save_models_and_results(evaluation_results)

        # Final summary
        print(f"\n🎯 Training Summary:")
        print(f"   Models trained: {len(trained_models)}")
        print(f"   Best model: {best_name}")

        # Check overall success
        any_meets_targets = False
        for model_name, results in evaluation_results.items():
            if (results['test_accuracy'] > 0.85 and
                results['test_f1_weighted'] > 0.80):
                any_meets_targets = True
                break

        if any_meets_targets:
            print("✅ Target performance achieved!")
        else:
            print("⚠️ Target performance not yet achieved - consider hyperparameter tuning")

        print(f"🎯 Ready for hyperparameter optimization!")

    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()