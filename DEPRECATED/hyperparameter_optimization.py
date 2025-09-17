#!/usr/bin/env python3
"""
Hyperparameter Optimization for Seoul Market Risk ML
===================================================

Uses Optuna for automatic hyperparameter tuning of the best performing models.
Since current performance already exceeds targets (99.7% accuracy vs 85% target),
this is for potential further optimization.

Author: Seoul Market Risk ML System
Date: 2025-09-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ML and optimization libraries
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer
import joblib

# Install and import Optuna
try:
    import optuna
except ImportError:
    print("Installing Optuna...")
    import subprocess
    subprocess.run(['pip', 'install', 'optuna'], check=True)
    import optuna

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class HyperparameterOptimizer:
    """Optimize hyperparameters for best-performing models"""

    def __init__(self, data_dir: str = "ml_preprocessed_data"):
        self.data_dir = Path(data_dir)
        self.best_params = {}
        self.optimization_results = {}
        self._load_data()

    def _load_data(self):
        """Load training and validation data"""
        print("📂 Loading data for hyperparameter optimization...")

        train_data = pd.read_csv(self.data_dir / "train_data.csv")
        val_data = pd.read_csv(self.data_dir / "validation_data.csv")

        # Use smaller sample for faster optimization
        sample_size = min(20000, len(train_data))
        train_sample = train_data.sample(n=sample_size, random_state=42)

        # Combine train and val for cross-validation
        combined_data = pd.concat([train_sample, val_data], ignore_index=True)

        self.X_combined = combined_data.drop('risk_label', axis=1)
        self.y_combined = combined_data['risk_label']

        print(f"✅ Optimization data: {len(self.X_combined):,} samples")

    def optimize_lightgbm(self, n_trials: int = 50) -> dict:
        """Optimize LightGBM hyperparameters"""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available")
            return {}

        print(f"🔍 Optimizing LightGBM ({n_trials} trials)...")

        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }

            model = lgb.LGBMClassifier(**params)

            # Use cross-validation for robust evaluation
            f1_scorer = make_scorer(f1_score, average='weighted')
            cv_scores = cross_val_score(model, self.X_combined, self.y_combined,
                                      cv=3, scoring=f1_scorer, n_jobs=1)

            return cv_scores.mean()

        # Run optimization
        study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        self.best_params['lightgbm'] = best_params
        self.optimization_results['lightgbm'] = {
            'best_score': best_score,
            'best_params': best_params,
            'n_trials': n_trials
        }

        print(f"✅ LightGBM optimization complete!")
        print(f"   Best F1-score: {best_score:.4f}")
        print(f"   Best parameters: {best_params}")

        return best_params

    def optimize_random_forest(self, n_trials: int = 30) -> dict:
        """Optimize Random Forest hyperparameters"""
        print(f"🌳 Optimizing Random Forest ({n_trials} trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42,
                'n_jobs': -1
            }

            model = RandomForestClassifier(**params)

            f1_scorer = make_scorer(f1_score, average='weighted')
            cv_scores = cross_val_score(model, self.X_combined, self.y_combined,
                                      cv=3, scoring=f1_scorer, n_jobs=1)

            return cv_scores.mean()

        study = optuna.create_study(direction='maximize', study_name='random_forest_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        self.best_params['random_forest'] = best_params
        self.optimization_results['random_forest'] = {
            'best_score': best_score,
            'best_params': best_params,
            'n_trials': n_trials
        }

        print(f"✅ Random Forest optimization complete!")
        print(f"   Best F1-score: {best_score:.4f}")

        return best_params

    def optimize_neural_network(self, n_trials: int = 20) -> dict:
        """Optimize Neural Network hyperparameters"""
        print(f"🧠 Optimizing Neural Network ({n_trials} trials)...")

        def objective(trial):
            # Define architecture
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_sizes = []

            for i in range(n_layers):
                size = trial.suggest_int(f'layer_{i}_size', 16, 128)
                hidden_sizes.append(size)

            params = {
                'hidden_layer_sizes': tuple(hidden_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.1),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'max_iter': 200,  # Keep limited for speed
                'random_state': 42
            }

            model = MLPClassifier(**params)

            f1_scorer = make_scorer(f1_score, average='weighted')
            cv_scores = cross_val_score(model, self.X_combined, self.y_combined,
                                      cv=3, scoring=f1_scorer, n_jobs=1)

            return cv_scores.mean()

        study = optuna.create_study(direction='maximize', study_name='neural_network_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        self.best_params['neural_network'] = best_params
        self.optimization_results['neural_network'] = {
            'best_score': best_score,
            'best_params': best_params,
            'n_trials': n_trials
        }

        print(f"✅ Neural Network optimization complete!")
        print(f"   Best F1-score: {best_score:.4f}")

        return best_params

    def train_optimized_models(self):
        """Train models with optimized hyperparameters"""
        print("\n🎯 Training Optimized Models")
        print("=" * 35)

        test_data = pd.read_csv(self.data_dir / "test_data.csv")
        X_test = test_data.drop('risk_label', axis=1)
        y_test = test_data['risk_label']

        optimized_models = {}
        optimized_results = {}

        for model_name, params in self.best_params.items():
            print(f"\n🔄 Training optimized {model_name}...")

            try:
                if model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMClassifier(**params)
                elif model_name == 'random_forest':
                    model = RandomForestClassifier(**params)
                elif model_name == 'neural_network':
                    # Filter out Optuna-specific parameters for MLPClassifier
                    mlp_params = {k: v for k, v in params.items()
                                 if k not in ['n_layers'] and not k.startswith('layer_')}
                    model = MLPClassifier(**mlp_params)
                else:
                    continue

                # Train on combined data
                start_time = time.time()
                model.fit(self.X_combined, self.y_combined)
                training_time = time.time() - start_time

                # Test evaluation
                test_pred = model.predict(X_test)
                test_f1 = f1_score(y_test, test_pred, average='weighted')

                optimized_models[model_name] = model
                optimized_results[model_name] = {
                    'test_f1_score': test_f1,
                    'training_time': training_time
                }

                print(f"✅ {model_name}: F1={test_f1:.4f} (training: {training_time:.1f}s)")

            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")

        return optimized_models, optimized_results

    def save_optimization_results(self, optimized_models, optimized_results, output_dir: str = "optimized_models"):
        """Save optimized models and results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n💾 Saving optimized results to {output_dir}/...")

        # Save optimized models
        for name, model in optimized_models.items():
            joblib.dump(model, output_path / f"optimized_{name}.joblib")

        # Save optimization details
        optimization_summary = {
            'best_parameters': self.best_params,
            'optimization_results': self.optimization_results,
            'final_model_results': optimized_results
        }

        import json
        with open(output_path / "optimization_results.json", 'w') as f:
            json.dump(optimization_summary, f, indent=2, default=str)

        print(f"✅ Saved optimized models and results")

    def run_optimization_pipeline(self):
        """Run complete hyperparameter optimization pipeline"""
        print("🚀 Hyperparameter Optimization Pipeline")
        print("=" * 45)
        print("ℹ️  Current models already exceed targets (99.7% accuracy vs 85%)")
        print("ℹ️  This optimization is for potential further improvements")

        start_time = time.time()

        # Optimize key models (focus on the best ones)
        if LIGHTGBM_AVAILABLE:
            self.optimize_lightgbm(n_trials=30)  # Focus on LightGBM since it was best

        self.optimize_random_forest(n_trials=20)
        self.optimize_neural_network(n_trials=15)

        optimization_time = time.time() - start_time

        # Train optimized models
        optimized_models, optimized_results = self.train_optimized_models()

        # Save results
        self.save_optimization_results(optimized_models, optimized_results)

        print(f"\n✅ Hyperparameter Optimization Complete!")
        print(f"   Total time: {optimization_time:.1f}s")
        print(f"   Optimized models: {len(optimized_models)}")

        return optimized_models, optimized_results

def main():
    """Main optimization pipeline"""
    print("Note: Models already exceed performance targets!")
    print("Current best: 99.7% accuracy vs 85% target")
    print("Running optimization for potential improvements...\n")

    optimizer = HyperparameterOptimizer()
    optimizer.run_optimization_pipeline()

if __name__ == "__main__":
    main()